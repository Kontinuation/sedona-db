// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
use crate::{
    aggregate_udf::{IntoSedonaAccumulatorRefs, SedonaAggregateUDF},
    scalar_udf::{classify_kernel_async_mode, IntoScalarKernelRefs, SedonaScalarUDF},
};
use datafusion_common::error::Result;
use datafusion_expr::{AggregateUDFImpl, ScalarUDFImpl};
use std::collections::HashMap;

/// Helper for managing groups of functions
///
/// Sedona coordinates the assembly of a large number of spatial functions with potentially
/// different sets of dependencies (e.g., geography vs. geometry), multiple implementations,
/// and/or implementations that live in different crates. This structure helps coordinate
/// these implementations.
pub struct FunctionSet {
    scalar_udfs: HashMap<String, SedonaScalarUDF>,
    aggregate_udfs: HashMap<String, SedonaAggregateUDF>,
}

impl FunctionSet {
    /// Create a new, empty FunctionSet
    pub fn new() -> Self {
        Self {
            scalar_udfs: HashMap::new(),
            aggregate_udfs: HashMap::new(),
        }
    }

    /// Iterate over references to all [SedonaScalarUDF]s
    pub fn scalar_udfs(&self) -> impl Iterator<Item = &SedonaScalarUDF> + '_ {
        self.scalar_udfs.values()
    }

    /// Return a reference to the function corresponding to the name
    pub fn scalar_udf(&self, name: &str) -> Option<&SedonaScalarUDF> {
        self.scalar_udfs.get(name)
    }

    /// Return a mutable reference to the function corresponding to the name
    pub fn scalar_udf_mut(&mut self, name: &str) -> Option<&mut SedonaScalarUDF> {
        self.scalar_udfs.get_mut(name)
    }

    /// Insert a new ScalarUDF and return the UDF that had previously been added, if any
    pub fn insert_scalar_udf(&mut self, udf: SedonaScalarUDF) -> Option<SedonaScalarUDF> {
        self.scalar_udfs.insert(udf.name().to_string(), udf)
    }

    /// Iterate over references to all [SedonaAggregateUDF]s
    pub fn aggregate_udfs(&self) -> impl Iterator<Item = &SedonaAggregateUDF> + '_ {
        self.aggregate_udfs.values()
    }

    /// Return a reference to the aggregate function corresponding to the name
    pub fn aggregate_udf(&self, name: &str) -> Option<&SedonaAggregateUDF> {
        self.aggregate_udfs.get(name)
    }

    /// Return a mutable reference to the aggregate function corresponding to the name
    pub fn aggregate_udf_mut(&mut self, name: &str) -> Option<&mut SedonaAggregateUDF> {
        self.aggregate_udfs.get_mut(name)
    }

    /// Insert a new AggregateUDF and return the UDF that had previously been added, if any
    pub fn insert_aggregate_udf(&mut self, udf: SedonaAggregateUDF) -> Option<SedonaAggregateUDF> {
        self.aggregate_udfs.insert(udf.name().to_string(), udf)
    }

    /// Consume another function set and merge its contents into this one
    pub fn merge(&mut self, other: FunctionSet) {
        for (_k, v) in other.scalar_udfs.into_iter() {
            self.insert_scalar_udf(v);
        }

        for (k, v) in other.aggregate_udfs.into_iter() {
            self.aggregate_udfs.insert(k, v);
        }
    }

    /// Add a kernel to a function in this set
    ///
    /// This adds a scalar UDF with immutable output if a function of that name does not
    /// exist in this set. A reference to the matching or inserted function is returned.
    pub fn add_scalar_udf_impl(
        &mut self,
        name: &str,
        kernels: impl IntoScalarKernelRefs,
    ) -> Result<&SedonaScalarUDF> {
        let kernels = kernels.into_scalar_kernel_refs();
        let incoming_is_async = classify_kernel_async_mode(&kernels)?;
        if let Some(function) = self.scalar_udf_mut(name) {
            if function.is_async() != incoming_is_async {
                return datafusion_common::internal_err!(
                    "{name}: cannot mix sync and async kernels in one SedonaScalarUDF"
                );
            }
            function.add_kernels(kernels);
        } else {
            let function = if incoming_is_async {
                SedonaScalarUDF::from_async_impl(name, kernels, None)
            } else {
                SedonaScalarUDF::from_impl(name, kernels)
            };
            self.insert_scalar_udf(function);
        }

        Ok(self.scalar_udf(name).unwrap())
    }

    /// Add an aggregate kernel to a function in this set
    ///
    /// This adds an aggregate UDF with immutable output if a function of that name does not
    /// exist in this set. A reference to the matching or inserted function is returned.
    pub fn add_aggregate_udf_kernel(
        &mut self,
        name: &str,
        kernel: impl IntoSedonaAccumulatorRefs,
    ) -> Result<&SedonaAggregateUDF> {
        if let Some(function) = self.aggregate_udf_mut(name) {
            function.add_kernel(kernel);
        } else {
            let function = SedonaAggregateUDF::from_impl(name, kernel);
            self.insert_aggregate_udf(function);
        }

        Ok(self.aggregate_udf(name).unwrap())
    }
}

impl Default for FunctionSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc};

    use arrow_schema::{DataType, FieldRef};
    use datafusion_common::{not_impl_err, scalar::ScalarValue};

    use datafusion_expr::{Accumulator, ColumnarValue, Volatility};
    use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

    use crate::{
        aggregate_udf::{SedonaAccumulator, SedonaAccumulatorRef},
        scalar_udf::{AsyncSedonaScalarKernel, SimpleSedonaScalarKernel},
    };

    use super::*;

    #[test]
    fn function_set() {
        let mut functions = FunctionSet::new();
        assert_eq!(functions.scalar_udfs().collect::<Vec<_>>().len(), 0);
        assert!(functions.scalar_udf("simple_udf").is_none());
        assert!(functions.scalar_udf_mut("simple_udf").is_none());

        let kernel = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_arrow(DataType::Boolean)],
                SedonaType::Arrow(DataType::Boolean),
            ),
            Arc::new(|_, _| Ok(ColumnarValue::Scalar(ScalarValue::Boolean(None)))),
        );

        let udf = SedonaScalarUDF::new("simple_udf", vec![kernel.clone()], Volatility::Immutable);

        functions.insert_scalar_udf(udf);
        assert_eq!(functions.scalar_udfs().collect::<Vec<_>>().len(), 1);
        assert!(functions.scalar_udf("simple_udf").is_some());
        assert!(functions.scalar_udf_mut("simple_udf").is_some());
        assert_eq!(
            functions
                .add_scalar_udf_impl("simple_udf", kernel.clone())
                .unwrap()
                .name(),
            "simple_udf"
        );
        let inserted_udf = functions
            .add_scalar_udf_impl("function that does not yet exist", kernel.clone())
            .unwrap();
        assert_eq!(inserted_udf.name(), "function that does not yet exist");

        let kernel2 = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_arrow(DataType::Utf8)],
                SedonaType::Arrow(DataType::Utf8),
            ),
            Arc::new(|_, _| Ok(ColumnarValue::Scalar(ScalarValue::Utf8(None)))),
        );

        let udf2 = SedonaScalarUDF::new("simple_udf2", vec![kernel2], Volatility::Immutable);
        let mut functions2 = FunctionSet::new();
        functions2.insert_scalar_udf(udf2);
        functions.merge(functions2);
        assert_eq!(
            functions
                .scalar_udfs()
                .map(|s| s.name())
                .collect::<HashSet<_>>(),
            vec![
                "simple_udf",
                "simple_udf2",
                "function that does not yet exist"
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        );
    }

    #[test]
    fn function_set_with_async_scalar_udfs() {
        let mut functions = FunctionSet::new();

        let sync_udf = SedonaScalarUDF::new(
            "sync_udf",
            vec![SimpleSedonaScalarKernel::new_ref(
                ArgMatcher::new(
                    vec![ArgMatcher::is_arrow(DataType::Boolean)],
                    SedonaType::Arrow(DataType::Boolean),
                ),
                Arc::new(|_, _| Ok(ColumnarValue::Scalar(ScalarValue::Boolean(None)))),
            )],
            Volatility::Immutable,
        );
        functions.insert_scalar_udf(sync_udf);

        #[derive(Debug)]
        struct AsyncTestKernel;

        #[async_trait::async_trait]
        impl AsyncSedonaScalarKernel for AsyncTestKernel {
            fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
                if args == [SedonaType::Arrow(DataType::Utf8)] {
                    Ok(Some(SedonaType::Arrow(DataType::Utf8)))
                } else {
                    Ok(None)
                }
            }

            async fn invoke_async_batch_from_args(
                &self,
                _arg_types: &[SedonaType],
                args: &[ColumnarValue],
                _return_type: &SedonaType,
                _num_rows: usize,
                _config_options: Option<&datafusion_common::config::ConfigOptions>,
            ) -> Result<ColumnarValue> {
                Ok(args[0].clone())
            }
        }

        let async_udf = SedonaScalarUDF::from_async_impl("async_udf", AsyncTestKernel, Some(5));
        functions.insert_scalar_udf(async_udf);

        assert_eq!(functions.scalar_udfs().count(), 2);
        assert!(!functions.scalar_udf("sync_udf").unwrap().is_async());
        assert!(functions.scalar_udf("async_udf").unwrap().is_async());
        assert_eq!(
            functions
                .scalar_udf("async_udf")
                .unwrap()
                .ideal_async_batch_size(),
            Some(5)
        );
        assert!(functions
            .scalar_udf("async_udf")
            .unwrap()
            .to_datafusion_udf()
            .as_async()
            .is_some());
    }

    #[test]
    fn function_set_rejects_mixed_sync_async_kernels() {
        let mut functions = FunctionSet::new();

        let sync_kernel = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(
                vec![ArgMatcher::is_arrow(DataType::Boolean)],
                SedonaType::Arrow(DataType::Boolean),
            ),
            Arc::new(|_, _| Ok(ColumnarValue::Scalar(ScalarValue::Boolean(None)))),
        );

        #[derive(Debug)]
        struct AsyncBoolKernel;

        #[async_trait::async_trait]
        impl AsyncSedonaScalarKernel for AsyncBoolKernel {
            fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
                if args == [SedonaType::Arrow(DataType::Boolean)] {
                    Ok(Some(SedonaType::Arrow(DataType::Boolean)))
                } else {
                    Ok(None)
                }
            }

            async fn invoke_async_batch_from_args(
                &self,
                _arg_types: &[SedonaType],
                args: &[ColumnarValue],
                _return_type: &SedonaType,
                _num_rows: usize,
                _config_options: Option<&datafusion_common::config::ConfigOptions>,
            ) -> Result<ColumnarValue> {
                Ok(args[0].clone())
            }
        }

        functions.add_scalar_udf_impl("mixed", sync_kernel).unwrap();
        let err = functions
            .add_scalar_udf_impl("mixed", AsyncBoolKernel)
            .unwrap_err();

        assert!(err
            .message()
            .contains("cannot mix sync and async kernels in one SedonaScalarUDF"));
    }

    #[derive(Debug, Clone)]
    struct TestAccumulator {}

    impl SedonaAccumulator for TestAccumulator {
        fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>> {
            not_impl_err!("")
        }

        fn accumulator(
            &self,
            _args: &[SedonaType],
            _output_type: &SedonaType,
        ) -> Result<Box<dyn Accumulator>> {
            not_impl_err!("")
        }

        fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
            not_impl_err!("")
        }
    }

    #[test]
    fn function_set_with_aggregates() {
        let mut functions = FunctionSet::new();
        assert_eq!(functions.scalar_udfs().collect::<Vec<_>>().len(), 0);
        assert!(functions.aggregate_udf("simple_udaf").is_none());
        assert!(functions.aggregate_udf_mut("simple_udaf").is_none());

        let udaf = SedonaAggregateUDF::new(
            "simple_udaf",
            Vec::<SedonaAccumulatorRef>::new(),
            Volatility::Immutable,
        );
        let kernel = TestAccumulator {};

        functions.insert_aggregate_udf(udaf);
        assert_eq!(functions.aggregate_udfs().collect::<Vec<_>>().len(), 1);
        assert!(functions.aggregate_udf("simple_udaf").is_some());
        assert!(functions.aggregate_udf_mut("simple_udaf").is_some());
        assert_eq!(
            functions
                .add_aggregate_udf_kernel("simple_udaf", kernel.clone())
                .unwrap()
                .name(),
            "simple_udaf"
        );
        let added_func = functions
            .add_aggregate_udf_kernel("function that does not exist yet", kernel.clone())
            .unwrap();
        assert_eq!(added_func.name(), "function that does not exist yet");

        let udaf2 = SedonaAggregateUDF::new(
            "simple_udaf2",
            vec![Arc::new(kernel.clone())],
            Volatility::Immutable,
        );
        let mut functions2 = FunctionSet::new();
        functions2.insert_aggregate_udf(udaf2);
        functions.merge(functions2);
        assert_eq!(
            functions
                .aggregate_udfs()
                .map(|s| s.name())
                .collect::<HashSet<_>>(),
            vec![
                "simple_udaf",
                "simple_udaf2",
                "function that does not exist yet"
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        );
    }
}
