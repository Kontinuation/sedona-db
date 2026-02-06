use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_GDAL_VERSION_NUMBER");

    let version_str = env::var("DEP_GDAL_VERSION_NUMBER")
        .expect("sedona-raster-gdal requires gdal-sys to emit DEP_GDAL_VERSION_NUMBER");
    let version_num: u64 = version_str
        .parse()
        .expect("DEP_GDAL_VERSION_NUMBER must be a numeric GDAL version");

    let major = version_num / 1_000_000;
    let minor = (version_num / 10_000) % 100;
    let patch = (version_num / 100) % 100;

    if version_num < 3_070_000 {
        panic!(
            "sedona-raster-gdal requires GDAL >= 3.7.0 when raster-gdal feature is enabled. Found {major}.{minor}.{patch}"
        );
    }
}
