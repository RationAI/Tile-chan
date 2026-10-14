//! Locates libopenslide.
//!
//! OpenSlide 4.0 is the floor: the ICC entry points this plugin needs
//! (`openslide_get_icc_profile_size`, `openslide_read_icc_profile` and their
//! associated-image counterparts) were introduced there, and the Python service depends on
//! them through `openslide-python`'s `color_profile`. Debian bookworm ships 3.4.1, so the
//! Docker images build it from source — see `rust/Dockerfile`.

fn main() {
    println!("cargo:rerun-if-env-changed=OPENSLIDE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=OPENSLIDE_INCLUDE_DIR");

    // An explicit location wins, for builds against a tree that pkg-config cannot see.
    if let Ok(dir) = std::env::var("OPENSLIDE_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=openslide");
        return;
    }

    match pkg_config::Config::new()
        .atleast_version("4.0.0")
        .probe("openslide")
    {
        Ok(_) => {}
        Err(err) => {
            // Fail loudly rather than emitting a link flag that produces a confusing
            // undefined-symbol error later.
            panic!(
                "could not find libopenslide >= 4.0.0 via pkg-config: {err}\n\
                 Install it, or set OPENSLIDE_LIB_DIR to the directory holding \
                 libopenslide.so. Versions below 4.0 lack the ICC profile API this plugin \
                 requires."
            );
        }
    }
}
