# Production image for the Rust WSI-Service.
#
# Two stages: a builder with the C toolchain the *-sys crates need, and a slim runtime
# carrying only the shared libraries the binary actually links against.
#
# The C dependencies are deliberate, not incidental — they are what makes output match the
# Python service:
#
#   libjpeg-turbo  built from source by turbojpeg-sys; the same encoder Pillow uses, so
#                  JPEG quality and subsampling produce the same bitstream
#   Little-CMS     vendored by lcms2-sys; the same library behind PIL.ImageCms
#   libopenslide   built from the same fork the Python image uses, so the two services
#                  share a decoder and region pixels match by construction

FROM rust:1-bookworm AS builder

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        cmake \
        nasm \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# --- libopenslide -------------------------------------------------------------------------
#
# Bookworm ships 3.4.1, which predates the ICC profile API added in 4.0 and reached through
# `openslide-python`'s `color_profile`. Built from the fork the Python image builds, so both
# link the same decoder. libdicom is not packaged for bookworm, but openslide's meson vendors
# it as a subproject, so the DICOM backend ends up enabled here as it does there.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        git \
        meson \
        ninja-build \
        zlib1g-dev \
        libzstd-dev \
        libpng-dev \
        libjpeg-dev \
        libtiff-dev \
        libopenjp2-7-dev \
        libgdk-pixbuf-2.0-dev \
        libxml2-dev \
        sqlite3 \
        libsqlite3-dev \
        libcairo2-dev \
        libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 https://github.com/iewchen/openslide.git /tmp/openslide \
    && meson setup /tmp/openslide/builddir /tmp/openslide \
        --prefix=/usr/local --buildtype=release \
    && meson compile -C /tmp/openslide/builddir \
    && meson install -C /tmp/openslide/builddir \
    && ldconfig \
    && rm -rf /tmp/openslide

ENV PKG_CONFIG_PATH=/usr/local/lib/x86_64-linux-gnu/pkgconfig:/usr/local/lib/pkgconfig

WORKDIR /build

# Dependency compilation is cached separately from the source, so an edit to a handler does
# not rebuild the whole crate graph. Every workspace member's manifest has to be present or
# cargo refuses to resolve, even though only wsi-server is built into the image.
COPY Cargo.toml Cargo.lock ./
COPY crates/wsi-core/Cargo.toml crates/wsi-core/
COPY crates/wsi-mappers/Cargo.toml crates/wsi-mappers/
COPY crates/wsi-auth/Cargo.toml crates/wsi-auth/
COPY crates/wsi-plugin-pil/Cargo.toml crates/wsi-plugin-pil/
COPY crates/wsi-plugin-openslide/Cargo.toml crates/wsi-plugin-openslide/
COPY crates/wsi-plugin-nifti/Cargo.toml crates/wsi-plugin-nifti/
COPY crates/wsi-plugin-mvt/Cargo.toml crates/wsi-plugin-mvt/
COPY crates/wsi-plugin-tifffile/Cargo.toml crates/wsi-plugin-tifffile/
COPY crates/wsi-plugin-tiffslide/Cargo.toml crates/wsi-plugin-tiffslide/
COPY crates/wsi-plugin-wsidicom/Cargo.toml crates/wsi-plugin-wsidicom/
COPY crates/wsi-plugin-openslide/build.rs crates/wsi-plugin-openslide/
COPY crates/wsi-server/Cargo.toml crates/wsi-server/
COPY crates/wsi-parity/Cargo.toml crates/wsi-parity/
RUN mkdir -p crates/wsi-core/src crates/wsi-mappers/src crates/wsi-auth/src \
             crates/wsi-plugin-pil/src crates/wsi-plugin-openslide/src \
             crates/wsi-plugin-nifti/src crates/wsi-plugin-mvt/src \
             crates/wsi-plugin-tifffile/src crates/wsi-plugin-tiffslide/src \
             crates/wsi-plugin-wsidicom/src \
             crates/wsi-server/src crates/wsi-parity/src \
    && touch crates/wsi-core/src/lib.rs crates/wsi-mappers/src/lib.rs \
             crates/wsi-auth/src/lib.rs crates/wsi-plugin-pil/src/lib.rs \
             crates/wsi-plugin-openslide/src/lib.rs crates/wsi-plugin-nifti/src/lib.rs \
             crates/wsi-plugin-mvt/src/lib.rs crates/wsi-plugin-tifffile/src/lib.rs \
             crates/wsi-plugin-tiffslide/src/lib.rs crates/wsi-plugin-wsidicom/src/lib.rs \
             crates/wsi-server/src/lib.rs \
    && echo 'fn main() {}' > crates/wsi-server/src/main.rs \
    && echo 'fn main() {}' > crates/wsi-parity/src/main.rs \
    && cargo build --release --locked -p wsi-server --features openslide || true

COPY assets/ assets/
COPY crates/ crates/
# Touch the real sources so cargo notices them after the stub build above.
RUN find crates -name '*.rs' -exec touch {} + \
    && cargo build --release --locked -p wsi-server --features openslide


FROM debian:bookworm-slim AS runtime

# libjpeg-turbo is compiled into the binary. Little-CMS and libopenslide's dependency chain
# are shared libraries and have to be present at runtime; `ldd target/release/wsi-server` in
# the builder stage is the check if this list ever needs revisiting.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        liblcms2-2 \
        libglib2.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libxml2 \
        libsqlite3-0 \
        libtiff6 \
        libopenjp2-7 \
        libpng16-16 \
        libjpeg62-turbo \
        libzstd1 \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

# libopenslide, plus the libdicom it vendors as a subproject.
COPY --from=builder /usr/local/lib/x86_64-linux-gnu/libopenslide.so* /usr/local/lib/x86_64-linux-gnu/
COPY --from=builder /usr/local/lib/x86_64-linux-gnu/libdicom.so* /usr/local/lib/x86_64-linux-gnu/
RUN ldconfig

COPY --from=builder /build/target/release/wsi-server /usr/local/bin/wsi-server

USER appuser
WORKDIR /home/appuser

ENV WS_DATA_DIR=/data
EXPOSE 8080/tcp

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/wsi-server"]

LABEL org.opencontainers.image.source="https://github.com/rationAI/WSI-Service"
