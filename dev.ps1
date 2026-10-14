# Runs a cargo command inside the Linux dev container.
#
# The host is Windows, but the C dependencies (libjpeg-turbo, Little-CMS, and later the
# forked libopenslide) build cleanly only on Linux, and Linux is the deployment target — so
# all building and testing happens in a container. Named volumes keep the target directory
# and the crate registry warm between runs.
#
#   ./dev.ps1 check --workspace
#   ./dev.ps1 test -p wsi-core
#   ./dev.ps1 clippy --workspace -- -D warnings
param([Parameter(ValueFromRemainingArguments = $true)] $CargoArgs)

$ErrorActionPreference = "Stop"
$src = $PSScriptRoot

docker volume create wsi-rs-target | Out-Null
docker volume create wsi-rs-registry | Out-Null

docker run --rm `
    -v "${src}:/work" `
    -v wsi-rs-target:/target `
    -v wsi-rs-registry:/usr/local/cargo/registry `
    -w /work `
    wsi-rs-dev `
    cargo @CargoArgs

exit $LASTEXITCODE
