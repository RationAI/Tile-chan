# Runs the differential harness from inside the Linux dev container.
#
# Both services must already be reachable from the host. `host.docker.internal` is mapped
# so the container can reach ports published on the host.
#
# `-Context` is required against a `PathsMapper` service: `/v3/cases/` answers
# `404 context is required` without one, and `.` is the whole data directory.
#
# Two configurations are worth running. The default one, where both services use
# openslide for the pyramidal slides, has to be exact:
#
#   ./parity.ps1 -Context . -Report parity-report.json
#   ./parity.ps1 -Context . -LimitSlides 2        # shake out a new setup first
#
# The second disables openslide on both services (`WS_PLUGIN_PRIORITY_OPENSLIDE=-1`),
# which is the only way the matrix reaches `tiffslide` -- it never asks for a plugin by
# name. That configuration does not share a decoder, so it needs the measured allowance
# of divergence 28:
#
#   ./parity.ps1 -Context . -LosslessDelta 1 -JpegDelta 16
param(
    [string] $PythonUrl = "http://host.docker.internal:8081",
    [string] $RustUrl = "http://host.docker.internal:8080",
    [string] $Context = "",
    [string] $Report = "",
    # Cap the run while shaking out a new setup: a misconfigured data directory or a
    # plugin-selection mismatch is far easier to read over two slides than over twenty.
    [int] $LimitSlides = 0,
    # Per-channel allowances. Both default to the harness's own defaults, which require
    # lossless formats to agree exactly; see the note above before raising either.
    [int] $LosslessDelta = -1,
    [int] $JpegDelta = -1
)

$ErrorActionPreference = "Stop"
$src = $PSScriptRoot

docker volume create wsi-rs-target | Out-Null
docker volume create wsi-rs-registry | Out-Null

$harnessArgs = @("--python-url", $PythonUrl, "--rust-url", $RustUrl)
if ($Context) { $harnessArgs += @("--context", $Context) }
if ($Report) { $harnessArgs += @("--report", $Report) }
if ($LimitSlides -gt 0) { $harnessArgs += @("--limit-slides", "$LimitSlides") }
if ($LosslessDelta -ge 0) { $harnessArgs += @("--lossless-delta", "$LosslessDelta") }
if ($JpegDelta -ge 0) { $harnessArgs += @("--jpeg-delta", "$JpegDelta") }

docker run --rm `
    -v "${src}:/work" `
    -v wsi-rs-target:/target `
    -v wsi-rs-registry:/usr/local/cargo/registry `
    -w /work `
    -e CARGO_TARGET_DIR=/target `
    --add-host=host.docker.internal:host-gateway `
    wsi-rs-dev `
    cargo run --quiet -p wsi-parity -- @harnessArgs

exit $LASTEXITCODE
