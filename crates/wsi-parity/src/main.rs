//! Differential test harness: runs the same requests against the Python and Rust services
//! and reports every difference.
//!
//! ```sh
//! wsi-parity --python-url http://localhost:8081 --rust-url http://localhost:8080 \
//!            --context . --report parity-report.json
//! ```
//!
//! Slides are discovered from the services themselves, so the harness needs no knowledge
//! of the data directory. Differences that are known and documented are allowlisted by
//! label prefix; everything else fails the run.

mod compare;
mod matrix;

use std::collections::BTreeMap;
use std::process::ExitCode;

use compare::{Response, Verdict};
use matrix::{Request, SlideInfo};

/// One expected difference.
struct Allowed {
    /// Matched against the start of the request's label.
    label: &'static str,
    /// Matched as a substring of the request *path*, which carries the encoded slide id.
    /// Empty means every request.
    ///
    /// Scoping matters: divergence 38 is only reachable on a slide whose plugin caches its
    /// associated images, so excusing `label:` everywhere would hide a genuine difference
    /// on the slides where that endpoint has to agree exactly.
    path_contains: &'static str,
    /// Why, naming the divergence it covers.
    reason: &'static str,
}

/// Differences that are expected.
///
/// Every entry names the divergence it covers. Adding one without a corresponding entry in
/// `docs/divergences.md` defeats the point of the harness.
const ALLOWLIST: &[Allowed] = &[
    Allowed {
        label: "files:label",
        path_contains: "",
        reason: "divergence 2: Python never resizes /files/label, the port does",
    },
    Allowed {
        label: "batch:label",
        path_contains: "",
        reason: "divergence 2: Python never resizes /batch/label, the port does",
    },
    Allowed {
        label: "files:macro",
        path_contains: "",
        reason: "divergence 2: Python never resizes /files/macro, the port does",
    },
    Allowed {
        label: "batch:macro",
        path_contains: "",
        reason: "divergence 2: Python never resizes /batch/macro, the port does",
    },
    Allowed {
        label: "openapi:",
        path_contains: "",
        reason: "the OpenAPI document is hand-written and is not expected to match FastAPI's byte for byte",
    },
    // Not a plugin *set* difference — both services report the same seven plugins at the
    // same versions and priorities, and this entry would hide it if they did not, because
    // the comparison is positional. What differs is the order of the two plugins that tie
    // at priority 0: Python's is `pkgutil.iter_modules()` discovery order, which is
    // filesystem-dependent, and the port's is by name.
    Allowed {
        label: "alive",
        path_contains: "",
        reason: "divergence 24: equal-priority plugins are ordered by name here and by discovery order in Python",
    },
    // Scoped to DICOM, because that is where it is reachable. `/slides/label` and
    // `/slides/macro` resize in place, and `wsidicom` hands back a *cached* image, so each
    // request shrinks it permanently: Python's answer depends on every earlier request and
    // converges toward 1x1. `openslide` decodes a fresh image per access and recovers,
    // which is why these labels must still match exactly on every other slide.
    Allowed {
        label: "label",
        path_contains: "dicom",
        reason: "divergence 38: Python shrinks the cached label on every request; the port returns the same image each time",
    },
    Allowed {
        label: "macro",
        path_contains: "dicom",
        reason: "divergence 38: Python shrinks the cached macro on every request; the port returns the same image each time",
    },
    Allowed {
        label: "files:thumbnail",
        path_contains: "dicom",
        reason: "divergence 38: the batch thumbnail reads the same cached image the shrunken label left behind",
    },
    Allowed {
        label: "batch:thumbnail",
        path_contains: "dicom",
        reason: "divergence 38: the batch thumbnail reads the same cached image the shrunken label left behind",
    },
    // A directory slide's archive has more than one entry, so their order is observable.
    // Python's `expand_folders` de-duplicates with `set()`, which makes it arbitrary; the
    // port sorts. A single-file slide has one entry and still has to match exactly.
    Allowed {
        label: "download",
        path_contains: "dicom",
        reason: "divergence 20: Python's expand_folders orders a directory's files by set                  iteration, the port sorts them",
    },
];

struct Config {
    python_url: String,
    rust_url: String,
    context: Option<String>,
    report: Option<String>,
    limit_slides: usize,
    tolerance: compare::Tolerance,
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config {
        python_url: "http://localhost:8081".into(),
        rust_url: "http://localhost:8080".into(),
        context: None,
        report: None,
        limit_slides: usize::MAX,
        tolerance: compare::Tolerance::default(),
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} requires a value"));
        match arg.as_str() {
            "--python-url" => config.python_url = value()?,
            "--rust-url" => config.rust_url = value()?,
            "--context" => config.context = Some(value()?),
            "--report" => config.report = Some(value()?),
            "--limit-slides" => {
                config.limit_slides = value()?
                    .parse()
                    .map_err(|_| "--limit-slides expects a number".to_string())?
            }
            "--lossless-delta" => {
                config.tolerance.lossless = value()?
                    .parse()
                    .map_err(|_| "--lossless-delta expects a number".to_string())?
            }
            "--jpeg-delta" => {
                config.tolerance.jpeg = value()?
                    .parse()
                    .map_err(|_| "--jpeg-delta expects a number".to_string())?
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }

    Ok(config)
}

fn print_usage() {
    println!(
        r#"wsi-parity — compare the Rust WSI-Service against the Python one

Options:
  --python-url <url>   default http://localhost:8081
  --rust-url <url>     default http://localhost:8080
  --context <path>     context for a context-dependent mapper (PathsMapper)
  --report <file>      write a JSON report
  --limit-slides <n>   compare at most n slides
  --lossless-delta <n> allowed per-channel difference on png/bmp/tiff. Default 0:
                       both services share a decoder, so every sample has to agree.
                       Raise it only for a configuration where they do not — with
                       openslide disabled, tiffslide serves the pyramidal slides and
                       its JPEG decoder differs by one per channel (divergence 28).
  --jpeg-delta <n>     allowed per-channel difference on jpeg (default 1). The
                       encoder amplifies whatever it was fed, so a one-per-channel
                       decoder difference reaches sixteen over a large region; the
                       lossless formats are what actually guard the pixels."#
    );
}

/// How much of a response body to hold in memory.
///
/// `/v3/slides/download` streams the slide file itself, so for a real slide the body runs
/// to hundreds of megabytes — buffering two of those is how the first run failed with
/// "error decoding response body". A capped read is enough: what that endpoint has to get
/// right is the status, the headers and the ZIP entry it opens with, and the file bytes it
/// then copies are the same bytes the region and tile comparisons already check.
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

async fn fetch(client: &reqwest::Client, base: &str, path: &str) -> Result<Response, String> {
    let response = client
        .get(format!("{}{}", base.trim_end_matches('/'), path))
        .send()
        .await
        .map_err(|e| format!("GET {path} failed: {e}"))?;

    let status = response.status().as_u16();
    let header = |name: &str| {
        response
            .headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    };
    let content_type = header("content-type").unwrap_or_default();
    // This client does not advertise `Accept-Encoding`, so a `content-encoding` header
    // means the endpoint compressed unconditionally — which the passthrough endpoints do.
    // The header is compared as-is and the body is decoded below, because two gzip
    // encoders agreeing byte for byte is not something either service promises.
    let content_encoding = header("content-encoding");

    let mut body = Vec::new();
    let mut truncated = false;
    {
        use futures::StreamExt;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("reading {path} failed: {e}"))?;
            if body.len() + chunk.len() > MAX_BODY_BYTES {
                let room = MAX_BODY_BYTES - body.len();
                body.extend_from_slice(&chunk[..room]);
                truncated = true;
                break;
            }
            body.extend_from_slice(&chunk);
        }
    }

    if content_encoding.as_deref() == Some("gzip") && !truncated {
        body = inflate_gzip(&body).map_err(|e| format!("reading {path} failed: {e}"))?;
    }

    Ok(Response {
        status,
        content_type,
        content_encoding,
        body,
        truncated,
    })
}

fn inflate_gzip(body: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    use std::io::Read;
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(body).read_to_end(&mut out)?;
    Ok(out)
}

/// Enumerates slides from the Python service, which is the reference.
async fn discover_slides(
    client: &reqwest::Client,
    base: &str,
    context: Option<&str>,
) -> Result<Vec<String>, String> {
    let path = match context {
        Some(context) => format!("/v3/cases/?context={}", matrix::encode(context)),
        None => "/v3/cases/".to_string(),
    };
    let response = fetch(client, base, &path).await?;
    if response.status != 200 {
        return Err(format!(
            "could not list cases ({}): {}",
            response.status,
            String::from_utf8_lossy(&response.body)
        ));
    }

    let cases: Vec<serde_json::Value> = serde_json::from_slice(&response.body)
        .map_err(|e| format!("could not parse case list: {e}"))?;

    let mut slides = Vec::new();
    for case in cases {
        if let Some(ids) = case.get("slides").and_then(|s| s.as_array()) {
            for id in ids.iter().filter_map(|v| v.as_str()) {
                slides.push(id.to_owned());
            }
        }
    }
    slides.sort();
    slides.dedup();
    Ok(slides)
}

async fn slide_info(
    client: &reqwest::Client,
    base: &str,
    slide_id: &str,
) -> Result<SlideInfo, String> {
    let path = format!("/v3/slides/info?slide_id={}", matrix::encode(slide_id));
    let response = fetch(client, base, &path).await?;
    if response.status != 200 {
        return Err(format!("info for {slide_id} returned {}", response.status));
    }
    serde_json::from_slice(&response.body)
        .map_err(|e| format!("could not parse info for {slide_id}: {e}"))
}

fn allowlist_reason(label: &str, path: &str) -> Option<&'static str> {
    ALLOWLIST
        .iter()
        .find(|allowed| {
            label.starts_with(allowed.label)
                && (allowed.path_contains.is_empty() || path.contains(allowed.path_contains))
        })
        .map(|allowed| allowed.reason)
}

#[derive(Default)]
struct Summary {
    matched: usize,
    tolerated: usize,
    allowlisted: usize,
    differed: Vec<(String, String, String)>,
    errored: Vec<(String, String)>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let config = match parse_args() {
        Ok(config) => config,
        Err(message) => {
            eprintln!("error: {message}\n");
            print_usage();
            return ExitCode::from(2);
        }
    };

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
    {
        Ok(client) => client,
        Err(err) => {
            eprintln!("error: could not build HTTP client: {err}");
            return ExitCode::from(2);
        }
    };

    println!("python: {}", config.python_url);
    println!("rust:   {}", config.rust_url);

    let slides = match discover_slides(&client, &config.python_url, config.context.as_deref()).await
    {
        Ok(slides) => slides,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::from(2);
        }
    };
    println!("discovered {} slides", slides.len());

    let mut requests: Vec<Request> = matrix::service_level(config.context.as_deref());
    let mut skipped = Vec::new();

    for slide_id in slides.iter().take(config.limit_slides) {
        match slide_info(&client, &config.python_url, slide_id).await {
            Ok(info) => requests.extend(matrix::build(slide_id, &info)),
            Err(err) => skipped.push(format!("{slide_id}: {err}")),
        }
    }

    if !skipped.is_empty() {
        // Never let a slide disappear from the run silently — a skipped slide is a gap in
        // coverage, not a pass.
        println!("\n{} slides skipped:", skipped.len());
        for note in &skipped {
            println!("  - {note}");
        }
    }
    if slides.len() > config.limit_slides {
        println!(
            "\nnote: --limit-slides capped the run at {} of {} slides",
            config.limit_slides,
            slides.len()
        );
    }

    println!("\nrunning {} comparisons\n", requests.len());

    let mut summary = Summary::default();
    let mut tolerances: BTreeMap<String, String> = BTreeMap::new();

    for request in &requests {
        let python = fetch(&client, &config.python_url, &request.path).await;
        let rust = fetch(&client, &config.rust_url, &request.path).await;

        let (python, rust) = match (python, rust) {
            (Ok(python), Ok(rust)) => (python, rust),
            (Err(err), _) | (_, Err(err)) => {
                summary.errored.push((request.label.clone(), err));
                continue;
            }
        };

        match compare::compare_with(&python, &rust, config.tolerance) {
            Verdict::Match => summary.matched += 1,
            Verdict::MatchWithTolerance(note) => {
                summary.tolerated += 1;
                tolerances.insert(request.label.clone(), note);
            }
            Verdict::Differ(reason) => match allowlist_reason(&request.label, &request.path) {
                Some(_) => summary.allowlisted += 1,
                None => {
                    summary
                        .differed
                        .push((request.label.clone(), request.path.clone(), reason))
                }
            },
        }
    }

    print_summary(&summary, &tolerances);

    if let Some(path) = &config.report {
        let report = serde_json::json!({
            "python_url": config.python_url,
            "rust_url": config.rust_url,
            "comparisons": requests.len(),
            "matched": summary.matched,
            "tolerated": summary.tolerated,
            "allowlisted": summary.allowlisted,
            "skipped_slides": skipped,
            "tolerances": tolerances,
            "differences": summary.differed.iter().map(|(label, path, reason)| {
                serde_json::json!({ "label": label, "path": path, "reason": reason })
            }).collect::<Vec<_>>(),
            "errors": summary.errored.iter().map(|(label, reason)| {
                serde_json::json!({ "label": label, "reason": reason })
            }).collect::<Vec<_>>(),
        });
        match serde_json::to_vec_pretty(&report) {
            Ok(bytes) => {
                if let Err(err) = std::fs::write(path, bytes) {
                    eprintln!("could not write report to {path}: {err}");
                } else {
                    println!("\nreport written to {path}");
                }
            }
            Err(err) => eprintln!("could not serialise report: {err}"),
        }
    }

    if summary.differed.is_empty() && summary.errored.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn print_summary(summary: &Summary, tolerances: &BTreeMap<String, String>) {
    if !tolerances.is_empty() {
        println!("tolerated ({}):", tolerances.len());
        // Group by the note so the output stays readable when a whole format is relaxed.
        let mut by_note: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (label, note) in tolerances {
            by_note.entry(note.as_str()).or_default().push(label);
        }
        for (note, labels) in by_note {
            println!("  {note}  ({} requests)", labels.len());
        }
        println!();
    }

    if !summary.differed.is_empty() {
        println!("DIFFERENCES ({}):", summary.differed.len());
        for (label, path, reason) in &summary.differed {
            println!("  {label}\n    {path}\n    {reason}");
        }
        println!();
    }

    if !summary.errored.is_empty() {
        println!("ERRORS ({}):", summary.errored.len());
        for (label, reason) in &summary.errored {
            println!("  {label}: {reason}");
        }
        println!();
    }

    println!(
        "matched {} | tolerated {} | allowlisted {} | differed {} | errored {}",
        summary.matched,
        summary.tolerated,
        summary.allowlisted,
        summary.differed.len(),
        summary.errored.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_entries_match_by_label_prefix() {
        let any = "/v3/slides/label/max_size/64/64?slide_id=.%2Faperio%2FCMU-1.svs";
        assert!(allowlist_reason("files:label:64x64", any).is_some());
        assert!(allowlist_reason("batch:macro:format=png", any).is_some());
        assert!(allowlist_reason("openapi:v3", any).is_some());
        assert!(allowlist_reason("tile:l0:origin", any).is_none());
        assert!(allowlist_reason("info", any).is_none());
    }

    /// A scoped entry must not excuse the same label on another slide. Divergence 38 is
    /// only reachable where the plugin caches its associated images, and `/slides/label`
    /// has to keep matching exactly everywhere else.
    #[test]
    fn a_slide_scoped_entry_only_matches_that_slide() {
        assert!(allowlist_reason(
            "label:64x64",
            "/v3/slides/x?slide_id=.%2Fdicomcase%2FDICOM_Aperio"
        )
        .is_some());
        assert!(allowlist_reason(
            "macro:format=png",
            "/v3/slides/x?slide_id=.%2Fdicomcase%2FDICOM_Aperio"
        )
        .is_some());
        assert!(allowlist_reason(
            "label:64x64",
            "/v3/slides/x?slide_id=.%2Faperio%2FCMU-1.svs"
        )
        .is_none());
        assert!(allowlist_reason(
            "macro:format=png",
            "/v3/slides/x?slide_id=.%2Fgeneric%2FCMU-1.tiff"
        )
        .is_none());
        // An unscoped entry still matches a DICOM slide.
        assert!(allowlist_reason(
            "openapi:v3",
            "/v3/slides/x?slide_id=.%2Fdicomcase%2FDICOM_Aperio"
        )
        .is_some());
    }

    #[test]
    fn every_allowlist_entry_carries_a_reason() {
        for allowed in ALLOWLIST {
            assert!(!allowed.label.is_empty());
            assert!(
                allowed.reason.len() > 20,
                "allowlist entry {} needs a real explanation",
                allowed.label
            );
        }
    }
}
