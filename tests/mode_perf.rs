//! Local regression guard: fail if any mode runs more than `MAX_RATIO×` the median.
//!
//! Marked `#[ignore]` so `cargo test` doesn't run it by default — needs a real
//! GPU and takes a couple of seconds. The pre-commit hook in `.githooks/`
//! invokes it explicitly via:
//!
//!     cargo test --release --test mode_perf -- --ignored --nocapture
//!
//! Uses a relative threshold (× median) rather than an absolute ms target so
//! the test stays portable across GPUs — what matters for a smooth interactive
//! experience is the ratio between the slowest and a typical mode.

use crystal_viz::bench::{run, slow_modes, BenchOpts};

const MAX_RATIO: f64 = 10.0;

/// Modes exempt from the ratio gate. The threshold is relative to the median,
/// which is set by cheap 2D field modes — heavy raymarchers can't meet 10×
/// without gutting their visuals, so they're intentionally exempt:
///   • LATTICE WALK — fullscreen first-person raymarcher, inherently ~30× heavier.
///   • WANNIER      — isosurface raymarcher that evaluates a full inverse-FT
///     (Σ over all G-vectors, cos+sin each) at every march sample. It sits right
///     at the ratio ceiling (~10× median); since the median is set by the cheap
///     2D modes, adding any new cheap mode nudges the median down and tips it
///     over. Its absolute cost (~1.6 ms, >600 fps) is still fully interactive.
const EXEMPT: &[&str] = &["LATTICE WALK", "WANNIER"];

#[test]
#[ignore = "GPU benchmark — run via the pre-commit hook or `cargo test -- --ignored`"]
fn no_mode_exceeds_ratio_of_median() {
    let opts = BenchOpts {
        width: 512,
        height: 512,
        warmup: 4,
        iters: 30,
        force_fallback: false,
    };

    let (gpu, results) = run(&opts).expect("benchmark could not run — is a GPU available?");

    eprintln!("GPU: {gpu}");
    eprintln!("Threshold: any mode > {MAX_RATIO}× median fails the test\n");
    for r in &results {
        eprintln!("  [{:2}] {:<14} {:7.3} ms", r.idx, r.name, r.ms);
    }

    let (median, slow) = slow_modes(&results, MAX_RATIO);
    eprintln!("\nMedian: {median:.3} ms — cutoff: {:.3} ms", median * MAX_RATIO);

    // Drop intentionally-heavy raymarcher modes from the failure set.
    let offenders: Vec<_> = slow
        .iter()
        .filter(|r| !EXEMPT.contains(&r.name))
        .collect();
    for r in slow.iter().filter(|r| EXEMPT.contains(&r.name)) {
        eprintln!(
            "  (exempt) [{:2}] {:<14} {:7.3} ms  ({:.1}× median) — allowed",
            r.idx, r.name, r.ms, r.ms / median,
        );
    }

    if !offenders.is_empty() {
        let mut msg = format!(
            "\n{} mode(s) exceed {MAX_RATIO}× the median ({median:.3} ms):\n",
            offenders.len(),
        );
        for r in &offenders {
            msg.push_str(&format!(
                "  [{:2}] {:<14} {:7.3} ms  ({:.1}× median)\n",
                r.idx, r.name, r.ms, r.ms / median,
            ));
        }
        msg.push_str("\nEither optimise the offending mode, add it to EXEMPT, or remove it from the dispatch list.\n");
        panic!("{msg}");
    }
}
