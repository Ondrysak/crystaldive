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

    if !slow.is_empty() {
        let mut msg = format!(
            "\n{} mode(s) exceed {MAX_RATIO}× the median ({median:.3} ms):\n",
            slow.len(),
        );
        for r in &slow {
            msg.push_str(&format!(
                "  [{:2}] {:<14} {:7.3} ms  ({:.1}× median)\n",
                r.idx, r.name, r.ms, r.ms / median,
            ));
        }
        msg.push_str("\nEither optimise the offending mode or remove it from the dispatch list.\n");
        panic!("{msg}");
    }
}
