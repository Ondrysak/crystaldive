//! Headless GPU benchmark for the field render modes.
//!
//!     cargo run --release --example bench_modes
//!
//! Prints per-mode ms/frame, sorted ranking, and flags any mode running
//! more than 2× the median.

use crystal_viz::bench::{run, slow_modes, BenchOpts};

fn main() {
    env_logger::init();

    let opts = BenchOpts::default();
    println!(
        "crystal-viz mode benchmark — {}×{}, {} iters/mode (after {} warmup)\n",
        opts.width, opts.height, opts.iters, opts.warmup,
    );

    let (gpu, results) = match run(&opts) {
        Ok(r) => r,
        Err(e) => { eprintln!("benchmark failed: {e}"); std::process::exit(1); }
    };
    println!("GPU: {gpu}\n");

    for r in &results {
        println!(
            "  [{:2}] {:<14} {:7.3} ms/frame  ({:6.1} fps)",
            r.idx, r.name, r.ms, 1000.0 / r.ms,
        );
    }

    println!("\n── sorted (fastest → slowest) ──────────────────────────────────");
    let mut sorted = results.clone();
    sorted.sort_by(|a, b| a.ms.partial_cmp(&b.ms).unwrap());
    for r in &sorted {
        println!("  [{:2}] {:<14} {:7.3} ms", r.idx, r.name, r.ms);
    }

    let (median, slow) = slow_modes(&results, 2.0);
    if slow.is_empty() {
        println!("\nNo modes are >2× slower than the median ({median:.3} ms).");
    } else {
        println!("\n── slow modes (>2× median = {:.3} ms) ─────────────────────────", median * 2.0);
        for r in &slow {
            println!("  [{:2}] {:<14} {:7.3} ms  ({:.1}× median)", r.idx, r.name, r.ms, r.ms / median);
        }
    }
}
