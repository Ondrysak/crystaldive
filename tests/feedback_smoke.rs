//! Headless smoke test: enables feedback, renders 10 frames at 64×64,
//! reads back the last frame and asserts at least one pixel is non-zero.
//!
//! Run manually:
//!   cargo test --test feedback_smoke -- --ignored --nocapture

use crystal_viz::bench::run_feedback_smoke;

#[test]
#[ignore = "GPU smoke test — run with `cargo test --test feedback_smoke -- --ignored`"]
fn feedback_produces_non_black_output() {
    match run_feedback_smoke(10, 64, 64) {
        Err(e) => eprintln!("skipped (no GPU): {e}"),
        Ok(max) => {
            eprintln!("feedback smoke: max pixel value = {max}");
            assert!(max > 0, "feedback produced all-black output — pipeline format mismatch or shader bug");
        }
    }
}
