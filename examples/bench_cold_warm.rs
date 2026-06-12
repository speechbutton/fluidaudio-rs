//! Bench: cold vs warm init_asr, with and without prewarm.
//!
//! Usage:
//!   cargo run --release --example bench_cold_warm
//!
//! What it measures:
//!   1. Baseline cold init: `FluidAudio::new()` + `init_asr()` back-to-back.
//!      This is what the consumer sees today (~16-30s).
//!   2. Prewarmed init: `FluidAudio::new()` + `prewarm_asr()` (returns
//!      immediately) + sleep N seconds (simulating the user doing other
//!      stuff in the app) + `init_asr()`. The third call is what the
//!      consumer sees — should be near-zero once N covers the load time.
//!
//! Each pass runs in its own bridge instance, but the OS-level e5rt cache
//! and downloaded model files are shared, so the "cold" cost here is closer
//! to the "warm" production case (16s). To exercise the real cold path, kill
//! the process between runs and re-launch with the test data area cleared —
//! that's tolki-client's job to verify on iOS device rails.
//!
//! Expected reading on a warm system:
//!   - baseline: ~10-20s on M-series macOS (less than iOS because macOS has
//!     better ANE-cache persistence and faster IO).
//!   - prewarmed (after 30s sleep): near-zero, because the in-flight task
//!     finished while we slept.

use std::time::{Duration, Instant};

use fluidaudio_rs::FluidAudio;

fn main() {
    println!("=== fluidaudio-rs bench: cold vs warm init_asr ===\n");

    // Pass 1: baseline (no prewarm)
    {
        let fa = FluidAudio::new().expect("bridge create");
        let t0 = Instant::now();
        fa.init_asr().expect("init_asr");
        let baseline = t0.elapsed();
        println!(
            "[1] baseline cold-style init_asr: {:>7.2}s",
            baseline.as_secs_f64()
        );
    }

    // Pass 2: prewarm + wait + init
    {
        let fa = FluidAudio::new().expect("bridge create");
        fa.prewarm_asr().expect("prewarm_asr");
        let prewarm_started = Instant::now();
        println!(
            "[2] prewarm_asr returned in {:?} (non-blocking)",
            prewarm_started.elapsed()
        );

        // Simulate the user doing other things while the load happens.
        let sleep = Duration::from_secs(30);
        println!(
            "    sleeping {:?} to let the background load finish…",
            sleep
        );
        std::thread::sleep(sleep);

        let t0 = Instant::now();
        fa.init_asr().expect("init_asr after prewarm");
        let prewarmed = t0.elapsed();
        println!(
            "[3] init_asr after {:?} prewarm:    {:>7.2}s",
            sleep,
            prewarmed.as_secs_f64()
        );
    }

    // Pass 3: same bridge, double init_asr (singleton check)
    {
        let fa = FluidAudio::new().expect("bridge create");
        let t0 = Instant::now();
        fa.init_asr().expect("init_asr #1");
        let first = t0.elapsed();
        let t1 = Instant::now();
        fa.init_asr().expect("init_asr #2 (idempotent)");
        let second = t1.elapsed();
        println!(
            "[4] singleton: init_asr #1 = {:>7.2}s, #2 = {:>7.3}s (must be ~0)",
            first.as_secs_f64(),
            second.as_secs_f64()
        );
    }

    println!("\n=== done ===");
}
