//! Sequencer preset format: a single JSON document capturing the full state
//! the user can edit (steps, transitions, LFOs, trip level). Used for:
//!
//!   - shipping curated presets bundled with the app (BUNDLED_PRESETS)
//!   - the UI "💾 Save / 📁 Load / 🎲 Random / 📋 Copy / 📥 Paste" buttons
//!   - the headless clip-render command (`crystal-viz --render preset.json …`)
//!
//! Versioning: bump `Preset::VERSION` when the JSON shape changes. Parsers
//! reject unknown major versions; minor additions stay backward-compatible.
//!
//! The format is deliberately small — every field maps 1:1 onto an in-app
//! piece of state. No derived data is stored.

use serde::{Deserialize, Serialize};

use crate::{
    LfoParams, LfoEngine, LfoSrc,
    SeqPlayMode, SeqStep, Sequencer, TranCurve, TripLevel,
    tour_lfo_preset_for_level, randomize_fp, bz_hash,
};

#[derive(Clone, Serialize, Deserialize)]
pub struct Preset {
    pub version:     u32,
    pub name:        String,
    pub step_dur:    f32,
    pub curve:       TranCurve,
    pub play_mode:   SeqPlayMode,
    pub trip_level:  TripLevel,
    pub lfo:         LfoParams,
    pub steps:       Vec<SeqStep>,
    /// Morph window (v2): fraction of each step spent transitioning.
    /// Old presets default to 1.0 — the classic wall-to-wall glide.
    #[serde(default = "morph_default")]
    pub morph:       f32,
    /// Drift lane depth (v2). Old presets default to 0 (off).
    #[serde(default)]
    pub drift:       f32,
}

fn morph_default() -> f32 { 1.0 }

impl Preset {
    pub const VERSION: u32 = 2;

    /// Snapshot the current app state (sequencer + LFO + trip level + name).
    pub fn from_state(name: &str, seq: &Sequencer, lfo: &LfoParams, level: TripLevel) -> Self {
        Self {
            version:    Self::VERSION,
            name:       name.to_string(),
            step_dur:   seq.step_dur,
            curve:      seq.curve,
            play_mode:  seq.play_mode,
            trip_level: level,
            lfo:        lfo.clone(),
            steps:      seq.steps.clone(),
            morph:      seq.morph,
            drift:      seq.drift,
        }
    }

    /// Apply the preset back into the live sequencer/LFO/level. The sequencer
    /// is rewound to step 0 and the LFO replaces the current routing.
    pub fn apply(&self, seq: &mut Sequencer, lfo: &mut LfoParams, level: &mut TripLevel) {
        seq.steps      = self.steps.clone();
        seq.cur        = 0;
        seq.step_timer = 0.0;
        seq.from_params = seq.steps.get(0).map(|s| s.params.clone())
            .unwrap_or_default();
        seq.step_dur   = self.step_dur;
        seq.curve      = self.curve;
        seq.play_mode  = self.play_mode;
        seq.pp_dir     = 1;
        seq.selected   = None;
        seq.morph      = self.morph;
        seq.drift      = self.drift;
        for s in &mut seq.steps { s.visits = 0; }
        *lfo   = self.lfo.clone();
        *level = self.trip_level;
    }

    pub fn to_pretty_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("Preset serialize")
    }

    pub fn from_json(s: &str) -> Result<Self, String> {
        let p: Self = serde_json::from_str(s).map_err(|e| format!("preset parse: {e}"))?;
        if p.version > Self::VERSION {
            return Err(format!(
                "preset version {} is newer than this build (max {})",
                p.version, Self::VERSION,
            ));
        }
        if p.steps.is_empty() {
            return Err("preset has zero steps".to_string());
        }
        if p.steps.len() > 32 {
            return Err(format!("preset has {} steps (max 32)", p.steps.len()));
        }
        Ok(p)
    }
}

// ── Random generation ────────────────────────────────────────────────────

/// Small deterministic LCG so a `seed` reproduces the same preset. Knuth's
/// constants. We don't need crypto here — just a fast portable PRNG.
struct Lcg(u32);
impl Lcg {
    fn new(seed: u32) -> Self { Self(seed.wrapping_mul(2654435761).wrapping_add(1)) }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u32() >> 8) & 0x00FF_FFFF) as f32 / 16_777_216.0
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + (hi - lo) * self.unit() }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        let i = (self.unit() * xs.len() as f32) as usize;
        xs[i.min(xs.len() - 1)]
    }
}

/// Build a fresh, coherent random preset. Coverage:
///   - 4..16 random steps from randomize_fp (existing per-step randomizer)
///   - per-step dur_mul, prob, curve_override randomized
///   - global step_dur, curve, play_mode randomized
///   - trip_level from a wider distribution (skewed toward middle)
///   - LFO config derived from a level-aware preset for coherence
pub fn random_preset(seed: u32) -> Preset {
    let mut r = Lcg::new(seed);

    let n_steps = 4 + (r.unit() * 12.0) as usize;       // 4..=15
    let mut steps = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
        let mut s = SeqStep::new(randomize_fp(seed as f32 + i as f32 * 71.3));
        s.dur_mul = r.pick(&[0.5, 0.75, 1.0, 1.0, 1.0, 1.5, 2.0]);
        // Probability defaults to 1.0 for ~80 % of steps; sub-1.0 only on a
        // small minority so the sequencer doesn't routinely skip half the
        // pattern. Mute is now ~2 % (was 7 %) — a rare "drop-out" accent.
        s.prob    = r.pick(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.6]);
        s.muted   = r.unit() < 0.02;
        s.curve_override = if r.unit() < 0.25 {
            Some(r.pick(&[TranCurve::Linear, TranCurve::EaseInOut, TranCurve::Silk,
                          TranCurve::Snap, TranCurve::Bounce, TranCurve::Over]))
        } else { None };
        // Elektron cycle conditions on a small minority — rhythmic variation
        // across pattern passes without routinely hollowing the loop out.
        s.cond = if r.unit() < 0.15 {
            r.pick(&[(1u8, 2u8), (2, 2), (1, 3), (1, 4)])
        } else { (1, 1) };
        steps.push(s);
    }

    let step_dur  = r.range(0.6, 6.0);
    let curve     = r.pick(&[TranCurve::Linear, TranCurve::EaseInOut, TranCurve::Silk,
                             TranCurve::Silk, TranCurve::Snap, TranCurve::Bounce, TranCurve::Over]);
    let play_mode = r.pick(&[SeqPlayMode::Forward, SeqPlayMode::Forward, SeqPlayMode::Forward,
                             SeqPlayMode::Reverse, SeqPlayMode::PingPong, SeqPlayMode::Random]);

    // Trip level: triangular distribution centred on ~5.
    let level_raw = ((r.unit() + r.unit()) * 0.5 * 10.0) as u8;
    let trip_level = TripLevel::new(level_raw);

    let mut lfo = tour_lfo_preset_for_level(trip_level);
    // Add a bit of variance to the LFO so two random presets at the same level
    // don't look identical.
    lfo.a = LfoEngine {
        rate: lfo.a.rate * r.range(0.8, 1.3),
        depth: (lfo.a.depth * r.range(0.7, 1.4)).min(1.0),
        wave: lfo.a.wave,
        phase: r.unit(),
    };
    lfo.b = LfoEngine {
        rate: lfo.b.rate * r.range(0.8, 1.3),
        depth: (lfo.b.depth * r.range(0.7, 1.4)).min(1.0),
        wave: lfo.b.wave,
        phase: r.unit(),
    };
    // Occasionally re-route a target so two random presets feel different.
    use crate::{MP_KSCALE, MP_SPEED, MP_FIELD_MIX, MP_ISO_LEVEL, MP_COLOR_SHIFT, MP_ZOOM};
    let maybe_reroute = |slot: &mut LfoSrc, r: &mut Lcg| {
        if r.unit() < 0.18 {
            *slot = match r.next_u32() & 3 {
                0 => LfoSrc::Off,
                1 => LfoSrc::A,
                _ => LfoSrc::B,
            };
        }
    };
    // Field-slot targets index into lfo.mp; feedback targets are named fields.
    for i in [MP_KSCALE, MP_SPEED, MP_FIELD_MIX, MP_ISO_LEVEL, MP_COLOR_SHIFT, MP_ZOOM] {
        maybe_reroute(&mut lfo.mp[i], &mut r);
    }
    maybe_reroute(&mut lfo.fb_zoom, &mut r);
    maybe_reroute(&mut lfo.fb_decay, &mut r);
    maybe_reroute(&mut lfo.fb_color_shift, &mut r);
    maybe_reroute(&mut lfo.fb_saturation, &mut r);
    maybe_reroute(&mut lfo.fb_fold_angle, &mut r);
    // Flux budget: mid-range so dense routing stays bounded but audible.
    lfo.flux = r.range(0.30, 0.65);

    let _ = bz_hash; // keep import warning-free if unused below

    let morph = r.range(0.35, 1.0);
    let drift = r.range(0.25, 0.75);
    let name = format!("RND-{seed:08x}");
    Preset {
        version: Preset::VERSION,
        name, step_dur, curve, play_mode, trip_level,
        lfo, steps, morph, drift,
    }
}

// ── Bundled presets ──────────────────────────────────────────────────────
//
// Each is a human-curated JSON file shipped in `presets/` and embedded at
// compile time. The order here is the order they appear in the UI dropdown.

pub static BUNDLED_PRESETS: &[(&str, &str)] = &[
    ("Slow Zen",         include_str!("../presets/slow_zen.preset.json")),
    ("Phase Drift",      include_str!("../presets/phase_drift.preset.json")),
    ("Quantum Pulse",    include_str!("../presets/quantum_pulse.preset.json")),
    ("Vortex Garden",    include_str!("../presets/vortex_garden.preset.json")),
    ("Crystal Bloom",    include_str!("../presets/crystal_bloom.preset.json")),
    ("Topological Run",  include_str!("../presets/topological_run.preset.json")),
    ("Strobe Lattice",   include_str!("../presets/strobe_lattice.preset.json")),
    ("Total Madness",    include_str!("../presets/total_madness.preset.json")),
];

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_preset_is_valid_at_many_seeds() {
        for seed in 0..200u32 {
            let p = random_preset(seed.wrapping_mul(7919));
            assert!(p.steps.len() >= 4 && p.steps.len() <= 16,
                "seed {seed}: bad step count {}", p.steps.len());
            assert!(p.step_dur > 0.0);
            assert!(p.trip_level.get() <= 9);
            for (i, s) in p.steps.iter().enumerate() {
                assert!(s.dur_mul > 0.0, "seed {seed} step {i}: dur_mul = {}", s.dur_mul);
                assert!(s.prob >= 0.0 && s.prob <= 1.0,
                    "seed {seed} step {i}: prob = {}", s.prob);
            }
        }
    }

    #[test]
    fn random_preset_is_deterministic() {
        let a = random_preset(0xDEADBEEF);
        let b = random_preset(0xDEADBEEF);
        let aj = a.to_pretty_json();
        let bj = b.to_pretty_json();
        assert_eq!(aj, bj, "same seed must produce identical JSON");
    }

    #[test]
    fn roundtrip_preserves_state() {
        let p = random_preset(12345);
        let s = p.to_pretty_json();
        let q = Preset::from_json(&s).expect("parse own output");
        assert_eq!(p.steps.len(), q.steps.len());
        assert_eq!(p.trip_level.get(), q.trip_level.get());
        assert!((p.step_dur - q.step_dur).abs() < 1e-5);
    }

    #[test]
    fn roundtrip_preserves_v2_fields() {
        let mut p = random_preset(777);
        p.morph = 0.42;
        p.drift = 0.63;
        p.lfo.flux = 0.71;
        p.steps[0].cond = (2, 2);
        let q = Preset::from_json(&p.to_pretty_json()).expect("parse own output");
        assert!((q.morph - 0.42).abs() < 1e-5);
        assert!((q.drift - 0.63).abs() < 1e-5);
        assert!((q.lfo.flux - 0.71).abs() < 1e-5);
        assert_eq!(q.steps[0].cond, (2, 2));
    }

    #[test]
    fn v1_preset_without_new_fields_gets_defaults() {
        // Simulate a v1 file: serialize, strip the new keys, reparse.
        let p = random_preset(31337);
        let mut v: serde_json::Value = serde_json::from_str(&p.to_pretty_json()).unwrap();
        let obj = v.as_object_mut().unwrap();
        obj.remove("morph");
        obj.remove("drift");
        obj["version"] = 1.into();
        obj["lfo"].as_object_mut().unwrap().remove("flux");
        for s in obj["steps"].as_array_mut().unwrap() {
            s.as_object_mut().unwrap().remove("cond");
        }
        let q = Preset::from_json(&v.to_string()).expect("v1 shape must parse");
        assert!((q.morph - 1.0).abs() < 1e-5, "v1 defaults to wall-to-wall glide");
        assert!(q.drift.abs() < 1e-5,         "v1 defaults to drift off");
        assert!((q.lfo.flux - 0.5).abs() < 1e-5, "flux defaults to 0.5");
        assert!(q.steps.iter().all(|s| s.cond == (1, 1)), "cond defaults to always");
    }

    #[test]
    fn rejects_future_version() {
        let mut p = random_preset(7);
        p.version = Preset::VERSION + 1;
        let s = p.to_pretty_json();
        assert!(Preset::from_json(&s).is_err());
    }

    #[test]
    fn rejects_empty_steps() {
        let mut p = random_preset(7);
        p.steps.clear();
        let s = p.to_pretty_json();
        assert!(Preset::from_json(&s).is_err());
    }

    /// Curated bundled presets. Run with:
    ///   cargo test --bin crystal-viz -- --ignored gen_bundled_presets
    /// This writes `presets/*.preset.json` next to Cargo.toml. It's
    /// intentionally gated so the bundled files don't get rewritten on every
    /// `cargo test` — review the diff and commit when you change tunings.
    #[test]
    #[ignore]
    fn gen_bundled_presets() {
        // (slug, display name, seed, trip_level override, step_dur override, mode bias)
        // mode_bias: clamp step modes to this single mode (Some) to give a
        // preset a coherent visual identity. None = leave randomizer's choice.
        let bundles: &[(&str, &str, u32, u8, f32, Option<u32>)] = &[
            ("slow_zen",        "Slow Zen",        0x0A11_CE01, 1, 5.0,  Some(14)),  // BAND SURFACE
            ("phase_drift",     "Phase Drift",     0xC0FFEE01, 3, 3.2,  Some(5)),    // PHASE
            ("quantum_pulse",   "Quantum Pulse",   0xC0FFEE02, 5, 2.0,  Some(22)),   // VORTEX KNOT
            ("vortex_garden",   "Vortex Garden",   0xC0FFEE03, 5, 2.4,  Some(24)),   // ABRIKOSOV
            ("crystal_bloom",   "Crystal Bloom",   0xC0FFEE04, 6, 1.8,  Some(0)),    // 3D ISO
            ("topological_run", "Topological Run", 0xC0FFEE05, 7, 1.4,  Some(20)),   // BERRY
            ("strobe_lattice",  "Strobe Lattice",  0xC0FFEE06, 8, 0.9,  Some(16)),   // CDW
            ("total_madness",   "Total Madness",   0xC0FFEE07, 9, 0.7,  None),
        ];

        std::fs::create_dir_all("presets").expect("mkdir presets");
        for (slug, name, seed, level, step_dur, mode_bias) in bundles {
            let mut p = random_preset(*seed);
            p.name       = (*name).to_string();
            p.trip_level = TripLevel::new(*level);
            p.step_dur   = *step_dur;
            // Re-derive LFO from the chosen level so the JSON has the
            // level's preset rather than the random one used for variance.
            p.lfo        = tour_lfo_preset_for_level(p.trip_level);
            if let Some(m) = mode_bias {
                for s in &mut p.steps {
                    s.params.mode = *m;
                }
            }
            // Bundled presets play every step every cycle — no surprise drops.
            // Random mutation lives in the 🎲 RANDOM button, not in the
            // hand-curated bundle.
            for s in &mut p.steps {
                s.muted = false;
                s.prob  = 1.0;
                s.cond  = (1, 1);
            }
            let path = format!("presets/{slug}.preset.json");
            std::fs::write(&path, p.to_pretty_json())
                .unwrap_or_else(|e| panic!("write {path}: {e}"));
            eprintln!("wrote {path} ({} steps, trip {})", p.steps.len(), p.trip_level.get());
        }
    }

    #[test]
    fn every_bundled_preset_parses() {
        for (name, json) in BUNDLED_PRESETS {
            let p = Preset::from_json(json)
                .unwrap_or_else(|e| panic!("bundled preset '{name}' failed to parse: {e}"));
            assert!(!p.steps.is_empty(), "'{name}' has no steps");
            assert!(p.trip_level.get() <= 9, "'{name}' bad trip level");
        }
    }

    // Lock in the "bundled presets play every step every cycle" guarantee.
    // The user reported steps being silently skipped after loading a preset
    // (turned out to be inherited `prob`<1 and `muted=true` from the random
    // generator). Bundles should never have those; the 🎲 button is where
    // randomness lives.
    #[test]
    fn bundled_presets_have_no_silent_skips() {
        for (name, json) in BUNDLED_PRESETS {
            let p = Preset::from_json(json).expect(name);
            for (i, s) in p.steps.iter().enumerate() {
                assert!(!s.muted,
                    "'{name}' step {i} is muted — bundled presets must play every step");
                assert!((s.prob - 1.0).abs() < 1e-6,
                    "'{name}' step {i} has prob={} — bundles must use prob=1.0", s.prob);
                assert_eq!(s.cond, (1, 1),
                    "'{name}' step {i} has cond {:?} — bundles must play every pass", s.cond);
            }
        }
    }
}
