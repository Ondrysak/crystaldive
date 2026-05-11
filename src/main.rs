use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::WindowAttributes,
};

use crystal_viz::{audio, crystals, kpoints, poscar, reciprocal, renderer, symmetry};
use crystals::{all_crystals, all_groups, CrystalDef};
use poscar::Crystal;
use reciprocal::{crossfade_pack, GpuField};
use renderer::{FieldUniform, GpuState};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};

// ── field parameters ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct FieldParams {
    pub mode:        u32,
    pub kscale:      f32,
    pub speed:       f32,
    pub field_mix:   f32,
    pub iso_level:   f32,
    pub color_shift: f32,
    pub zoom:        f32,
    pub w_lattice:   f32,
    pub w_motif:     f32,
    pub w_band:      f32,
}

impl Default for FieldParams {
    fn default() -> Self {
        Self {
            mode: 4, kscale: 1.4, speed: 0.3, field_mix: 0.55,
            iso_level: 0.5, color_shift: 0.0, zoom: 1.0,
            w_lattice: 1.0, w_motif: 0.6, w_band: 0.4,
        }
    }
}

const MODE_NAMES: [&str; 36] = [
    "3D ISO", "BZ SLICE", "FERMI", "DENSITY", "NODAL", "PHASE",
    "STRIPES", "WARP", "LINKS", "XRD", "RECIP", "NONEUC",
    "PHONON", "MOIRE", "EWALD", "WANNIER", "MAGNETIC", "DISPERSION",
    "KIKUCHI", "DEFECT", "BAND SURFACE", "SPIN TEXTURE",
    "BZ PATH", "CDW", "QUASICRYSTAL", "THERMAL", "DOMAIN WALL",
    "FRACTURE",
    "BERRY", "HOFSTADTER", "STM", "SPECTRAL", "VORTEX KNOT",
    "BLOCH WAVE", "PLASMON", "NEMATIC",
];

// ── LFO ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Default)]
pub enum LfoWave {
    #[default]
    Sine,
    Triangle,
    Saw,
    Square,
    Pulse,
    Steps,
}

impl LfoWave {
    fn next(self) -> Self {
        match self {
            Self::Sine => Self::Triangle,
            Self::Triangle => Self::Saw,
            Self::Saw => Self::Square,
            Self::Square => Self::Pulse,
            Self::Pulse => Self::Steps,
            Self::Steps => Self::Sine,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Sine => "sine",
            Self::Triangle => "triangle",
            Self::Saw => "saw",
            Self::Square => "square",
            Self::Pulse => "pulse",
            Self::Steps => "steps",
        }
    }

    fn sample(self, phase: f32) -> f32 {
        let x = phase.fract();
        match self {
            Self::Sine => (TAU * x).sin(),
            Self::Triangle => 1.0 - 4.0 * (x - 0.5).abs(),
            Self::Saw => x * 2.0 - 1.0,
            Self::Square => if x < 0.5 { 1.0 } else { -1.0 },
            Self::Pulse => if x < 0.18 { 1.0 } else { -0.35 },
            Self::Steps => {
                let n = (x * 8.0).floor();
                bz_hash(n + phase.floor() * 17.0) * 2.0 - 1.0
            }
        }
    }
}

fn bz_hash(n: f32) -> f32 {
    (n * 127.1 + 19.19).sin().fract().abs()
}

const TAU: f32 = std::f32::consts::PI * 2.0;

#[derive(Clone)]
pub struct LfoParams {
    pub kscale:      bool,
    pub speed:       bool,
    pub field_mix:   bool,
    pub iso_level:   bool,
    pub color_shift: bool,
    pub zoom:        bool,
    pub w_lattice:   bool,
    pub w_motif:     bool,
    pub w_band:      bool,
    pub rate:        f32,
    pub depth:       f32,
    pub wave:        LfoWave,
}

impl Default for LfoParams {
    fn default() -> Self {
        Self {
            kscale: false, speed: false, field_mix: false, iso_level: false,
            color_shift: false, zoom: false, w_lattice: false, w_motif: false,
            w_band: false, rate: 0.2, depth: 0.3, wave: LfoWave::Sine,
        }
    }
}

// ── Mic modulation ────────────────────────────────────────────────────────

/// Which audio band (or none) drives a parameter.
#[derive(Clone, Copy, PartialEq, Default)]
pub enum MicSrc { #[default] Off, Amp, Bass, Mid, Treble }

impl MicSrc {
    fn next(self) -> Self {
        match self {
            Self::Off    => Self::Amp,
            Self::Amp    => Self::Bass,
            Self::Bass   => Self::Mid,
            Self::Mid    => Self::Treble,
            Self::Treble => Self::Off,
        }
    }
    fn label(self) -> &'static str {
        match self { Self::Off => "·", Self::Amp => "A", Self::Bass => "B", Self::Mid => "M", Self::Treble => "T" }
    }
    fn color(self) -> egui::Color32 {
        match self {
            Self::Off    => egui::Color32::from_gray(70),
            Self::Amp    => egui::Color32::from_rgb(220, 220, 220),
            Self::Bass   => egui::Color32::from_rgb(255,  80,  80),
            Self::Mid    => egui::Color32::from_rgb( 80, 220, 120),
            Self::Treble => egui::Color32::from_rgb( 80, 160, 255),
        }
    }
    fn value(self, b: &audio::AudioBands) -> f32 {
        match self {
            Self::Off    => 0.0,
            Self::Amp    => b.amplitude,
            Self::Bass   => b.bass,
            Self::Mid    => b.mid,
            Self::Treble => b.treble,
        }
    }
}

#[derive(Clone)]
pub struct MicParams {
    pub kscale:      MicSrc,
    pub speed:       MicSrc,
    pub field_mix:   MicSrc,
    pub iso_level:   MicSrc,
    pub color_shift: MicSrc,
    pub zoom:        MicSrc,
    pub w_lattice:   MicSrc,
    pub w_motif:     MicSrc,
    pub w_band:      MicSrc,
    pub depth:       f32,
}

impl Default for MicParams {
    fn default() -> Self {
        Self {
            kscale: MicSrc::Off, speed: MicSrc::Off, field_mix: MicSrc::Off,
            iso_level: MicSrc::Off, color_shift: MicSrc::Off, zoom: MicSrc::Off,
            w_lattice: MicSrc::Off, w_motif: MicSrc::Off, w_band: MicSrc::Off,
            depth: 0.5,
        }
    }
}

// ── Combined LFO + Mic modulation ─────────────────────────────────────────

const TOUR_MODES: [u32; 14] = [22, 32, 28, 20, 30, 15, 24, 34, 4, 29, 21, 25, 18, 35];

// ── Sequencer ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum TranCurve { Linear, EaseInOut, Snap, Bounce }

impl TranCurve {
    fn apply(self, t: f32) -> f32 {
        match self {
            Self::Linear    => t,
            Self::EaseInOut => t * t * (3.0 - 2.0 * t),
            Self::Snap      => if t >= 1.0 { 1.0 } else { 0.0 },
            Self::Bounce    => {
                // quartic ease-in-out: quick rush then slow settle
                if t < 0.5 { 8.0 * t * t * t * t }
                else { let u = t - 1.0; 1.0 - 8.0 * u * u * u * u }
            }
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Linear    => "LINEAR",
            Self::EaseInOut => "EASE",
            Self::Snap      => "SNAP",
            Self::Bounce    => "BOUNCE",
        }
    }
    fn next(self) -> Self {
        match self {
            Self::Linear    => Self::EaseInOut,
            Self::EaseInOut => Self::Snap,
            Self::Snap      => Self::Bounce,
            Self::Bounce    => Self::Linear,
        }
    }
}

#[derive(Clone)]
struct SeqStep {
    params: FieldParams,
    muted:  bool,
}

impl SeqStep {
    fn new(p: FieldParams) -> Self { Self { params: p, muted: false } }
}

fn lerp_fp(a: &FieldParams, b: &FieldParams, t: f32) -> FieldParams {
    let l = |x: f32, y: f32| x + (y - x) * t;
    let d = b.color_shift - a.color_shift;
    let cs_delta = if d > 0.5 { d - 1.0 } else if d < -0.5 { d + 1.0 } else { d };
    FieldParams {
        mode:        if t < 0.5 { a.mode } else { b.mode },
        kscale:      l(a.kscale,    b.kscale),
        speed:       l(a.speed,     b.speed),
        field_mix:   l(a.field_mix, b.field_mix),
        iso_level:   l(a.iso_level, b.iso_level),
        color_shift: (a.color_shift + cs_delta * t).rem_euclid(1.0),
        zoom:        l(a.zoom,      b.zoom),
        w_lattice:   l(a.w_lattice, b.w_lattice),
        w_motif:     l(a.w_motif,   b.w_motif),
        w_band:      l(a.w_band,    b.w_band),
    }
}

// Compact builder used by presets
fn sp(mode: u32, ks: f32, sp: f32, fm: f32, il: f32, cs: f32, zm: f32, wl: f32, wm: f32, wb: f32) -> SeqStep {
    SeqStep::new(FieldParams {
        mode, kscale: ks, speed: sp, field_mix: fm, iso_level: il,
        color_shift: cs, zoom: zm, w_lattice: wl, w_motif: wm, w_band: wb,
    })
}

fn seq_preset_phase_space() -> Vec<SeqStep> { vec![
    sp( 1, 1.5, 0.35, 0.30, 0.55, 0.00, 1.0, 1.5, 0.3, 0.5), // BZ SLICE
    sp( 2, 2.0, 0.45, 0.50, 0.60, 0.12, 1.1, 0.8, 1.2, 0.8), // FERMI
    sp( 4, 1.8, 0.30, 0.65, 0.40, 0.25, 0.9, 1.2, 1.8, 0.6), // NODAL
    sp( 6, 2.5, 0.55, 0.40, 0.50, 0.37, 1.3, 0.5, 0.8, 1.8), // STRIPES
    sp( 5, 1.2, 0.80, 0.55, 0.45, 0.50, 1.0, 1.2, 0.6, 0.8), // PHASE
    sp(10, 2.0, 0.40, 0.70, 0.50, 0.62, 0.9, 0.6, 1.0, 1.5), // RECIP
    sp(13, 1.0, 0.25, 0.45, 0.60, 0.75, 0.8, 1.0, 1.2, 0.9), // MOIRE
    sp(23, 1.8, 0.50, 0.35, 0.70, 0.87, 1.2, 0.7, 0.5, 2.0), // CDW
]}

fn seq_preset_quantum() -> Vec<SeqStep> { vec![
    sp(15, 1.2, 0.20, 0.50, 0.55, 0.00, 1.2, 1.0, 0.6, 0.4), // WANNIER
    sp( 3, 1.5, 0.30, 0.55, 0.50, 0.12, 1.0, 1.8, 0.4, 0.3), // DENSITY
    sp(28, 2.0, 0.40, 0.50, 0.55, 0.25, 0.9, 0.8, 1.0, 1.2), // BERRY
    sp(35, 1.5, 0.30, 0.45, 0.60, 0.37, 1.0, 1.0, 0.5, 0.8), // NEMATIC
    sp(21, 1.8, 0.35, 0.30, 0.50, 0.50, 1.1, 1.2, 0.8, 0.5), // SPIN TEXTURE
    sp(20, 2.2, 0.45, 0.60, 0.45, 0.62, 1.3, 1.5, 0.6, 0.7), // BAND SURFACE
    sp(30, 1.0, 0.35, 0.60, 0.55, 0.75, 1.5, 0.5, 1.2, 0.6), // STM
    sp(29, 1.5, 0.25, 0.50, 0.50, 0.87, 1.0, 0.8, 0.9, 1.0), // HOFSTADTER
]}

fn seq_preset_geometric() -> Vec<SeqStep> { vec![
    sp( 0, 1.5, 0.30, 0.50, 0.40, 0.00, 1.2, 1.0, 0.6, 0.8), // 3D ISO
    sp( 8, 1.2, 0.35, 0.30, 0.50, 0.12, 1.0, 0.8, 1.0, 1.2), // LINKS
    sp( 7, 2.0, 0.80, 0.55, 0.45, 0.25, 0.9, 0.6, 0.8, 1.8), // WARP
    sp(32, 1.8, 0.45, 0.40, 0.55, 0.37, 1.3, 1.2, 0.5, 1.0), // VORTEX KNOT
    sp(12, 1.5, 0.55, 0.60, 0.65, 0.50, 1.1, 0.7, 1.2, 0.8), // PHONON
    sp(14, 1.2, 0.35, 0.45, 0.55, 0.62, 1.1, 1.5, 0.6, 0.5), // EWALD
    sp(26, 1.8, 0.40, 0.55, 0.50, 0.75, 1.0, 0.8, 1.0, 1.8), // DOMAIN WALL
    sp(27, 2.0, 0.50, 0.35, 0.60, 0.87, 1.4, 0.5, 1.5, 1.0), // FRACTURE
]}

fn seq_preset_chromatic() -> Vec<SeqStep> {
    let modes: [u32; 16] = [22, 32, 28, 35, 15, 3, 0, 8, 13, 24, 34, 18, 5, 21, 29, 11];
    modes.iter().enumerate().map(|(i, &mode)| {
        let phi = i as f32 / 16.0;
        SeqStep::new(FieldParams {
            mode,
            kscale:      1.0 + 1.2 * (phi * TAU).sin().abs(),
            speed:       0.2 + 0.6 * (phi * TAU * 0.7).cos().abs(),
            field_mix:   0.3 + 0.5 * (phi * TAU * 1.3).sin().abs(),
            iso_level:   0.3 + 0.4 * (phi * TAU * 0.5).cos().abs(),
            color_shift: phi,
            zoom:        0.8 + 0.6 * (phi * TAU * 1.1).sin().abs(),
            w_lattice:   0.5 + 1.2 * (phi * TAU).cos().abs(),
            w_motif:     0.3 + 1.0 * (phi * TAU * 1.7).sin().abs(),
            w_band:      0.4 + 1.2 * (phi * TAU * 0.9).cos().abs(),
        })
    }).collect()
}

const SEQ_PRESETS: [(&str, fn() -> Vec<SeqStep>); 4] = [
    ("PHASE",    seq_preset_phase_space),
    ("QUANTUM",  seq_preset_quantum),
    ("GEO",      seq_preset_geometric),
    ("CHROMA",   seq_preset_chromatic),
];

struct Sequencer {
    pub active:     bool,
    pub manual:     bool,
    pub steps:      Vec<SeqStep>,
    pub cur:        usize,
    pub step_dur:   f32,
    pub step_timer: f32,
    pub curve:      TranCurve,
    pub selected:   Option<usize>,
    from_params:    FieldParams,
}

impl Sequencer {
    fn new() -> Self {
        let steps = seq_preset_phase_space();
        let from_params = steps[0].params.clone();
        Self {
            active: false, manual: false, steps, cur: 0,
            step_dur: 3.0, step_timer: 0.0,
            curve: TranCurve::EaseInOut, selected: None, from_params,
        }
    }

    fn current_params(&self) -> FieldParams {
        let t = self.curve.apply((self.step_timer / self.step_dur).clamp(0.0, 1.0));
        lerp_fp(&self.from_params, &self.steps[self.cur].params, t)
    }

    fn tick(&mut self, dt: f32) {
        if !self.active { return; }
        self.step_timer += dt;
        if self.manual {
            self.step_timer = self.step_timer.min(self.step_dur); // arrive but don't advance
            return;
        }
        if self.step_timer >= self.step_dur {
            self.step_timer -= self.step_dur;
            self.from_params = self.steps[self.cur].params.clone();
            self.advance_next();
        }
    }

    fn advance_next(&mut self) {
        let n = self.steps.len();
        let mut next = (self.cur + 1) % n;
        for _ in 0..n { if !self.steps[next].muted { break; } next = (next + 1) % n; }
        self.cur = next;
    }

    fn manual_step(&mut self, dir: i32) {
        if self.steps.is_empty() { return; }
        self.from_params = self.current_params();
        self.step_timer = 0.0;
        let n = self.steps.len();
        if dir > 0 {
            self.advance_next();
        } else {
            let mut prev = self.cur.checked_sub(1).unwrap_or(n - 1);
            for _ in 0..n {
                if !self.steps[prev].muted { break; }
                prev = prev.checked_sub(1).unwrap_or(n - 1);
            }
            self.cur = prev;
        }
    }

    fn add_step_after(&mut self, after: usize, seed: f32) {
        if self.steps.len() >= 32 { return; }
        let new_step = self.steps.get(after).cloned()
            .unwrap_or_else(|| SeqStep::new(randomize_fp(seed)));
        let pos = (after + 1).min(self.steps.len());
        self.steps.insert(pos, new_step);
        if self.cur >= pos { self.cur += 1; }
        if let Some(sel) = self.selected { if sel >= pos { self.selected = Some(sel + 1); } }
    }

    fn remove_step(&mut self, idx: usize) {
        if self.steps.len() <= 1 || idx >= self.steps.len() { return; }
        self.steps.remove(idx);
        // shift cur down if removed step was before it; clamp only if cur was the removed step
        if self.cur > idx {
            self.cur -= 1;
        } else {
            self.cur = self.cur.min(self.steps.len() - 1);
        }
        self.selected = match self.selected {
            Some(s) if s == idx => None,
            Some(s) if s > idx  => Some(s - 1),
            s => s,
        };
    }

    fn load_preset(&mut self, make: fn() -> Vec<SeqStep>) {
        self.steps = make();
        self.cur = 0;
        self.step_timer = 0.0;
        self.selected = None;
        if !self.steps.is_empty() { self.from_params = self.steps[0].params.clone(); }
    }
}

fn mode_color(mode: u32) -> egui::Color32 {
    let h = mode as f32 / 36.0;
    let (r, g, b) = hue_to_rgb(h);
    egui::Color32::from_rgb(r, g, b)
}

fn hue_to_rgb(h: f32) -> (u8, u8, u8) {
    let h6 = h * 6.0;
    let hi = h6 as u32 % 6;
    let f  = h6 - h6.floor();
    let q  = 1.0 - f;
    let (r, g, b) = match hi {
        0 => (1.0, f,   0.0),
        1 => (q,   1.0, 0.0),
        2 => (0.0, 1.0, f  ),
        3 => (0.0, q,   1.0),
        4 => (f,   0.0, 1.0),
        _ => (1.0, 0.0, q  ),
    };
    ((r * 200.0) as u8, (g * 200.0) as u8, (b * 200.0) as u8)
}

fn randomize_fp(seed: f32) -> FieldParams {
    let r = |s: f32| tour_rand(seed + s * 13.17);
    FieldParams {
        mode:        ((r(1.0) * MODE_NAMES.len() as f32) as u32).min(MODE_NAMES.len() as u32 - 1),
        kscale:      0.3 + r(2.0) * 3.5,
        speed:       r(3.0) * 1.6,
        field_mix:   r(4.0),
        iso_level:   0.05 + r(5.0) * 0.9,
        color_shift: r(6.0),
        zoom:        0.4 + r(7.0) * 1.9,
        w_lattice:   r(8.0) * 2.0,
        w_motif:     r(9.0) * 2.0,
        w_band:      r(10.0) * 2.0,
    }
}

#[derive(Clone, Copy, PartialEq, Default)]
enum TourStyle {
    #[default]
    Curated,
    Random,
}

impl TourStyle {
    fn next(self) -> Self {
        match self {
            Self::Curated => Self::Random,
            Self::Random => Self::Curated,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Curated => "CURATED",
            Self::Random => "RANDOM",
        }
    }
}

fn tour_lfo_preset() -> LfoParams {
    LfoParams {
        kscale: true,
        speed: true,
        field_mix: true,
        iso_level: true,
        color_shift: true,
        zoom: true,
        w_lattice: true,
        w_motif: true,
        w_band: true,
        rate: 0.18,
        depth: 0.18,
        wave: LfoWave::Triangle,
    }
}

fn tour_rand(seed: f32) -> f32 {
    bz_hash(seed * 71.17 + 4.91)
}

fn tour_random_field_params(t: f32, crystal_idx: usize) -> FieldParams {
    let scene_f = (t / 4.6).floor();
    let local = (t / 4.6).fract();
    let seed = scene_f + crystal_idx as f32 * 37.0;
    let ease = local * local * (3.0 - 2.0 * local);
    let burst = if local < 0.22 { (1.0 - local / 0.22).powf(2.0) } else { 0.0 };

    let mode_a = (tour_rand(seed + 1.0) * MODE_NAMES.len() as f32).floor() as u32;
    let mode_b = (tour_rand(seed + 2.0) * MODE_NAMES.len() as f32).floor() as u32;
    let mode = if local < 0.72 { mode_a } else { mode_b };

    let morph = |slot: f32, min: f32, max: f32| -> f32 {
        let a = tour_rand(seed + slot);
        let b = tour_rand(seed + slot + 19.0);
        min + (max - min) * (a + (b - a) * ease)
    };

    FieldParams {
        mode: mode.min((MODE_NAMES.len() - 1) as u32),
        kscale: (morph(3.0, 0.25, 3.8) + burst * 0.55).clamp(0.1, 5.0),
        speed: (morph(4.0, 0.05, 1.75) + burst * 0.25).clamp(0.0, 2.0),
        field_mix: morph(5.0, 0.0, 1.0).clamp(0.0, 1.0),
        iso_level: morph(6.0, 0.05, 0.96).clamp(0.0, 1.0),
        color_shift: (morph(7.0, 0.0, 1.0) + t * 0.025).rem_euclid(1.0),
        zoom: (morph(8.0, 0.35, 2.15) + burst * 0.25).clamp(0.2, 5.0),
        w_lattice: morph(9.0, 0.0, 2.0).clamp(0.0, 2.0),
        w_motif: morph(10.0, 0.0, 2.0).clamp(0.0, 2.0),
        w_band: morph(11.0, 0.0, 2.0).clamp(0.0, 2.0),
    }
}

fn tour_field_params(t: f32, crystal_idx: usize, style: TourStyle) -> FieldParams {
    if style == TourStyle::Random {
        return tour_random_field_params(t, crystal_idx);
    }

    let scene = (t / 5.8).floor() as usize;
    let local = (t / 5.8).fract();
    let drift = t * 0.17 + crystal_idx as f32 * 0.31;
    let punch = (TAU * local).sin().max(0.0).powf(1.8);
    let snap = if local < 0.16 { (1.0 - local / 0.16).powf(2.0) } else { 0.0 };

    FieldParams {
        mode: TOUR_MODES[(scene + crystal_idx) % TOUR_MODES.len()],
        kscale: (1.05 + 0.72 * (TAU * drift).sin().abs() + 0.45 * snap).clamp(0.1, 5.0),
        speed: (0.22 + 0.82 * punch + 0.18 * (TAU * (drift * 0.37)).sin().abs()).clamp(0.0, 2.0),
        field_mix: (0.50 + 0.38 * (TAU * (local + drift * 0.11)).sin()).clamp(0.0, 1.0),
        iso_level: (0.42 + 0.36 * (TAU * (local * 0.5 + drift * 0.19)).cos()).clamp(0.0, 1.0),
        color_shift: (drift * 0.22 + 0.08 * (TAU * local).sin()).rem_euclid(1.0),
        zoom: (0.78 + 0.36 * (TAU * (local * 0.75)).sin().abs() + 0.22 * snap).clamp(0.2, 5.0),
        w_lattice: (0.75 + 0.65 * (TAU * (local + 0.10)).sin().abs()).clamp(0.0, 2.0),
        w_motif: (0.38 + 0.92 * (TAU * (local * 0.7 + 0.35)).sin().abs()).clamp(0.0, 2.0),
        w_band: (0.48 + 1.05 * (TAU * (local * 1.2 + drift * 0.07)).cos().abs()).clamp(0.0, 2.0),
    }
}

/// Returns a FieldParams with LFO and mic deltas applied additively from the base.
fn apply_modulation(
    fp:    &FieldParams,
    lfo:   &LfoParams,
    mic:   &MicParams,
    bands: &audio::AudioBands,
    t:     f32,
) -> FieldParams {
    let lfo_s = lfo.wave.sample(lfo.rate * t);
    macro_rules! modulate {
        ($val:expr, $lfo_en:expr, $mic_src:expr, $min:expr, $max:expr) => {{
            let range   = ($max as f32) - ($min as f32);
            let lfo_d   = if $lfo_en { lfo.depth * range * lfo_s } else { 0.0 };
            let mic_d   = $mic_src.value(bands) * mic.depth * range;
            ($val + lfo_d + mic_d).clamp($min as f32, $max as f32)
        }};
    }
    FieldParams {
        mode:        fp.mode,
        kscale:      modulate!(fp.kscale,      lfo.kscale,      mic.kscale,      0.1, 5.0),
        speed:       modulate!(fp.speed,       lfo.speed,       mic.speed,       0.0, 2.0),
        field_mix:   modulate!(fp.field_mix,   lfo.field_mix,   mic.field_mix,   0.0, 1.0),
        iso_level:   modulate!(fp.iso_level,   lfo.iso_level,   mic.iso_level,   0.0, 1.0),
        color_shift: modulate!(fp.color_shift, lfo.color_shift, mic.color_shift, 0.0, 1.0),
        zoom:        modulate!(fp.zoom,        lfo.zoom,        mic.zoom,        0.2, 5.0),
        w_lattice:   modulate!(fp.w_lattice,   lfo.w_lattice,   mic.w_lattice,   0.0, 2.0),
        w_motif:     modulate!(fp.w_motif,     lfo.w_motif,     mic.w_motif,     0.0, 2.0),
        w_band:      modulate!(fp.w_band,      lfo.w_band,      mic.w_band,      0.0, 2.0),
    }
}

// ── Tour state machine ────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum TourPhase { Settling, Walking, Fading }

struct Tour {
    pub active:      bool,
    pub crystal_idx: usize,
    phase:           TourPhase,
    phase_timer:     f32,
    pub prev_field:  Option<GpuField>,
    pub fade_t:      f32,
}

const SETTLE_DUR: f32 = 1.2;
const WALK_DUR:   f32 = 8.0;
const FADE_DUR:   f32 = 2.5;

impl Tour {
    fn new() -> Self {
        Self {
            active: false, crystal_idx: 0,
            phase: TourPhase::Settling, phase_timer: 0.0,
            prev_field: None, fade_t: 0.0,
        }
    }

    /// Returns true when it's time to snapshot and load the next crystal.
    fn tick(&mut self, dt: f32) -> bool {
        if !self.active { return false; }
        self.phase_timer += dt;
        match self.phase {
            TourPhase::Settling => {
                if self.phase_timer >= SETTLE_DUR {
                    self.phase = TourPhase::Walking;
                    self.phase_timer = 0.0;
                }
            }
            TourPhase::Walking => {
                if self.phase_timer >= WALK_DUR {
                    self.phase = TourPhase::Fading;
                    self.phase_timer = 0.0;
                    self.fade_t = 0.0;
                    return true;
                }
            }
            TourPhase::Fading => {
                self.fade_t = (self.phase_timer / FADE_DUR).clamp(0.0, 1.0);
                if self.phase_timer >= FADE_DUR {
                    self.prev_field = None;
                    self.fade_t = 0.0;
                    self.phase = TourPhase::Settling;
                    self.phase_timer = 0.0;
                }
            }
        }
        false
    }

    fn is_fading(&self) -> bool  { self.phase == TourPhase::Fading }
    fn is_walking(&self) -> bool { self.phase == TourPhase::Walking }
}

// ── App ───────────────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum RenderMode { Atoms, Field }

/// Mutations requested by the egui UI within one frame.
#[derive(Default)]
struct UiReq {
    switch_to:    Option<usize>,
    prev:         bool,
    next:         bool,
    screenshot:   bool,
    tour_toggle:  bool,
    tour_style_toggle: bool,
    kpath_toggle: bool,
    panel_toggle: bool,
    render_mode:  Option<RenderMode>,
    seq_toggle:        bool,
    seq_manual_toggle: bool,
    seq_manual_step:   Option<i32>,   // +1 / -1
    seq_select_step:   Option<usize>, // select for editor (toggle)
    seq_mute_step:     Option<usize>, // toggle mute
    seq_randomize:     Option<usize>,
    seq_add_step:      bool,
    seq_remove_step:   bool,
    seq_preset:        Option<usize>,
    seq_curve:         Option<TranCurve>,
    seq_dur:           Option<f32>,
}

struct App {
    gpu:          Option<GpuState>,
    #[cfg(target_arch = "wasm32")]
    event_proxy:  Option<winit::event_loop::EventLoopProxy<UserEvent>>,
    start_crystal: Crystal,
    start:        Instant,
    prev_t:       f32,

    dragging:     bool,
    last_mouse:   Option<(f64, f64)>,
    auto_rotate:  bool,

    render_mode:  RenderMode,
    field_params: FieldParams,
    mouse_norm:   [f32; 2],
    mouse_btn_down: bool,

    kpath_active: bool,
    kpt_idx:      usize,

    tour:         Tour,
    tour_style:   TourStyle,
    sequencer:    Sequencer,
    all_crystals: Vec<&'static CrystalDef>,

    panel_open:   bool,
    search_str:   String,
    lfo:          LfoParams,
    mic_params:   MicParams,
    audio:        Option<audio::AudioCapture>,
}

enum UserEvent {
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    GpuReady(GpuState),
}

impl App {
    fn new(crystal: Crystal) -> Self {
        let all = all_crystals();
        Self {
            gpu: None, start_crystal: crystal,
            #[cfg(target_arch = "wasm32")]
            event_proxy: None,
            start: Instant::now(), prev_t: 0.0,
            dragging: false, last_mouse: None, auto_rotate: true,
            render_mode: RenderMode::Field,
            field_params: FieldParams::default(),
            mouse_norm: [0.5, 0.5], mouse_btn_down: false,
            kpath_active: false, kpt_idx: 0,
            tour: Tour::new(),
            tour_style: TourStyle::default(),
            sequencer: Sequencer::new(),
            all_crystals: all,
            panel_open: true,
            search_str: String::new(),
            lfo: LfoParams::default(),
            mic_params: MicParams::default(),
            audio: audio::AudioCapture::start(),
        }
    }

    fn time(&self) -> f32 { self.start.elapsed().as_secs_f32() }

    fn load_crystal_def(&mut self, def: &CrystalDef) {
        let Some(gpu) = self.gpu.as_mut() else { return };
        let crystal = def.to_crystal();
        let new_field = GpuField::from_crystal(&crystal, 3);
        gpu.gpu_field = new_field;
        gpu.gpu_field.seed_kpoint([0.0, 0.0, 0.0], 1.0);
        gpu.update_field();
        let sys = symmetry::detect(&crystal.lattice);
        gpu.kpath = Some(kpoints::build_kpath(sys));
        self.kpt_idx = 0;
        self.kpath_active = false;
    }

    fn switch_to(&mut self, idx: usize) {
        if let Some(gpu) = &self.gpu {
            self.tour.prev_field = Some(GpuField {
                gvecs: gpu.gpu_field.gvecs.clone(),
                amps:  gpu.gpu_field.amps.clone(),
                phases: gpu.gpu_field.phases.clone(),
                b_mat: gpu.gpu_field.b_mat,
                count: gpu.gpu_field.count,
            });
        }
        self.tour.crystal_idx = idx;
        let def = self.all_crystals[idx];
        self.load_crystal_def(def);
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = WindowAttributes::default()
            .with_title("crystal-viz")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));
        #[cfg(target_arch = "wasm32")]
        let attrs = attrs.with_append(true);
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let crystal = self.start_crystal.clone();

        #[cfg(target_arch = "wasm32")]
        {
            let proxy = self.event_proxy.clone().expect("missing web event proxy");
            wasm_bindgen_futures::spawn_local(async move {
                let mut gpu = GpuState::new(window, crystal).await;
                let sys = gpu.crystal_system();
                gpu.kpath = Some(kpoints::build_kpath(sys));
                let _ = proxy.send_event(UserEvent::GpuReady(gpu));
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
        let mut gpu = pollster::block_on(GpuState::new(window, crystal));
        let sys = gpu.crystal_system();
        gpu.kpath = Some(kpoints::build_kpath(sys));
        self.gpu = Some(gpu);
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::GpuReady(gpu) => {
                gpu.window.request_redraw();
                self.gpu = Some(gpu);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gpu) = &self.gpu {
            gpu.window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // Feed to egui first
        let egui_consumed = self.gpu.as_mut()
            .map(|g| g.handle_ui_event(&event))
            .unwrap_or(false);

        let auto_rotate  = self.auto_rotate;
        let render_mode  = self.render_mode;
        let t            = self.time();
        let Some(gpu) = self.gpu.as_mut() else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{KeyCode, PhysicalKey};
                if event.state != ElementState::Pressed { return }
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                    PhysicalKey::Code(KeyCode::Tab) => {
                        self.render_mode = match self.render_mode {
                            RenderMode::Atoms => RenderMode::Field,
                            RenderMode::Field => RenderMode::Atoms,
                        };
                    }
                    PhysicalKey::Code(KeyCode::Space) => { self.auto_rotate = !self.auto_rotate; }
                    PhysicalKey::Code(KeyCode::Digit1) => gpu.set_supercell(1),
                    PhysicalKey::Code(KeyCode::Digit2) => gpu.set_supercell(2),
                    PhysicalKey::Code(KeyCode::Digit3) => gpu.set_supercell(3),
                    PhysicalKey::Code(KeyCode::KeyM) => {
                        self.field_params.mode = (self.field_params.mode + 1) % MODE_NAMES.len() as u32;
                    }
                    PhysicalKey::Code(KeyCode::KeyK) => {
                        if let Some(kp) = &gpu.kpath {
                            let n = kp.n_points();
                            self.kpt_idx = (self.kpt_idx + 1) % n;
                            let coords = kp.snap_to(self.kpt_idx);
                            gpu.gpu_field.seed_kpoint(coords, 1.0);
                            gpu.update_field();
                        }
                        self.kpath_active = false;
                    }
                    PhysicalKey::Code(KeyCode::KeyP) => {
                        self.kpath_active = !self.kpath_active;
                        if self.kpath_active {
                            if let Some(kp) = &mut gpu.kpath { kp.reset(); }
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        if render_mode == RenderMode::Field {
                            gpu.gpu_field.randomize();
                            gpu.update_field();
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyT) => {
                        self.tour.active = !self.tour.active;
                        if self.tour.active {
                            self.render_mode = RenderMode::Field;
                            self.kpath_active = true;
                            self.lfo = tour_lfo_preset();
                            if let Some(kp) = &mut gpu.kpath { kp.reset(); }
                        }
                    }
                    PhysicalKey::Code(KeyCode::BracketRight) => {
                        self.field_params.color_shift = (self.field_params.color_shift + 0.05) % 1.0;
                    }
                    PhysicalKey::Code(KeyCode::BracketLeft) => {
                        self.field_params.color_shift = (self.field_params.color_shift - 0.05).rem_euclid(1.0);
                    }
                    _ => {}
                }
            }

            WindowEvent::Resized(size) => gpu.resize(size),

            WindowEvent::MouseInput { state, button, .. } if !egui_consumed => {
                if button == MouseButton::Left {
                    let pressed = state == ElementState::Pressed;
                    self.dragging       = pressed;
                    self.mouse_btn_down = pressed;
                    if !pressed { self.last_mouse = None; }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let sz = gpu.size;
                self.mouse_norm = [
                    (position.x as f32 / sz.width  as f32).clamp(0.0, 1.0),
                    (position.y as f32 / sz.height as f32).clamp(0.0, 1.0),
                ];
                if self.dragging && !egui_consumed && render_mode == RenderMode::Atoms {
                    if let Some((lx, ly)) = self.last_mouse {
                        gpu.orbit((position.x - lx) as f32, (position.y - ly) as f32);
                    }
                    self.last_mouse = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p)   => p.y as f32 * 0.05,
                };
                match render_mode {
                    RenderMode::Atoms => gpu.zoom(d),
                    RenderMode::Field => {
                        self.field_params.zoom = (self.field_params.zoom + d * 0.15).clamp(0.2, 5.0);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                gpu.window.request_redraw();

                let dt = (t - self.prev_t).clamp(0.0, 0.1);
                self.prev_t = t;

                // ── Sequencer tick ────────────────────────────────────
                self.sequencer.tick(dt);

                // ── Tour tick ─────────────────────────────────────────
                if self.tour.active && render_mode == RenderMode::Field {
                    let should_advance = self.tour.tick(dt);
                    if should_advance {
                        let total = self.all_crystals.len();
                        let next_idx = (self.tour.crystal_idx + 1) % total;
                        // Snapshot current for crossfade
                        self.tour.prev_field = Some(GpuField {
                            gvecs: gpu.gpu_field.gvecs.clone(),
                            amps:  gpu.gpu_field.amps.clone(),
                            phases: gpu.gpu_field.phases.clone(),
                            b_mat: gpu.gpu_field.b_mat,
                            count: gpu.gpu_field.count,
                        });
                        self.tour.crystal_idx = next_idx;
                        let def = self.all_crystals[next_idx];
                        let crystal = def.to_crystal();
                        gpu.gpu_field = GpuField::from_crystal(&crystal, 3);
                        gpu.gpu_field.seed_kpoint([0.0, 0.0, 0.0], 1.0);
                        let sys = symmetry::detect(&crystal.lattice);
                        gpu.kpath = Some(kpoints::build_kpath(sys));
                        if let Some(kp) = &mut gpu.kpath { kp.reset(); }
                        self.kpt_idx = 0;
                        self.kpath_active = true;
                    }

                    if self.tour.is_walking() && self.kpath_active {
                        if let Some(kp) = &mut gpu.kpath {
                            let (k, _) = kp.tick(dt);
                            gpu.gpu_field.seed_kpoint(k, 0.4);
                        }
                    }

                    if self.tour.is_fading() {
                        if let Some(prev) = &self.tour.prev_field {
                            let packed = crossfade_pack(prev, &gpu.gpu_field, self.tour.fade_t);
                            gpu.field_pl.upload_gfield(&gpu.queue, &packed);
                        } else {
                            gpu.update_field();
                        }
                    } else {
                        gpu.update_field();
                    }
                }

                // ── Non-tour k-path walking ────────────────────────────
                if !self.tour.active && render_mode == RenderMode::Field && self.kpath_active {
                    if let Some(kp) = &mut gpu.kpath {
                        let (k, _) = kp.tick(dt);
                        gpu.gpu_field.seed_kpoint(k, 0.4);
                    }
                    gpu.update_field();
                }

                // ── Snapshot all state read by the UI closure ──────────
                // (We copy/clone so the closure doesn't hold borrows into self.)
                let cur_render_mode   = self.render_mode;
                let cur_tour_active   = self.tour.active;
                let cur_tour_style    = self.tour_style;
                let cur_kpath_active  = self.kpath_active;
                let cur_panel_open    = self.panel_open;
                let cur_crystal_idx   = self.tour.crystal_idx;
                let tour_fp = if self.tour.active {
                    Some(tour_field_params(t, self.tour.crystal_idx, self.tour_style))
                } else {
                    None
                };
                let seq_fp = if self.sequencer.active { Some(self.sequencer.current_params()) } else { None };
                // Sequencer overrides tour which overrides manual field_params
                let cur_mode_idx      = seq_fp.as_ref()
                    .or(tour_fp.as_ref())
                    .map(|fp| fp.mode)
                    .unwrap_or(self.field_params.mode) as usize;
                let cur_mode_name     = MODE_NAMES[cur_mode_idx];
                let cur_crystal_name  = self.all_crystals[cur_crystal_idx].name;
                let cur_sys_name      = self.all_crystals[cur_crystal_idx].system.name();
                let cur_kpt_label: String = gpu.kpath.as_ref()
                    .map(|kp| kp.label_at(self.kpt_idx % kp.n_points()).to_owned())
                    .unwrap_or_default();
                // Snapshot sequencer state for UI
                let cur_seq_active    = self.sequencer.active;
                let cur_seq_manual    = self.sequencer.manual;
                let cur_seq_selected  = self.sequencer.selected;
                let cur_seq_steps: Vec<(bool, bool, bool, u32)> = self.sequencer.steps.iter().enumerate()
                    .map(|(i, s)| (i == self.sequencer.cur, s.muted, Some(i) == self.sequencer.selected, s.params.mode))
                    .collect();
                let cur_seq_dur       = self.sequencer.step_dur;
                let cur_seq_curve     = self.sequencer.curve;
                let cur_seq_progress  = (self.sequencer.step_timer / self.sequencer.step_dur).clamp(0.0, 1.0);
                // Editor: mutable local that closure can modify; written back in mutations
                let mut seq_selected_edit: Option<(usize, FieldParams)> = self.sequencer.selected
                    .and_then(|i| self.sequencer.steps.get(i).map(|s| (i, s.params.clone())));
                // Snapshot field_params, lfo, mic, and current audio bands.
                let mut fp  = seq_fp.or(tour_fp).unwrap_or_else(|| self.field_params.clone());
                let mut lfo = self.lfo.clone();
                let mut mic = self.mic_params.clone();
                let cur_bands = self.audio.as_ref()
                    .and_then(|a| a.bands.lock().ok().map(|b| b.clone()))
                    .unwrap_or_default();
                let mic_active = self.audio.is_some();

                // Apply LFO + mic modulation to get effective values for this frame.
                let fp_eff = apply_modulation(&fp, &lfo, &mic, &cur_bands, t);

                // Pre-build FieldUniform from the LFO-modulated snapshot so the closure can
                // freely mutate fp.* without conflicting with the match arm below.
                let cdef = self.all_crystals[cur_crystal_idx];
                let field_params_uniform = FieldUniform {
                    time:          t,
                    kscale:        fp_eff.kscale,
                    speed:         fp_eff.speed,
                    field_mix:     fp_eff.field_mix,
                    iso_level:     fp_eff.iso_level,
                    color_shift:   fp_eff.color_shift,
                    zoom:          fp_eff.zoom,
                    w_lattice:     fp_eff.w_lattice,
                    w_motif:       fp_eff.w_motif,
                    w_band:        fp_eff.w_band,
                    mode:          fp_eff.mode,
                    num_g:         gpu.gpu_field.count as u32,
                    crystal_color: [cdef.color[0], cdef.color[1], cdef.color[2], 0.0],
                    mouse:         self.mouse_norm,
                    mouse_down:    if self.mouse_btn_down { 1.0 } else { 0.0 },
                    aspect:        gpu.size.width as f32 / gpu.size.height.max(1) as f32,
                };

                // Snapshot search string
                let mut search = self.search_str.clone();

                // Crystal list (static data, no borrows)
                let groups   = all_groups();
                let all_defs = all_crystals();

                // Accumulator for mutations
                let mut req = UiReq::default();
                let req_panel_open = cur_panel_open;

                // The closure now captures &mut fp.* freely — no conflict with match below.
                let ui_fn = |ctx: &egui::Context| {
                    // Side panel
                    if req_panel_open {
                        egui::SidePanel::left("ctrl")
                            .min_width(260.0).max_width(300.0)
                            .resizable(false)
                            .show(ctx, |ui| {
                                ui.heading("Crystal Field Synthesizer");
                                ui.separator();

                                // Crystal library
                                ui.label(egui::RichText::new("CRYSTAL LIBRARY")
                                    .small().color(egui::Color32::from_rgb(120, 120, 160)));
                                ui.text_edit_singleline(&mut search);
                                let q = search.to_ascii_lowercase();

                                egui::ScrollArea::vertical()
                                    .max_height(180.0)
                                    .show(ui, |ui| {
                                        for (label, color, defs) in &groups {
                                            ui.colored_label(*color, *label);
                                            for def in defs.iter() {
                                                if !q.is_empty() && !def.name.to_ascii_lowercase().contains(&q) {
                                                    continue;
                                                }
                                                let flat_idx = all_defs.iter().position(|d| std::ptr::eq(*d, def as &CrystalDef));
                                                let is_sel = flat_idx == Some(cur_crystal_idx);
                                                let lbl = egui::RichText::new(format!("  {}", def.name))
                                                    .color(if is_sel {
                                                        egui::Color32::from_rgb(200, 200, 255)
                                                    } else {
                                                        egui::Color32::from_rgb(150, 150, 190)
                                                    });
                                                if ui.selectable_label(is_sel, lbl).clicked() {
                                                    req.switch_to = flat_idx;
                                                }
                                            }
                                        }
                                    });

                                ui.separator();

                                // Render mode buttons
                                ui.label(egui::RichText::new("RENDER MODE")
                                    .small().color(egui::Color32::from_rgb(120, 120, 160)));
                                ui.horizontal_wrapped(|ui| {
                                    for (i, name) in MODE_NAMES.iter().enumerate() {
                                        if ui.selectable_label(fp.mode == i as u32, *name).clicked() {
                                            fp.mode = i as u32;
                                        }
                                    }
                                });

                                ui.separator();

                                // Sliders — [~] LFO  [·/A/B/M/T] mic  label  [====slider====]  eff
                                ui.label(egui::RichText::new("FIELD PARAMETERS")
                                    .small().color(egui::Color32::from_rgb(120, 120, 160)));
                                macro_rules! sld {
                                    ($ui:expr, $label:literal, $val:expr, $eff:expr,
                                     $lfo_en:expr, $mic_src:expr, $min:expr, $max:expr) => {
                                        $ui.horizontal(|ui| {
                                            // [~] LFO toggle — normal sized button, coloured text
                                            let lc = if *$lfo_en {
                                                egui::Color32::from_rgb(80, 220, 120)
                                            } else {
                                                egui::Color32::from_gray(90)
                                            };
                                            if ui.add(egui::Button::new(
                                                egui::RichText::new("~").color(lc)
                                            ).min_size(egui::vec2(18.0, 18.0))).clicked() {
                                                *$lfo_en = !*$lfo_en;
                                            }

                                            // [·/A/B/M/T] mic source — cycles on click
                                            let mc = (*$mic_src).color();
                                            let ml = (*$mic_src).label();
                                            if ui.add(egui::Button::new(
                                                egui::RichText::new(ml).color(mc)
                                            ).min_size(egui::vec2(18.0, 18.0))).clicked() {
                                                let ns = (*$mic_src).next();
                                                *$mic_src = ns;
                                            }

                                            ui.label(egui::RichText::new($label).small());
                                            ui.add(egui::Slider::new($val, $min..=$max)
                                                .show_value(false));

                                            // effective value: dim if same as base, orange if modulated
                                            let eff_v: f32 = $eff;
                                            let base_v: f32 = *$val;
                                            let modulated = (eff_v - base_v).abs() > 0.001;
                                            let eff_col = if modulated {
                                                egui::Color32::from_rgb(255, 160, 60)
                                            } else {
                                                egui::Color32::from_gray(130)
                                            };
                                            ui.label(egui::RichText::new(
                                                format!("{:.2}", eff_v)
                                            ).small().color(eff_col));
                                        });
                                    }
                                }
                                sld!(ui, "kscale   ", &mut fp.kscale,      fp_eff.kscale,      &mut lfo.kscale,      &mut mic.kscale,      0.1_f32, 5.0_f32);
                                sld!(ui, "speed    ", &mut fp.speed,       fp_eff.speed,       &mut lfo.speed,       &mut mic.speed,       0.0_f32, 2.0_f32);
                                sld!(ui, "field_mix", &mut fp.field_mix,   fp_eff.field_mix,   &mut lfo.field_mix,   &mut mic.field_mix,   0.0_f32, 1.0_f32);
                                sld!(ui, "iso_level", &mut fp.iso_level,   fp_eff.iso_level,   &mut lfo.iso_level,   &mut mic.iso_level,   0.0_f32, 1.0_f32);
                                sld!(ui, "color_sft", &mut fp.color_shift, fp_eff.color_shift, &mut lfo.color_shift, &mut mic.color_shift, 0.0_f32, 1.0_f32);
                                sld!(ui, "zoom     ", &mut fp.zoom,        fp_eff.zoom,        &mut lfo.zoom,        &mut mic.zoom,        0.2_f32, 5.0_f32);
                                sld!(ui, "w_lattice", &mut fp.w_lattice,   fp_eff.w_lattice,   &mut lfo.w_lattice,   &mut mic.w_lattice,   0.0_f32, 2.0_f32);
                                sld!(ui, "w_motif  ", &mut fp.w_motif,     fp_eff.w_motif,     &mut lfo.w_motif,     &mut mic.w_motif,     0.0_f32, 2.0_f32);
                                sld!(ui, "w_band   ", &mut fp.w_band,      fp_eff.w_band,      &mut lfo.w_band,      &mut mic.w_band,      0.0_f32, 2.0_f32);

                                ui.separator();

                                // LFO section
                                ui.label(egui::RichText::new(format!("LFO  ({})", lfo.wave.label()))
                                    .small().color(egui::Color32::from_rgb(80, 220, 120)));
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("wave ").small());
                                    if ui.button(lfo.wave.label()).clicked() {
                                        lfo.wave = lfo.wave.next();
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("rate ").small());
                                    ui.add(egui::Slider::new(&mut lfo.rate, 0.01..=4.0)
                                        .show_value(true).suffix(" Hz"));
                                });
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("depth").small());
                                    ui.add(egui::Slider::new(&mut lfo.depth, 0.0..=1.0)
                                        .show_value(true));
                                });

                                ui.separator();

                                // Mic section
                                let mic_col = if mic_active {
                                    egui::Color32::from_rgb(255, 140, 80)
                                } else {
                                    egui::Color32::from_gray(100)
                                };
                                ui.label(egui::RichText::new(if mic_active { "MIC  (live)" } else { "MIC  (no device)" })
                                    .small().color(mic_col));
                                if mic_active {
                                    // Band meters
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new("A").small()
                                            .color(egui::Color32::from_rgb(220,220,220)));
                                        ui.add(egui::ProgressBar::new(cur_bands.amplitude).desired_width(38.0));
                                        ui.label(egui::RichText::new("B").small()
                                            .color(egui::Color32::from_rgb(255,80,80)));
                                        ui.add(egui::ProgressBar::new(cur_bands.bass).desired_width(38.0));
                                        ui.label(egui::RichText::new("M").small()
                                            .color(egui::Color32::from_rgb(80,220,120)));
                                        ui.add(egui::ProgressBar::new(cur_bands.mid).desired_width(38.0));
                                        ui.label(egui::RichText::new("T").small()
                                            .color(egui::Color32::from_rgb(80,160,255)));
                                        ui.add(egui::ProgressBar::new(cur_bands.treble).desired_width(38.0));
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new("depth").small());
                                        ui.add(egui::Slider::new(&mut mic.depth, 0.0..=2.0).show_value(true));
                                    });
                                    ui.label(egui::RichText::new("Click · on a slider to pick source")
                                        .small().color(egui::Color32::from_gray(130)));
                                }

                                ui.separator();

                                // K-path controls
                                ui.label(egui::RichText::new("K-PATH")
                                    .small().color(egui::Color32::from_rgb(120, 120, 160)));
                                ui.horizontal(|ui| {
                                    ui.monospace(format!("k = {cur_kpt_label}"));
                                    if ui.button(if cur_kpath_active { "■ STOP" } else { "▶ WALK" }).clicked() {
                                        req.kpath_toggle = true;
                                    }
                                });

                                ui.separator();

                                // Mode toggle
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(cur_render_mode == RenderMode::Atoms, "ATOMS").clicked() {
                                        req.render_mode = Some(RenderMode::Atoms);
                                    }
                                    if ui.selectable_label(cur_render_mode == RenderMode::Field, "FIELD").clicked() {
                                        req.render_mode = Some(RenderMode::Field);
                                    }
                                });

                                ui.separator();

                                // Tour controls
                                ui.horizontal(|ui| {
                                    if ui.button("◀ PREV").clicked() { req.prev = true; }
                                    let tour_lbl = if cur_tour_active { "■ STOP" } else { "▶ TOUR" };
                                    if ui.button(tour_lbl).clicked() { req.tour_toggle = true; }
                                    if ui.button(cur_tour_style.label()).clicked() { req.tour_style_toggle = true; }
                                    if ui.button("NEXT ▶").clicked() { req.next = true; }
                                });

                                ui.separator();

                                // Sequencer controls (panel section)
                                ui.label(egui::RichText::new("SEQUENCER")
                                    .small().color(egui::Color32::from_rgb(180, 140, 255)));
                                ui.horizontal(|ui| {
                                    let seq_lbl = if cur_seq_active { "■ STOP" } else { "▶ SEQ" };
                                    if ui.button(seq_lbl).clicked() { req.seq_toggle = true; }
                                    let man_col = if cur_seq_manual {
                                        egui::Color32::from_rgb(255, 200, 60)
                                    } else { egui::Color32::from_gray(150) };
                                    if ui.add(egui::Button::new(
                                        egui::RichText::new("MANUAL").color(man_col)
                                    )).clicked() { req.seq_manual_toggle = true; }
                                    if cur_seq_manual {
                                        if ui.button("◀").clicked() { req.seq_manual_step = Some(-1); }
                                        if ui.button("▶").clicked() { req.seq_manual_step = Some(1); }
                                    }
                                });
                                ui.horizontal(|ui| {
                                    if ui.button(cur_seq_curve.label()).clicked() {
                                        req.seq_curve = Some(cur_seq_curve.next());
                                    }
                                    ui.label(egui::RichText::new("step").small());
                                    let mut dur = cur_seq_dur;
                                    if ui.add(egui::Slider::new(&mut dur, 0.5_f32..=8.0_f32)
                                        .show_value(true).suffix("s")).changed() {
                                        req.seq_dur = Some(dur);
                                    }
                                });
                                ui.horizontal(|ui| {
                                    for (i, (name, _)) in SEQ_PRESETS.iter().enumerate() {
                                        if ui.small_button(*name).clicked() { req.seq_preset = Some(i); }
                                    }
                                });
                                ui.horizontal(|ui| {
                                    let n = cur_seq_steps.len();
                                    ui.label(egui::RichText::new(format!("{n}/32 steps")).small()
                                        .color(egui::Color32::from_gray(160)));
                                    let can_add = n < 32;
                                    let can_del = n > 1 && cur_seq_selected.is_some();
                                    if ui.add_enabled(can_add, egui::Button::new("+")).clicked() {
                                        req.seq_add_step = true;
                                    }
                                    if ui.add_enabled(can_del, egui::Button::new("−")).clicked() {
                                        req.seq_remove_step = true;
                                    }
                                });

                                if ui.button("📷 Screenshot").clicked() {
                                    req.screenshot = true;
                                }
                            });
                    }

                    // Panel toggle button
                    egui::Area::new("panel_toggle".into())
                        .fixed_pos(egui::pos2(4.0, 4.0))
                        .show(ctx, |ui| {
                            if ui.button(if req_panel_open { "◀" } else { "▶" }).clicked() {
                                req.panel_toggle = true;
                            }
                        });

                    // Sequencer grid (bottom bar)
                    if cur_seq_active || !cur_seq_steps.is_empty() {
                        egui::Area::new("seq_grid".into())
                            .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -8.0))
                            .show(ctx, |ui| {
                                ui.visuals_mut().widgets.inactive.bg_fill =
                                    egui::Color32::from_rgba_unmultiplied(10, 8, 20, 220);
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgba_unmultiplied(12, 10, 24, 220))
                                    .rounding(6.0)
                                    .inner_margin(egui::vec2(8.0, 6.0))
                                    .show(ui, |ui| {
                                        let n = cur_seq_steps.len();
                                        let cell_w = (440.0_f32 / n as f32).min(38.0).max(18.0);
                                        let cell_h = 40.0_f32;

                                        // Pass 1: allocate rects, collect interaction
                                        let cells = ui.horizontal(|ui| {
                                            let mut cells: Vec<(usize, egui::Rect, bool, bool, bool, u32)> = Vec::new();
                                            for (i, &(is_cur, muted, is_sel, mode)) in cur_seq_steps.iter().enumerate() {
                                                let (rect, resp) = ui.allocate_exact_size(
                                                    egui::vec2(cell_w, cell_h),
                                                    egui::Sense::click(),
                                                );
                                                if resp.double_clicked() {
                                                    req.seq_randomize = Some(i);
                                                } else if resp.clicked() {
                                                    req.seq_select_step = Some(i);
                                                }
                                                cells.push((i, rect, is_cur, muted, is_sel, mode));
                                                ui.add_space(3.0);
                                            }
                                            cells
                                        }).inner;

                                        // Pass 2: paint
                                        let painter = ui.painter();
                                        for &(_i, rect, is_cur, muted, is_sel, mode) in &cells {
                                            let base_col = mode_color(mode);
                                            let dim = if muted { 4 } else { 1 };
                                            let fill = egui::Color32::from_rgba_unmultiplied(
                                                base_col.r() / dim, base_col.g() / dim, base_col.b() / dim, 210);
                                            painter.rect_filled(rect, 4.0, fill);

                                            if is_cur && cur_seq_active {
                                                let prog_rect = egui::Rect::from_min_size(
                                                    rect.min,
                                                    egui::vec2(rect.width() * cur_seq_progress, rect.height()),
                                                );
                                                painter.rect_filled(prog_rect, 4.0,
                                                    egui::Color32::from_white_alpha(50));
                                                painter.rect_stroke(rect, 4.0,
                                                    egui::Stroke::new(2.0, egui::Color32::WHITE));
                                            }

                                            if is_sel {
                                                painter.rect_stroke(rect, 4.0,
                                                    egui::Stroke::new(1.5,
                                                        egui::Color32::from_rgb(255, 210, 60)));
                                            }

                                            if muted {
                                                let s = egui::Stroke::new(1.5, egui::Color32::from_gray(80));
                                                painter.line_segment([rect.min, rect.max], s);
                                                painter.line_segment(
                                                    [egui::pos2(rect.max.x, rect.min.y),
                                                     egui::pos2(rect.min.x, rect.max.y)], s);
                                            }

                                            let label = MODE_NAMES[mode as usize].chars().take(3).collect::<String>();
                                            let txt_col = if muted { egui::Color32::from_gray(70) }
                                                else if is_cur && cur_seq_active { egui::Color32::WHITE }
                                                else { egui::Color32::from_gray(220) };
                                            painter.text(rect.center(), egui::Align2::CENTER_CENTER,
                                                label, egui::FontId::monospace(9.0), txt_col);
                                        }

                                        // Status row
                                        ui.horizontal(|ui| {
                                            let status_col = if cur_seq_active {
                                                egui::Color32::from_rgb(140, 255, 140)
                                            } else {
                                                egui::Color32::from_gray(120)
                                            };
                                            ui.label(egui::RichText::new(
                                                if cur_seq_active { "▶ SEQ" } else { "■ SEQ" }
                                            ).monospace().size(10.0).color(status_col));
                                            ui.label(egui::RichText::new(
                                                format!("  {}  {:.1}s", cur_seq_curve.label(), cur_seq_dur)
                                            ).monospace().size(10.0).color(egui::Color32::from_gray(160)));
                                        });
                                    });
                            });
                    }

                    // Step editor window
                    if let Some((edit_idx, ref mut ep)) = seq_selected_edit {
                        egui::Window::new(format!("STEP {}  —  {}", edit_idx + 1, MODE_NAMES[ep.mode as usize]))
                            .id(egui::Id::new("step_editor"))
                            .collapsible(false)
                            .resizable(false)
                            .default_width(220.0)
                            .show(ctx, |ui| {
                                // Mode selector
                                egui::ComboBox::from_label("mode")
                                    .selected_text(MODE_NAMES[ep.mode as usize])
                                    .show_ui(ui, |ui| {
                                        for (i, name) in MODE_NAMES.iter().enumerate() {
                                            ui.selectable_value(&mut ep.mode, i as u32, *name);
                                        }
                                    });
                                ui.separator();
                                // Param sliders
                                macro_rules! esl {
                                    ($ui:expr, $label:expr, $val:expr, $min:expr, $max:expr) => {
                                        $ui.horizontal(|ui| {
                                            ui.label(egui::RichText::new($label).small().monospace());
                                            ui.add(egui::Slider::new($val, $min..=$max).show_value(true));
                                        });
                                    }
                                }
                                esl!(ui, "kscale   ", &mut ep.kscale,      0.1_f32, 5.0_f32);
                                esl!(ui, "speed    ", &mut ep.speed,       0.0_f32, 2.0_f32);
                                esl!(ui, "field_mix", &mut ep.field_mix,   0.0_f32, 1.0_f32);
                                esl!(ui, "iso_level", &mut ep.iso_level,   0.0_f32, 1.0_f32);
                                esl!(ui, "color_sft", &mut ep.color_shift, 0.0_f32, 1.0_f32);
                                esl!(ui, "zoom     ", &mut ep.zoom,        0.2_f32, 5.0_f32);
                                esl!(ui, "w_lattice", &mut ep.w_lattice,   0.0_f32, 2.0_f32);
                                esl!(ui, "w_motif  ", &mut ep.w_motif,     0.0_f32, 2.0_f32);
                                esl!(ui, "w_band   ", &mut ep.w_band,      0.0_f32, 2.0_f32);
                                ui.separator();
                                ui.horizontal(|ui| {
                                    let is_muted = cur_seq_steps.get(edit_idx)
                                        .map(|&(_, m, _, _)| m).unwrap_or(false);
                                    let mute_col = if is_muted {
                                        egui::Color32::from_rgb(255, 80, 80)
                                    } else { egui::Color32::from_gray(160) };
                                    if ui.add(egui::Button::new(
                                        egui::RichText::new(if is_muted { "MUTED" } else { "MUTE" }).color(mute_col)
                                    )).clicked() { req.seq_mute_step = Some(edit_idx); }
                                    if ui.button("RAND").clicked() { req.seq_randomize = Some(edit_idx); }
                                });
                            });
                    }

                    // OSD
                    let osd_offset = if cur_seq_active { -78.0 } else { -12.0 };
                    egui::Area::new("osd".into())
                        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, osd_offset))
                        .show(ctx, |ui| {
                            ui.visuals_mut().override_text_color =
                                Some(egui::Color32::from_rgba_unmultiplied(210, 210, 255, 210));
                            ui.label(
                                egui::RichText::new(format!(
                                    "{cur_crystal_name}  ·  {cur_sys_name}  ·  {cur_mode_name}  ·  k={cur_kpt_label}"
                                )).monospace().size(11.0)
                            );
                        });
                };

                // ── Render ────────────────────────────────────────────
                match cur_render_mode {
                    RenderMode::Atoms => {
                        if auto_rotate { gpu.orbit(0.4, 0.0); }
                        match gpu.render(t, ui_fn) {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                let sz = gpu.size; gpu.resize(sz);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("OOM"); event_loop.exit();
                            }
                            Err(e) => log::warn!("render: {e:?}"),
                        }
                    }
                    RenderMode::Field => {
                        // field_params_uniform was pre-built from `fp` snapshot before the closure.
                        match gpu.render_field(&field_params_uniform, ui_fn) {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                let sz = gpu.size; gpu.resize(sz);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("OOM"); event_loop.exit();
                            }
                            Err(e) => log::warn!("field: {e:?}"),
                        }
                    }
                }

                // ── Apply UI mutations (closure is dropped, borrows released) ──
                self.field_params = fp;
                self.lfo          = lfo;
                self.mic_params   = mic;
                self.search_str   = search;
                if req.panel_toggle { self.panel_open = !cur_panel_open; }

                if let Some(rm) = req.render_mode { self.render_mode = rm; }
                if req.kpath_toggle {
                    self.kpath_active = !cur_kpath_active;
                    if self.kpath_active {
                        if let Some(gpu2) = &mut self.gpu {
                            if let Some(kp) = &mut gpu2.kpath { kp.reset(); }
                        }
                    }
                }
                if req.tour_toggle {
                    self.tour.active = !cur_tour_active;
                    if self.tour.active {
                        self.render_mode = RenderMode::Field;
                        self.kpath_active = true;
                        self.lfo = tour_lfo_preset();
                        if let Some(gpu2) = &mut self.gpu {
                            if let Some(kp) = &mut gpu2.kpath { kp.reset(); }
                        }
                    }
                }
                if req.tour_style_toggle {
                    self.tour_style = self.tour_style.next();
                    self.lfo.wave = match self.tour_style {
                        TourStyle::Curated => LfoWave::Triangle,
                        TourStyle::Random => LfoWave::Steps,
                    };
                    self.lfo.depth = match self.tour_style {
                        TourStyle::Curated => 0.18,
                        TourStyle::Random => 0.28,
                    };
                }
                if req.seq_toggle {
                    self.sequencer.active = !cur_seq_active;
                    if self.sequencer.active { self.render_mode = RenderMode::Field; }
                }
                if req.seq_manual_toggle {
                    self.sequencer.manual = !cur_seq_manual;
                    // Enabling manual mode must activate the sequencer so seq_fp is Some(...)
                    if self.sequencer.manual {
                        self.sequencer.active = true;
                        self.render_mode = RenderMode::Field;
                    }
                }
                if let Some(dir) = req.seq_manual_step { self.sequencer.manual_step(dir); }
                if let Some(i) = req.seq_select_step {
                    self.sequencer.selected = if self.sequencer.selected == Some(i) { None } else { Some(i) };
                }
                if let Some(i) = req.seq_mute_step {
                    if i < self.sequencer.steps.len() {
                        self.sequencer.steps[i].muted = !self.sequencer.steps[i].muted;
                    }
                }
                // Randomize: update step AND editor local so write-back doesn't overwrite it
                if let Some(i) = req.seq_randomize {
                    let rp = randomize_fp(t + i as f32 * 17.7);
                    if i < self.sequencer.steps.len() {
                        self.sequencer.steps[i].params = rp.clone();
                    }
                    if seq_selected_edit.as_ref().map_or(false, |&(ei, _)| ei == i) {
                        seq_selected_edit = Some((i, rp));
                    }
                }
                // Write editor params back (randomize already updated if applicable)
                if let Some((i, ref params)) = seq_selected_edit {
                    if i < self.sequencer.steps.len() {
                        self.sequencer.steps[i].params = params.clone();
                    }
                }
                if req.seq_add_step {
                    let after = self.sequencer.selected.unwrap_or(self.sequencer.cur);
                    self.sequencer.add_step_after(after, t);
                    self.sequencer.selected = Some((after + 1).min(self.sequencer.steps.len() - 1));
                }
                if req.seq_remove_step {
                    if let Some(sel) = self.sequencer.selected {
                        self.sequencer.remove_step(sel);
                    }
                }
                if let Some(i) = req.seq_preset {
                    if i < SEQ_PRESETS.len() { self.sequencer.load_preset(SEQ_PRESETS[i].1); }
                }
                if let Some(c) = req.seq_curve { self.sequencer.curve = c; }
                if let Some(d) = req.seq_dur { self.sequencer.step_dur = d; }
                if req.prev {
                    self.tour.active = false;
                    let total = self.all_crystals.len();
                    let idx = cur_crystal_idx.checked_sub(1).unwrap_or(total - 1);
                    self.switch_to(idx);
                }
                if req.next {
                    self.tour.active = false;
                    let total = self.all_crystals.len();
                    let idx = (cur_crystal_idx + 1) % total;
                    self.switch_to(idx);
                }
                if let Some(idx) = req.switch_to {
                    self.tour.active = false;
                    self.switch_to(idx);
                }
                if req.screenshot {
                    #[cfg(target_arch = "wasm32")]
                    {
                        log::warn!("Screenshots are not wired for the browser build yet");
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(gpu2) = &mut self.gpu {
                        let png = gpu2.screenshot_field(&field_params_uniform);
                        let path = format!("crystal-viz-{}.png", chrono_stamp());
                        match std::fs::write(&path, &png) {
                            Ok(_)  => log::info!("Saved screenshot → {path}"),
                            Err(e) => log::error!("Screenshot save failed: {e}"),
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

// ── entry point ───────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn chrono_stamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    format!("{secs}")
}

fn default_crystal() -> Crystal {
    all_crystals()[0].to_crystal()
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();

    let crystal = if let Some(path) = std::env::args().nth(1) {
        poscar::Crystal::from_file(&path).unwrap_or_else(|e| {
            eprintln!("Error loading '{path}': {e}");
            std::process::exit(1);
        })
    } else if std::path::Path::new("POSCAR").exists() {
        poscar::Crystal::from_file("POSCAR").unwrap_or_else(|e| {
            eprintln!("Error loading 'POSCAR': {e}");
            std::process::exit(1);
        })
    } else {
        // No POSCAR supplied — start from the built-in crystal library
        default_crystal()
    };
    println!("Loaded {} atoms", crystal.atoms.len());
    println!("Controls:");
    println!("  Tab      — Atoms / Field toggle");
    println!("  T        — auto-tour on/off");
    println!("  M        — cycle render mode  P — k-path walk  K — next k-pt");
    println!("  [ / ]    — colour shift       scroll — zoom");
    println!("  1/2/3    — supercell  Space — auto-rotate  (atom mode)");

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new(crystal)).unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(default_crystal());
    app.event_proxy = Some(event_loop.create_proxy());
    event_loop.spawn_app(app);
}

#[cfg(target_arch = "wasm32")]
fn main() {}

// ── Sequencer unit tests ──────────────────────────────────────────────────
#[cfg(test)]
mod seq_tests {
    use super::*;

    fn make_seq(n: usize) -> Sequencer {
        let steps = (0..n).map(|i| SeqStep::new(FieldParams {
            mode: i as u32 % MODE_NAMES.len() as u32,
            kscale: 1.0 + i as f32 * 0.1,
            speed: 0.5,
            field_mix: i as f32 / n as f32,
            iso_level: 0.5,
            color_shift: i as f32 / n as f32,
            zoom: 1.0,
            w_lattice: 1.0, w_motif: 0.5, w_band: 0.5,
        })).collect::<Vec<_>>();
        let from_params = steps[0].params.clone();
        Sequencer {
            active: true, manual: true,
            cur: 0, step_dur: 2.0, step_timer: 2.0, // fully arrived at step 0
            curve: TranCurve::Linear,
            selected: None,
            steps,
            from_params,
        }
    }

    // manual_step(+1) advances cur and resets timer
    #[test]
    fn manual_step_forward_advances_cur() {
        let mut seq = make_seq(4);
        assert_eq!(seq.cur, 0);
        seq.manual_step(1);
        assert_eq!(seq.cur, 1);
        assert_eq!(seq.step_timer, 0.0);
    }

    // manual_step(-1) from cur=0 wraps to last step
    #[test]
    fn manual_step_prev_wraps_around() {
        let mut seq = make_seq(4);
        seq.manual_step(-1);
        assert_eq!(seq.cur, 3, "prev from 0 should wrap to last");
        assert_eq!(seq.step_timer, 0.0);
    }

    // manual_step(+1) wraps from last to first
    #[test]
    fn manual_step_forward_wraps_around() {
        let mut seq = make_seq(4);
        seq.cur = 3;
        seq.manual_step(1);
        assert_eq!(seq.cur, 0, "next from last should wrap to first");
    }

    // manual_step skips muted steps going forward
    #[test]
    fn manual_step_skips_muted_forward() {
        let mut seq = make_seq(4);
        seq.steps[1].muted = true; // step 1 is muted
        seq.manual_step(1);
        assert_eq!(seq.cur, 2, "should skip muted step 1, land on 2");
    }

    // manual_step skips muted steps going backward
    #[test]
    fn manual_step_skips_muted_backward() {
        let mut seq = make_seq(4);
        seq.cur = 2;
        seq.steps[1].muted = true;
        seq.manual_step(-1);
        assert_eq!(seq.cur, 0, "going back from 2, step 1 muted, should land on 0");
    }

    // tick in manual mode clamps timer, never auto-advances cur
    #[test]
    fn tick_manual_clamps_timer_no_advance() {
        let mut seq = make_seq(4);
        seq.step_timer = 0.0;
        seq.tick(1.0); // half of step_dur=2.0
        assert_eq!(seq.cur, 0, "manual mode should not auto-advance");
        assert!((seq.step_timer - 1.0).abs() < 1e-6);
        seq.tick(5.0); // big dt — should clamp, not overflow
        assert_eq!(seq.cur, 0);
        assert!((seq.step_timer - 2.0).abs() < 1e-6, "should clamp at step_dur");
    }

    // tick in auto mode advances cur when timer expires
    #[test]
    fn tick_auto_advances_cur_on_expiry() {
        let mut seq = make_seq(4);
        seq.manual = false;
        seq.step_timer = 0.0;
        seq.tick(1.9); // just under step_dur
        assert_eq!(seq.cur, 0, "not yet expired");
        seq.tick(0.2); // now 2.1, over step_dur=2.0
        assert_eq!(seq.cur, 1, "should auto-advance to step 1");
    }

    // tick when not active is a no-op
    #[test]
    fn tick_inactive_is_noop() {
        let mut seq = make_seq(4);
        seq.active = false;
        seq.step_timer = 0.0;
        seq.tick(10.0);
        assert_eq!(seq.cur, 0);
        assert_eq!(seq.step_timer, 0.0);
    }

    // current_params at step_timer=0 returns from_params (t=0)
    #[test]
    fn current_params_at_t0_is_from() {
        let mut seq = make_seq(4);
        seq.step_timer = 0.0;
        // from_params is steps[0]; cur is also 0 initially
        // After manual_step: from=steps[0], cur=1, timer=0 → lerp at t=0 = from
        seq.manual_step(1);
        let p = seq.current_params();
        assert!((p.kscale - seq.from_params.kscale).abs() < 1e-5,
            "at t=0 current_params should equal from_params");
    }

    // current_params at step_timer=step_dur returns the destination (t=1)
    #[test]
    fn current_params_at_t1_is_destination() {
        let mut seq = make_seq(4);
        seq.manual_step(1); // cur=1, from=steps[0], timer=0
        seq.step_timer = seq.step_dur; // t=1
        let p = seq.current_params();
        let target = &seq.steps[seq.cur].params;
        assert!((p.kscale - target.kscale).abs() < 1e-5,
            "at t=1 current_params should equal target step params");
    }

    // add_step_after inserts at correct position and adjusts cur
    #[test]
    fn add_step_after_inserts_correctly() {
        let mut seq = make_seq(3); // steps 0,1,2
        seq.cur = 2;
        seq.add_step_after(0, 42.0); // insert copy of step 0 after index 0
        assert_eq!(seq.steps.len(), 4);
        // cur was 2, insert at pos 1 → cur should shift to 3
        assert_eq!(seq.cur, 3, "cur should shift when insert is before it");
        // new step at index 1 should be a copy of old step 0
        assert_eq!(seq.steps[1].params.mode, seq.steps[0].params.mode);
    }

    // remove_step shrinks len, adjusts cur and selected
    #[test]
    fn remove_step_adjusts_cur() {
        let mut seq = make_seq(4); // steps 0,1,2,3
        seq.cur = 2;
        seq.selected = Some(3);
        seq.remove_step(0); // remove step 0
        assert_eq!(seq.steps.len(), 3);
        assert_eq!(seq.cur, 1, "cur should shift down by 1");
        assert_eq!(seq.selected, Some(2), "selected should shift down by 1");
    }

    #[test]
    fn remove_selected_step_clears_selection() {
        let mut seq = make_seq(4);
        seq.selected = Some(1);
        seq.remove_step(1);
        assert_eq!(seq.selected, None, "removing selected step clears selection");
    }

    // All-muted: advance_next should not infinite-loop
    #[test]
    fn advance_next_all_muted_no_infinite_loop() {
        let mut seq = make_seq(3);
        for s in seq.steps.iter_mut() { s.muted = true; }
        seq.advance_next(); // should terminate
        // cur ends up at (0+1)%3=1 (muted, but no hang)
        assert!(seq.cur < seq.steps.len());
    }

    // Cannot remove when only 1 step
    #[test]
    fn remove_step_guards_min_one() {
        let mut seq = make_seq(1);
        seq.remove_step(0);
        assert_eq!(seq.steps.len(), 1, "cannot remove last step");
    }

    // Cannot add beyond 32 steps
    #[test]
    fn add_step_guards_max_32() {
        let mut seq = make_seq(32);
        seq.add_step_after(0, 1.0);
        assert_eq!(seq.steps.len(), 32, "cannot exceed 32 steps");
    }

    // BUG: manual stepping requires active=true for seq_fp to be Some(...)
    // This test documents the requirement: manual=true alone is not enough
    #[test]
    fn manual_step_requires_active_for_rendering() {
        let mut seq = make_seq(4);
        seq.active = false; // NOT active
        seq.manual = true;
        seq.manual_step(1);
        // cur advances internally...
        assert_eq!(seq.cur, 1, "cur changes regardless of active");
        // ...but the Sequencer's own logic is correct.
        // The regression: App only uses seq_fp when active=true.
        // Enabling MANUAL must also set active=true (fixed in App mutations).
    }
}
