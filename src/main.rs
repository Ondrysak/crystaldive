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

mod midi;
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
    pub mode:           u32,
    pub kscale:         f32,
    pub speed:          f32,
    pub field_mix:      f32,
    pub iso_level:      f32,
    pub color_shift:    f32,
    pub zoom:           f32,
    pub w_lattice:      f32,
    pub w_motif:        f32,
    pub w_band:         f32,
    // feedback
    pub fb_enabled:     bool,
    pub fb_mirror:      u32,
    pub fb_zoom:        f32,
    pub fb_offset_x:    f32,
    pub fb_offset_y:    f32,
    pub fb_rotation:    f32,
    pub fb_decay:       f32,
    pub fb_color_shift: f32,
    pub fb_inject:      f32,
    pub fb_fold_angle:  f32,
    pub fb_saturation:  f32,
    pub fb_brightness:  f32,
    pub fb_blend_mode:  u32,
    pub fb_motion_blur: f32,
}

impl Default for FieldParams {
    fn default() -> Self {
        Self {
            mode: 4, kscale: 1.4, speed: 0.3, field_mix: 0.55,
            iso_level: 0.5, color_shift: 0.0, zoom: 1.0,
            w_lattice: 1.0, w_motif: 0.6, w_band: 0.4,
            fb_enabled: false, fb_mirror: 0,
            fb_zoom: 0.98, fb_offset_x: 0.0, fb_offset_y: 0.0,
            fb_rotation: 0.0, fb_decay: 0.85, fb_color_shift: 0.0, fb_inject: 1.0,
            fb_fold_angle: 0.0, fb_saturation: 1.0, fb_brightness: 1.0, fb_blend_mode: 0,
            fb_motion_blur: 0.0,
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

#[derive(Clone, Copy, PartialEq, Default, Debug)]
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

/// Which LFO bank drives a target parameter.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub enum LfoSrc {
    #[default]
    Off,
    A,
    B,
}

impl LfoSrc {
    fn next(self) -> Self {
        match self { Self::Off => Self::A, Self::A => Self::B, Self::B => Self::Off }
    }
    fn label(self) -> &'static str {
        match self { Self::Off => "~", Self::A => "A", Self::B => "B" }
    }
    fn color(self) -> egui::Color32 {
        match self {
            Self::Off => egui::Color32::from_gray(90),
            Self::A   => egui::Color32::from_rgb(80, 220, 120),
            Self::B   => egui::Color32::from_rgb(120, 180, 255),
        }
    }
}

/// One LFO bank: a waveform, rate, depth, and phase offset.
/// Two independent banks (A and B) live inside `LfoParams`.
#[derive(Clone)]
pub struct LfoEngine {
    pub rate:  f32,    // cycles per second
    pub depth: f32,    // 0..1 — scaled by each param's range
    pub wave:  LfoWave,
    pub phase: f32,    // phase offset in cycles (0..1), lets B de-sync from A
}

impl LfoEngine {
    fn sample(&self, t: f32) -> f32 {
        self.wave.sample(self.rate * t + self.phase)
    }
}

#[derive(Clone)]
pub struct LfoParams {
    pub a: LfoEngine,
    pub b: LfoEngine,
    // field params — each target picks LfoSrc::{Off, A, B}
    pub kscale:         LfoSrc,
    pub speed:          LfoSrc,
    pub field_mix:      LfoSrc,
    pub iso_level:      LfoSrc,
    pub color_shift:    LfoSrc,
    pub zoom:           LfoSrc,
    pub w_lattice:      LfoSrc,
    pub w_motif:        LfoSrc,
    pub w_band:         LfoSrc,
    // feedback params
    pub fb_zoom:        LfoSrc,
    pub fb_decay:       LfoSrc,
    pub fb_offset_x:    LfoSrc,
    pub fb_offset_y:    LfoSrc,
    pub fb_rotation:    LfoSrc,
    pub fb_color_shift: LfoSrc,
    pub fb_saturation:  LfoSrc,
    pub fb_brightness:  LfoSrc,
    pub fb_inject:      LfoSrc,
    pub fb_fold_angle:  LfoSrc,
    pub fb_motion_blur: LfoSrc,
}

impl Default for LfoParams {
    fn default() -> Self {
        Self {
            a: LfoEngine { rate: 0.20, depth: 0.30, wave: LfoWave::Sine,     phase: 0.0  },
            b: LfoEngine { rate: 0.07, depth: 0.20, wave: LfoWave::Triangle, phase: 0.25 },
            kscale: LfoSrc::Off, speed: LfoSrc::Off, field_mix: LfoSrc::Off,
            iso_level: LfoSrc::Off, color_shift: LfoSrc::Off, zoom: LfoSrc::Off,
            w_lattice: LfoSrc::Off, w_motif: LfoSrc::Off, w_band: LfoSrc::Off,
            fb_zoom: LfoSrc::Off, fb_decay: LfoSrc::Off, fb_offset_x: LfoSrc::Off,
            fb_offset_y: LfoSrc::Off, fb_rotation: LfoSrc::Off,
            fb_color_shift: LfoSrc::Off, fb_saturation: LfoSrc::Off,
            fb_brightness: LfoSrc::Off, fb_inject: LfoSrc::Off,
            fb_fold_angle: LfoSrc::Off, fb_motion_blur: LfoSrc::Off,
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
    // field params
    pub kscale:         MicSrc,
    pub speed:          MicSrc,
    pub field_mix:      MicSrc,
    pub iso_level:      MicSrc,
    pub color_shift:    MicSrc,
    pub zoom:           MicSrc,
    pub w_lattice:      MicSrc,
    pub w_motif:        MicSrc,
    pub w_band:         MicSrc,
    // feedback params
    pub fb_zoom:        MicSrc,
    pub fb_decay:       MicSrc,
    pub fb_offset_x:    MicSrc,
    pub fb_offset_y:    MicSrc,
    pub fb_rotation:    MicSrc,
    pub fb_color_shift: MicSrc,
    pub fb_saturation:  MicSrc,
    pub fb_brightness:  MicSrc,
    pub fb_inject:      MicSrc,
    pub fb_fold_angle:  MicSrc,
    pub fb_motion_blur: MicSrc,
    pub depth:          f32,
}

impl Default for MicParams {
    fn default() -> Self {
        Self {
            kscale: MicSrc::Off, speed: MicSrc::Off, field_mix: MicSrc::Off,
            iso_level: MicSrc::Off, color_shift: MicSrc::Off, zoom: MicSrc::Off,
            w_lattice: MicSrc::Off, w_motif: MicSrc::Off, w_band: MicSrc::Off,
            fb_zoom: MicSrc::Off, fb_decay: MicSrc::Off, fb_offset_x: MicSrc::Off,
            fb_offset_y: MicSrc::Off, fb_rotation: MicSrc::Off, fb_color_shift: MicSrc::Off,
            fb_saturation: MicSrc::Off, fb_brightness: MicSrc::Off,
            fb_inject: MicSrc::Off, fb_fold_angle: MicSrc::Off,
            fb_motion_blur: MicSrc::Off,
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
    params:         FieldParams,
    muted:          bool,
    /// Per-step duration multiplier in [0.25, 4.0]. 1.0 = global step_dur.
    dur_mul:        f32,
    /// Per-step transition curve override (None = use Sequencer.curve).
    curve_override: Option<TranCurve>,
    /// Probability the step plays when advance lands on it (0..1, 1.0 = always).
    prob:           f32,
}

impl SeqStep {
    fn new(p: FieldParams) -> Self {
        Self { params: p, muted: false, dur_mul: 1.0, curve_override: None, prob: 1.0 }
    }
}

/// Step traversal order for the sequencer.
#[derive(Clone, Copy, PartialEq, Default)]
enum SeqPlayMode {
    #[default]
    Forward,
    Reverse,
    PingPong,
    Random,
}

impl SeqPlayMode {
    fn next(self) -> Self {
        match self {
            Self::Forward  => Self::Reverse,
            Self::Reverse  => Self::PingPong,
            Self::PingPong => Self::Random,
            Self::Random   => Self::Forward,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Forward  => "FWD",
            Self::Reverse  => "REV",
            Self::PingPong => "PP",
            Self::Random   => "RND",
        }
    }
}

fn lerp_fp(a: &FieldParams, b: &FieldParams, t: f32) -> FieldParams {
    let l = |x: f32, y: f32| x + (y - x) * t;
    let d = b.color_shift - a.color_shift;
    let cs_delta = if d > 0.5 { d - 1.0 } else if d < -0.5 { d + 1.0 } else { d };
    FieldParams {
        mode:           if t < 0.5 { a.mode } else { b.mode },
        kscale:         l(a.kscale,    b.kscale),
        speed:          l(a.speed,     b.speed),
        field_mix:      l(a.field_mix, b.field_mix),
        iso_level:      l(a.iso_level, b.iso_level),
        color_shift:    (a.color_shift + cs_delta * t).rem_euclid(1.0),
        zoom:           l(a.zoom,      b.zoom),
        w_lattice:      l(a.w_lattice, b.w_lattice),
        w_motif:        l(a.w_motif,   b.w_motif),
        w_band:         l(a.w_band,    b.w_band),
        fb_enabled:     if t < 0.5 { a.fb_enabled } else { b.fb_enabled },
        fb_mirror:      if t < 0.5 { a.fb_mirror } else { b.fb_mirror },
        fb_zoom:        l(a.fb_zoom,        b.fb_zoom),
        fb_offset_x:    l(a.fb_offset_x,    b.fb_offset_x),
        fb_offset_y:    l(a.fb_offset_y,    b.fb_offset_y),
        fb_rotation:    l(a.fb_rotation,    b.fb_rotation),
        fb_decay:       l(a.fb_decay,       b.fb_decay),
        fb_color_shift: l(a.fb_color_shift, b.fb_color_shift),
        fb_inject:      l(a.fb_inject,      b.fb_inject),
        fb_fold_angle:  l(a.fb_fold_angle,  b.fb_fold_angle),
        fb_saturation:  l(a.fb_saturation,  b.fb_saturation),
        fb_brightness:  l(a.fb_brightness,  b.fb_brightness),
        fb_blend_mode:  if t < 0.5 { a.fb_blend_mode } else { b.fb_blend_mode },
        fb_motion_blur: l(a.fb_motion_blur,  b.fb_motion_blur),
    }
}

// Compact builder used by presets
fn sp(mode: u32, ks: f32, sp: f32, fm: f32, il: f32, cs: f32, zm: f32, wl: f32, wm: f32, wb: f32) -> SeqStep {
    SeqStep::new(FieldParams {
        mode, kscale: ks, speed: sp, field_mix: fm, iso_level: il,
        color_shift: cs, zoom: zm, w_lattice: wl, w_motif: wm, w_band: wb,
        ..FieldParams::default()
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
            ..FieldParams::default()
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
    pub play_mode:  SeqPlayMode,
    pub pp_dir:     i8,       // direction for PingPong: +1 or -1 (0 = not started)
    pub rng_seed:   u32,      // LCG state for Random mode and prob gates
    from_params:    FieldParams,
}

impl Sequencer {
    fn new() -> Self {
        let steps = seq_preset_phase_space();
        let from_params = steps[0].params.clone();
        Self {
            active: false, manual: false, steps, cur: 0,
            step_dur: 3.0, step_timer: 0.0,
            curve: TranCurve::EaseInOut, selected: None,
            play_mode: SeqPlayMode::Forward,
            pp_dir: 1, rng_seed: 0x9E3779B9,
            from_params,
        }
    }

    /// Duration for the *current* step (respects per-step dur_mul).
    fn effective_step_dur(&self) -> f32 {
        let mul = self.steps.get(self.cur).map(|s| s.dur_mul).unwrap_or(1.0);
        (self.step_dur * mul).max(0.05)
    }

    /// Transition curve for the current step (per-step override else global).
    fn effective_curve(&self) -> TranCurve {
        self.steps.get(self.cur)
            .and_then(|s| s.curve_override)
            .unwrap_or(self.curve)
    }

    fn current_params(&self) -> FieldParams {
        let dur = self.effective_step_dur();
        let t = self.effective_curve().apply((self.step_timer / dur).clamp(0.0, 1.0));
        lerp_fp(&self.from_params, &self.steps[self.cur].params, t)
    }

    fn tick(&mut self, dt: f32) {
        if !self.active { return; }
        self.step_timer += dt;
        let dur = self.effective_step_dur();
        if self.manual {
            self.step_timer = self.step_timer.min(dur); // arrive but don't advance
            return;
        }
        if self.step_timer >= dur {
            self.step_timer -= dur;
            self.from_params = self.steps[self.cur].params.clone();
            self.advance_next();
        }
    }

    /// Knuth LCG; deterministic. Used for Random play mode and probability gates.
    fn rng_next(&mut self) -> u32 {
        self.rng_seed = self.rng_seed.wrapping_mul(1664525).wrapping_add(1013904223);
        self.rng_seed
    }
    fn rng_unit(&mut self) -> f32 {
        let v = self.rng_next();
        ((v >> 8) & 0x00FF_FFFF) as f32 / 16_777_216.0  // 24-bit fraction
    }

    /// Compute the next raw index using the play mode (no muted/prob filtering).
    fn raw_next_index(&mut self) -> usize {
        let n = self.steps.len();
        if n <= 1 { return 0; }
        match self.play_mode {
            SeqPlayMode::Forward => (self.cur + 1) % n,
            SeqPlayMode::Reverse => self.cur.checked_sub(1).unwrap_or(n - 1),
            SeqPlayMode::PingPong => {
                if self.pp_dir == 0 { self.pp_dir = 1; }
                let cur = self.cur as i32;
                let mut nx = cur + self.pp_dir as i32;
                if nx < 0 || nx >= n as i32 {
                    self.pp_dir = -self.pp_dir;
                    nx = cur + self.pp_dir as i32;
                    if nx < 0 || nx >= n as i32 { nx = cur; }
                }
                nx as usize
            }
            SeqPlayMode::Random => {
                let mut nx = (self.rng_next() as usize) % n;
                if nx == self.cur { nx = (nx + 1) % n; }
                nx
            }
        }
    }

    fn advance_next(&mut self) {
        let n = self.steps.len();
        if n == 0 { return; }
        // Try up to 2n candidates: skip muted and probability-failed steps.
        for _ in 0..(2 * n) {
            let cand = self.raw_next_index();
            self.cur = cand;
            if self.steps[cand].muted { continue; }
            if self.steps[cand].prob < 0.999 {
                if self.rng_unit() > self.steps[cand].prob { continue; }
            }
            return;
        }
        // All blocked (every step muted or prob=0): leave self.cur where it last landed.
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

    /// Append a captured FieldParams as a fresh step at the end. Returns the new
    /// step's index, or None if the 32-step cap was hit.
    fn capture(&mut self, params: FieldParams) -> Option<usize> {
        if self.steps.len() >= 32 { return None; }
        self.steps.push(SeqStep::new(params));
        Some(self.steps.len() - 1)
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
    let fb_on = r(11.0) > 0.55; // ~45% chance of feedback
    let mirror_roll = r(12.0);
    let fb_mirror = if mirror_roll < 0.25 { 0u32 }
        else if mirror_roll < 0.45 { 1 }
        else if mirror_roll < 0.60 { 3 }
        else if mirror_roll < 0.72 { 5 }
        else if mirror_roll < 0.82 { 6 }
        else if mirror_roll < 0.90 { 7 }
        else { 9 };
    FieldParams {
        mode:           ((r(1.0) * MODE_NAMES.len() as f32) as u32).min(MODE_NAMES.len() as u32 - 1),
        kscale:         0.3 + r(2.0) * 3.5,
        speed:          r(3.0) * 1.6,
        field_mix:      r(4.0),
        iso_level:      0.05 + r(5.0) * 0.9,
        color_shift:    r(6.0),
        zoom:           0.4 + r(7.0) * 1.9,
        w_lattice:      r(8.0) * 2.0,
        w_motif:        r(9.0) * 2.0,
        w_band:         r(10.0) * 2.0,
        fb_enabled:     fb_on,
        fb_mirror,
        fb_zoom:        0.93 + r(13.0) * 0.09,
        fb_decay:       0.60 + r(14.0) * 0.38,
        fb_color_shift: (r(15.0) - 0.5) * 0.6,
        fb_inject:      0.5 + r(16.0) * 0.5,
        fb_saturation:  0.7 + r(17.0) * 0.6,
        fb_brightness:  0.8 + r(18.0) * 0.4,
        fb_rotation:    (r(19.0) - 0.5) * 0.06,
        fb_offset_x:    (r(20.0) - 0.5) * 0.04,
        fb_offset_y:    (r(21.0) - 0.5) * 0.04,
        fb_fold_angle:  (r(22.0) - 0.5) * 2.0,
        fb_blend_mode:  (r(23.0) * 4.0) as u32, // modes 0-3 most useful
        // Fraksl MotionBlur: ~40% chance of light trails, else off
        fb_motion_blur: if r(24.0) > 0.6 { r(25.0) * 0.6 } else { 0.0 },
        ..FieldParams::default()
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
        a: LfoEngine { rate: 0.18, depth: 0.18, wave: LfoWave::Triangle, phase: 0.0  },
        b: LfoEngine { rate: 0.07, depth: 0.12, wave: LfoWave::Sine,     phase: 0.33 },
        kscale: LfoSrc::A, speed: LfoSrc::A, field_mix: LfoSrc::A, iso_level: LfoSrc::A,
        color_shift: LfoSrc::A, zoom: LfoSrc::A,
        w_lattice: LfoSrc::A, w_motif: LfoSrc::A, w_band: LfoSrc::A,
        fb_zoom: LfoSrc::B, fb_decay: LfoSrc::Off, fb_color_shift: LfoSrc::B,
        fb_saturation: LfoSrc::A, fb_brightness: LfoSrc::Off,
        fb_rotation: LfoSrc::B, fb_offset_x: LfoSrc::Off, fb_offset_y: LfoSrc::Off,
        fb_inject: LfoSrc::Off, fb_fold_angle: LfoSrc::B,
        fb_motion_blur: LfoSrc::Off,
    }
}

fn tour_lfo_preset_fb_heavy() -> LfoParams {
    LfoParams {
        a: LfoEngine { rate: 0.07, depth: 0.25, wave: LfoWave::Sine,     phase: 0.0  },
        b: LfoEngine { rate: 0.21, depth: 0.18, wave: LfoWave::Triangle, phase: 0.5  },
        kscale: LfoSrc::Off, speed: LfoSrc::A, field_mix: LfoSrc::Off, iso_level: LfoSrc::Off,
        color_shift: LfoSrc::A, zoom: LfoSrc::Off,
        w_lattice: LfoSrc::Off, w_motif: LfoSrc::Off, w_band: LfoSrc::Off,
        fb_zoom: LfoSrc::A, fb_decay: LfoSrc::B, fb_color_shift: LfoSrc::A,
        fb_saturation: LfoSrc::A, fb_brightness: LfoSrc::B,
        fb_rotation: LfoSrc::A, fb_offset_x: LfoSrc::B, fb_offset_y: LfoSrc::B,
        fb_inject: LfoSrc::Off, fb_fold_angle: LfoSrc::A,
        fb_motion_blur: LfoSrc::B,
    }
}

fn tour_rand(seed: f32) -> f32 {
    bz_hash(seed * 71.17 + 4.91)
}

/// Auto-pilot for the feedback layer: overrides every fb_* field of `base` with an
/// evolving Fraksl-style mix of scene-randomized picks (mirror/blend) and continuous
/// multi-frequency sin/cos drift. Non-feedback params come through unchanged so this
/// composes with the manual/tour/sequencer source of `base`.
fn fb_auto_params(t: f32, base: &FieldParams) -> FieldParams {
    const SCENE_LEN: f32 = 7.3;
    let scene_f = (t / SCENE_LEN).floor();
    let local   = (t / SCENE_LEN).fract();
    let drift   = t * 0.13;
    let punch   = (TAU * local).sin().max(0.0).powf(1.6);
    let snap    = if local < 0.18 { (1.0 - local / 0.18).powf(2.0) } else { 0.0 };

    // Per-scene discrete picks
    let mirrors: [u32; 9] = [0, 1, 3, 5, 6, 7, 8, 9, 11];
    let blends:  [u32; 7] = [0, 1, 2, 3, 5, 6, 9];
    let mi = (tour_rand(scene_f + 11.0) * mirrors.len() as f32) as usize % mirrors.len();
    let bi = (tour_rand(scene_f + 23.0) * blends.len()  as f32) as usize % blends.len();

    // Continuous, energy-bounded variation. Keeps things "trippy" but stable.
    let fb_zoom        = (0.965 + 0.045 * (TAU * (local * 0.35 + drift * 0.27)).sin()
                                + 0.020 * snap).clamp(0.93, 1.05);
    let fb_decay       = (0.82 + 0.14 * (TAU * (local * 0.55 + drift * 0.19)).cos()).clamp(0.50, 0.98);
    let fb_inject      = (0.65 + 0.35 * punch).clamp(0.0, 1.0);
    let fb_color_shift = 0.55 * (TAU * (drift * 0.17 + local * 0.41)).sin();
    let fb_saturation  = (1.05 + 0.45 * (TAU * (local * 0.7 + drift * 0.11)).sin()).clamp(0.0, 2.0);
    let fb_brightness  = (1.0  + 0.18 * (TAU * (local * 0.4 + drift * 0.09)).cos()).clamp(0.5, 1.5);
    let fb_rotation    = 0.06  * (TAU * (drift * 0.09 + local * 0.27)).sin();
    let fb_offset_x    = 0.035 * (TAU * (drift * 0.14 + local * 0.42)).sin();
    let fb_offset_y    = 0.035 * (TAU * (drift * 0.11 + local * 0.36)).cos();
    let fb_fold_angle  = 2.4   * (TAU * (drift * 0.06 + local * 0.22)).sin()
                              + (tour_rand(scene_f + 37.0) - 0.5) * 1.6;
    // Motion blur on for ~60% of scenes, lightly modulated.
    let mb_on   = tour_rand(scene_f + 53.0) > 0.40;
    let fb_motion_blur = if mb_on {
        (0.18 + 0.22 * (TAU * (local * 0.30 + drift * 0.05)).sin().abs()).clamp(0.0, 0.65)
    } else { 0.0 };

    FieldParams {
        fb_enabled:     true,
        fb_mirror:      mirrors[mi],
        fb_blend_mode:  blends[bi],
        fb_zoom, fb_decay, fb_inject, fb_color_shift,
        fb_saturation, fb_brightness, fb_rotation,
        fb_offset_x, fb_offset_y, fb_fold_angle, fb_motion_blur,
        ..base.clone()
    }
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

    let fb_on = tour_rand(seed + 77.0) > 0.45;
    let mirror_roll = tour_rand(seed + 88.0);
    let fb_mirror = if mirror_roll < 0.3 { 0u32 }
        else if mirror_roll < 0.5 { 1 }
        else if mirror_roll < 0.65 { 5 }
        else if mirror_roll < 0.78 { 6 }
        else if mirror_roll < 0.88 { 7 }
        else { 9 };
    FieldParams {
        mode: mode.min((MODE_NAMES.len() - 1) as u32),
        kscale:      (morph(3.0, 0.25, 3.8) + burst * 0.55).clamp(0.1, 5.0),
        speed:       (morph(4.0, 0.05, 1.75) + burst * 0.25).clamp(0.0, 2.0),
        field_mix:   morph(5.0, 0.0, 1.0).clamp(0.0, 1.0),
        iso_level:   morph(6.0, 0.05, 0.96).clamp(0.0, 1.0),
        color_shift: (morph(7.0, 0.0, 1.0) + t * 0.025).rem_euclid(1.0),
        zoom:        (morph(8.0, 0.35, 2.15) + burst * 0.25).clamp(0.2, 5.0),
        w_lattice:   morph(9.0, 0.0, 2.0).clamp(0.0, 2.0),
        w_motif:     morph(10.0, 0.0, 2.0).clamp(0.0, 2.0),
        w_band:      morph(11.0, 0.0, 2.0).clamp(0.0, 2.0),
        fb_enabled:  fb_on,
        fb_mirror,
        fb_zoom:        morph(12.0, 0.93, 1.02),
        fb_decay:       morph(13.0, 0.65, 0.97),
        fb_color_shift: morph(14.0, -0.3, 0.3),
        fb_inject:      morph(15.0, 0.5, 1.0),
        fb_saturation:  morph(16.0, 0.7, 1.4),
        fb_brightness:  morph(17.0, 0.85, 1.15),
        fb_rotation:    morph(18.0, -0.03, 0.03),
        fb_offset_x:    morph(19.0, -0.02, 0.02),
        fb_offset_y:    morph(20.0, -0.02, 0.02),
        fb_fold_angle:  morph(21.0, -1.57, 1.57),
        fb_blend_mode:  (tour_rand(seed + 99.0) * 4.0) as u32,
        fb_motion_blur: if tour_rand(seed + 101.0) > 0.55 { morph(22.0, 0.0, 0.55) } else { 0.0 },
        ..FieldParams::default()
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

    let fb_on = (scene % 3) != 0; // feedback in 2/3 of curated scenes
    let fb_mirrors = [0u32, 1, 3, 5, 6, 7, 8, 9];
    let fb_mirror = fb_mirrors[scene % fb_mirrors.len()];
    FieldParams {
        mode: TOUR_MODES[(scene + crystal_idx) % TOUR_MODES.len()],
        kscale:      (1.05 + 0.72 * (TAU * drift).sin().abs() + 0.45 * snap).clamp(0.1, 5.0),
        speed:       (0.22 + 0.82 * punch + 0.18 * (TAU * (drift * 0.37)).sin().abs()).clamp(0.0, 2.0),
        field_mix:   (0.50 + 0.38 * (TAU * (local + drift * 0.11)).sin()).clamp(0.0, 1.0),
        iso_level:   (0.42 + 0.36 * (TAU * (local * 0.5 + drift * 0.19)).cos()).clamp(0.0, 1.0),
        color_shift: (drift * 0.22 + 0.08 * (TAU * local).sin()).rem_euclid(1.0),
        zoom:        (0.78 + 0.36 * (TAU * (local * 0.75)).sin().abs() + 0.22 * snap).clamp(0.2, 5.0),
        w_lattice:   (0.75 + 0.65 * (TAU * (local + 0.10)).sin().abs()).clamp(0.0, 2.0),
        w_motif:     (0.38 + 0.92 * (TAU * (local * 0.7 + 0.35)).sin().abs()).clamp(0.0, 2.0),
        w_band:      (0.48 + 1.05 * (TAU * (local * 1.2 + drift * 0.07)).cos().abs()).clamp(0.0, 2.0),
        fb_enabled:  fb_on,
        fb_mirror,
        fb_zoom:        (0.96 + 0.04 * (TAU * (local * 0.3 + drift * 0.07)).sin()).clamp(0.90, 1.10),
        fb_decay:       (0.82 + 0.12 * (TAU * (local * 0.5)).cos()).clamp(0.30, 0.99),
        fb_color_shift: 0.08 * (TAU * (drift * 0.11 + local * 0.3)).sin(),
        fb_inject:      (0.7 + 0.3 * punch).clamp(0.0, 1.0),
        fb_saturation:  (1.0 + 0.3 * (TAU * (local * 0.7 + drift * 0.13)).sin()).clamp(0.0, 2.0),
        fb_brightness:  (1.0 + 0.15 * (TAU * local).cos()).clamp(0.0, 2.0),
        fb_rotation:    0.015 * (TAU * (drift * 0.07 + local * 0.25)).sin(),
        fb_offset_x:    0.012 * (TAU * (drift * 0.13 + local * 0.4)).sin(),
        fb_offset_y:    0.012 * (TAU * (drift * 0.09 + local * 0.35)).cos(),
        fb_fold_angle:  1.2 * (TAU * (drift * 0.05 + local * 0.2)).sin(),
        fb_blend_mode:  (scene % 4) as u32,
        // Curated tour: motion blur cycles in 1/4 of scenes (smooth tail feel)
        fb_motion_blur: if (scene % 4) == 2 {
            (0.20 + 0.18 * (TAU * (local * 0.4)).sin()).clamp(0.0, 0.6)
        } else { 0.0 },
        ..FieldParams::default()
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
    let lfo_a_s = lfo.a.sample(t);
    let lfo_b_s = lfo.b.sample(t);
    macro_rules! modulate {
        ($val:expr, $lfo_src:expr, $mic_src:expr, $min:expr, $max:expr) => {{
            let range   = ($max as f32) - ($min as f32);
            let lfo_d   = match $lfo_src {
                LfoSrc::Off => 0.0,
                LfoSrc::A   => lfo.a.depth * range * lfo_a_s,
                LfoSrc::B   => lfo.b.depth * range * lfo_b_s,
            };
            let mic_d   = $mic_src.value(bands) * mic.depth * range;
            ($val + lfo_d + mic_d).clamp($min as f32, $max as f32)
        }};
    }
    FieldParams {
        mode:           fp.mode,
        kscale:         modulate!(fp.kscale,      lfo.kscale,      mic.kscale,      0.1, 5.0),
        speed:          modulate!(fp.speed,       lfo.speed,       mic.speed,       0.0, 2.0),
        field_mix:      modulate!(fp.field_mix,   lfo.field_mix,   mic.field_mix,   0.0, 1.0),
        iso_level:      modulate!(fp.iso_level,   lfo.iso_level,   mic.iso_level,   0.0, 1.0),
        color_shift:    modulate!(fp.color_shift, lfo.color_shift, mic.color_shift, 0.0, 1.0),
        zoom:           modulate!(fp.zoom,        lfo.zoom,        mic.zoom,        0.2, 5.0),
        w_lattice:      modulate!(fp.w_lattice,   lfo.w_lattice,   mic.w_lattice,   0.0, 2.0),
        w_motif:        modulate!(fp.w_motif,     lfo.w_motif,     mic.w_motif,     0.0, 2.0),
        w_band:         modulate!(fp.w_band,      lfo.w_band,      mic.w_band,      0.0, 2.0),
        // feedback params — modulated when fb_enabled
        fb_enabled:     fp.fb_enabled,
        fb_mirror:      fp.fb_mirror,
        fb_blend_mode:  fp.fb_blend_mode,
        fb_zoom:        modulate!(fp.fb_zoom,        lfo.fb_zoom,        mic.fb_zoom,        0.90, 1.10),
        fb_decay:       modulate!(fp.fb_decay,       lfo.fb_decay,       mic.fb_decay,       0.30, 0.99),
        fb_offset_x:    modulate!(fp.fb_offset_x,    lfo.fb_offset_x,    mic.fb_offset_x,   -0.10, 0.10),
        fb_offset_y:    modulate!(fp.fb_offset_y,    lfo.fb_offset_y,    mic.fb_offset_y,   -0.10, 0.10),
        fb_rotation:    modulate!(fp.fb_rotation,    lfo.fb_rotation,    mic.fb_rotation,   -0.30, 0.30),
        fb_color_shift: modulate!(fp.fb_color_shift, lfo.fb_color_shift, mic.fb_color_shift,-1.00, 1.00),
        fb_saturation:  modulate!(fp.fb_saturation,  lfo.fb_saturation,  mic.fb_saturation,  0.00, 2.00),
        fb_brightness:  modulate!(fp.fb_brightness,  lfo.fb_brightness,  mic.fb_brightness,  0.00, 2.00),
        fb_inject:      modulate!(fp.fb_inject,      lfo.fb_inject,      mic.fb_inject,      0.00, 1.00),
        fb_fold_angle:  modulate!(fp.fb_fold_angle,  lfo.fb_fold_angle,  mic.fb_fold_angle, -3.14, 3.14),
        fb_motion_blur: modulate!(fp.fb_motion_blur, lfo.fb_motion_blur, mic.fb_motion_blur, 0.00, 0.95),
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
    fb_reset:          bool,
    fb_auto_toggle:    bool,
    keymap_toggle:     bool,
    seq_play_mode_toggle: bool,
    seq_capture:          bool,
    seq_step_dur_mul:     Option<(usize, f32)>,
    seq_step_curve_cycle: Option<usize>,
    seq_step_prob:        Option<(usize, f32)>,
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
    midi_in:      Option<midi::MidiCapture>,
    fb_auto:      bool,
    show_keymap:  bool,
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
            midi_in: midi::MidiCapture::start(),
            fb_auto: false,
            show_keymap: false,
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
                    PhysicalKey::Code(KeyCode::KeyA) => {
                        self.fb_auto = !self.fb_auto;
                        if self.fb_auto {
                            self.field_params.fb_enabled = true;
                            gpu.fb_clear = true;
                        }
                    }
                    PhysicalKey::Code(KeyCode::Slash) => {
                        // `?` (Shift+/) or plain `/` — toggle the keymap overlay
                        self.show_keymap = !self.show_keymap;
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
                let cur_fb_auto       = self.fb_auto;
                let cur_show_keymap   = self.show_keymap;
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
                let cur_seq_play_mode = self.sequencer.play_mode;
                let cur_seq_eff_dur   = self.sequencer.effective_step_dur();
                let cur_seq_progress  = (self.sequencer.step_timer / cur_seq_eff_dur).clamp(0.0, 1.0);
                // Editor: mutable local that closure can modify; written back in mutations
                let mut seq_selected_edit: Option<(usize, FieldParams)> = self.sequencer.selected
                    .and_then(|i| self.sequencer.steps.get(i).map(|s| (i, s.params.clone())));
                // Per-step extras for the selected step editor: (dur_mul, curve_override, prob)
                let seq_selected_extras: Option<(f32, Option<TranCurve>, f32)> = self.sequencer.selected
                    .and_then(|i| self.sequencer.steps.get(i))
                    .map(|s| (s.dur_mul, s.curve_override, s.prob));
                // Snapshot field_params, lfo, mic, and current audio bands.
                let mut fp  = seq_fp.or(tour_fp).unwrap_or_else(|| self.field_params.clone());
                // FB AUTO: override every fb_* field with an evolving auto-pilot pattern.
                // Composes on top of the manual/tour/sequencer source, before LFO/mic.
                if self.fb_auto { fp = fb_auto_params(t, &fp); }

                // MIDI snapshot: pull CC values and drain triggered notes.
                let mut midi_notes: Vec<u8> = Vec::new();
                let mut midi_port_name: Option<String> = None;
                let mut midi_last_cc: Option<(u8, u8)> = None;
                let midi_cc_snap = if let Some(m) = self.midi_in.as_ref() {
                    if let Ok(mut s) = m.state.lock() {
                        midi_notes.append(&mut s.notes);
                        midi_port_name = s.port_name.clone();
                        midi_last_cc = s.last_cc;
                        midi::CcSnapshot::from_state(&s)
                    } else { midi::CcSnapshot::default() }
                } else { midi::CcSnapshot::default() };
                // CC: directly drive params (overrides tour/seq for the set CCs).
                midi::apply_midi_cc(&mut fp, &midi_cc_snap);

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
                    time:           t,
                    kscale:         fp_eff.kscale,
                    speed:          fp_eff.speed,
                    field_mix:      fp_eff.field_mix,
                    iso_level:      fp_eff.iso_level,
                    color_shift:    fp_eff.color_shift,
                    zoom:           fp_eff.zoom,
                    w_lattice:      fp_eff.w_lattice,
                    w_motif:        fp_eff.w_motif,
                    w_band:         fp_eff.w_band,
                    mode:           fp_eff.mode,
                    num_g:          gpu.gpu_field.count as u32,
                    crystal_color:  [cdef.color[0], cdef.color[1], cdef.color[2], 0.0],
                    mouse:          self.mouse_norm,
                    mouse_down:     if self.mouse_btn_down { 1.0 } else { 0.0 },
                    aspect:         gpu.size.width as f32 / gpu.size.height.max(1) as f32,
                    fb_enabled:     if fp_eff.fb_enabled { 1 } else { 0 },
                    fb_mirror:      fp_eff.fb_mirror,
                    fb_zoom:        fp_eff.fb_zoom,
                    fb_offset_x:    fp_eff.fb_offset_x,
                    fb_offset_y:    fp_eff.fb_offset_y,
                    fb_rotation:    fp_eff.fb_rotation,
                    fb_decay:       fp_eff.fb_decay,
                    fb_color_shift: fp_eff.fb_color_shift,
                    fb_inject:      fp_eff.fb_inject,
                    fb_fold_angle:  fp_eff.fb_fold_angle,
                    fb_saturation:  fp_eff.fb_saturation,
                    fb_brightness:  fp_eff.fb_brightness,
                    fb_blend_mode:  fp_eff.fb_blend_mode,
                    fb_motion_blur: fp_eff.fb_motion_blur,
                    _pad:           [0.0; 2],
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
                    // ── Top toolbar — pinned transport controls ───────
                    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
                        ui.add_space(2.0);
                        ui.horizontal(|ui| {
                            // Panel toggle (◀/▶)
                            if ui.small_button(if req_panel_open { "◀" } else { "▶" }).clicked() {
                                req.panel_toggle = true;
                            }
                            ui.separator();
                            // Atoms / Field render mode
                            if ui.selectable_label(cur_render_mode == RenderMode::Atoms, "ATOMS").clicked() {
                                req.render_mode = Some(RenderMode::Atoms);
                            }
                            if ui.selectable_label(cur_render_mode == RenderMode::Field, "FIELD").clicked() {
                                req.render_mode = Some(RenderMode::Field);
                            }
                            ui.separator();
                            // Tour / sequencer transport
                            let tour_lbl = if cur_tour_active { "■ TOUR" } else { "▶ TOUR" };
                            let tour_col = if cur_tour_active {
                                egui::Color32::from_rgb(140, 255, 140)
                            } else { egui::Color32::from_gray(220) };
                            if ui.add(egui::Button::new(
                                egui::RichText::new(tour_lbl).color(tour_col)
                            )).clicked() { req.tour_toggle = true; }
                            if ui.small_button(cur_tour_style.label()).clicked() {
                                req.tour_style_toggle = true;
                            }
                            let seq_lbl = if cur_seq_active { "■ SEQ" } else { "▶ SEQ" };
                            let seq_col = if cur_seq_active {
                                egui::Color32::from_rgb(180, 140, 255)
                            } else { egui::Color32::from_gray(220) };
                            if ui.add(egui::Button::new(
                                egui::RichText::new(seq_lbl).color(seq_col)
                            )).clicked() { req.seq_toggle = true; }
                            ui.separator();
                            // K-path walk
                            let kp_lbl = if cur_kpath_active { "■ K" } else { "▶ K" };
                            if ui.small_button(kp_lbl).clicked() { req.kpath_toggle = true; }
                            ui.separator();
                            // Feedback auto-pilot
                            let auto_col = if cur_fb_auto {
                                egui::Color32::from_rgb(255, 180, 80)
                            } else { egui::Color32::from_gray(180) };
                            if ui.add(egui::Button::new(
                                egui::RichText::new(if cur_fb_auto { "■ FB AUTO" } else { "□ FB AUTO" })
                                    .color(auto_col)
                            )).clicked() { req.fb_auto_toggle = true; }
                            ui.separator();
                            // Right-aligned: screenshot + help
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let help_col = if cur_show_keymap {
                                    egui::Color32::from_rgb(160, 200, 255)
                                } else { egui::Color32::from_gray(180) };
                                if ui.add(egui::Button::new(
                                    egui::RichText::new("?").color(help_col).monospace()
                                )).on_hover_text("Keyboard shortcuts (?, /)").clicked() {
                                    req.keymap_toggle = true;
                                }
                                if ui.small_button("📷").on_hover_text("Screenshot").clicked() {
                                    req.screenshot = true;
                                }
                            });
                        });
                        ui.add_space(2.0);
                    });

                    // Side panel
                    if req_panel_open {
                        egui::SidePanel::left("ctrl")
                            .min_width(260.0).max_width(300.0)
                            .resizable(false)
                            .show(ctx, |ui| {
                                // Compact header — crystal name + system on one line
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("◇")
                                        .heading().color(egui::Color32::from_rgb(160, 170, 220)));
                                    ui.vertical(|ui| {
                                        ui.label(egui::RichText::new(cur_crystal_name)
                                            .strong().size(13.0));
                                        ui.label(egui::RichText::new(cur_sys_name)
                                            .small().color(egui::Color32::from_gray(150)));
                                    });
                                });
                                ui.separator();

                                // Shared slider macro for both Field and Feedback sections.
                                // Defined here in the side-panel scope so sibling CollapsingHeader
                                // closures (Field, Feedback) can both reach it.
                                macro_rules! sld {
                                    ($ui:expr, $label:literal, $val:expr, $eff:expr,
                                     $lfo_src:expr, $mic_src:expr, $min:expr, $max:expr) => {
                                        $ui.horizontal(|ui| {
                                            // [~/A/B] LFO source — cycles Off→A→B→Off
                                            let lc = (*$lfo_src).color();
                                            let ll = (*$lfo_src).label();
                                            if ui.add(egui::Button::new(
                                                egui::RichText::new(ll).color(lc)
                                            ).min_size(egui::vec2(18.0, 18.0))).clicked() {
                                                *$lfo_src = (*$lfo_src).next();
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

                                // Crystal library (collapsible)
                                egui::CollapsingHeader::new(egui::RichText::new("🔍 Crystal library")
                                    .color(egui::Color32::from_rgb(150, 160, 200)))
                                    .id_source("sec_library").default_open(false)
                                    .show(ui, |ui| {
                                        ui.text_edit_singleline(&mut search);
                                        let q = search.to_ascii_lowercase();
                                        egui::ScrollArea::vertical()
                                            .max_height(220.0)
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
                                    });

                                // Field params (collapsible, open by default)
                                egui::CollapsingHeader::new(egui::RichText::new("🎨 Field")
                                    .color(egui::Color32::from_rgb(180, 200, 200)))
                                    .id_source("sec_field").default_open(true)
                                    .show(ui, |ui| {
                                        // Compact mode picker: ComboBox + arrow buttons
                                        ui.horizontal(|ui| {
                                            ui.label(egui::RichText::new("mode").small().monospace());
                                            egui::ComboBox::from_id_source("mode_combo")
                                                .selected_text(MODE_NAMES[fp.mode as usize])
                                                .width(160.0)
                                                .show_ui(ui, |ui| {
                                                    for (i, name) in MODE_NAMES.iter().enumerate() {
                                                        ui.selectable_value(&mut fp.mode, i as u32, *name);
                                                    }
                                                });
                                            if ui.small_button("◀").clicked() {
                                                let n = MODE_NAMES.len() as u32;
                                                fp.mode = (fp.mode + n - 1) % n;
                                            }
                                            if ui.small_button("▶").clicked() {
                                                fp.mode = (fp.mode + 1) % MODE_NAMES.len() as u32;
                                            }
                                        });
                                        ui.add_space(2.0);
                                sld!(ui, "kscale   ", &mut fp.kscale,      fp_eff.kscale,      &mut lfo.kscale,      &mut mic.kscale,      0.1_f32, 5.0_f32);
                                sld!(ui, "speed    ", &mut fp.speed,       fp_eff.speed,       &mut lfo.speed,       &mut mic.speed,       0.0_f32, 2.0_f32);
                                sld!(ui, "field_mix", &mut fp.field_mix,   fp_eff.field_mix,   &mut lfo.field_mix,   &mut mic.field_mix,   0.0_f32, 1.0_f32);
                                sld!(ui, "iso_level", &mut fp.iso_level,   fp_eff.iso_level,   &mut lfo.iso_level,   &mut mic.iso_level,   0.0_f32, 1.0_f32);
                                sld!(ui, "color_sft", &mut fp.color_shift, fp_eff.color_shift, &mut lfo.color_shift, &mut mic.color_shift, 0.0_f32, 1.0_f32);
                                sld!(ui, "zoom     ", &mut fp.zoom,        fp_eff.zoom,        &mut lfo.zoom,        &mut mic.zoom,        0.2_f32, 5.0_f32);
                                sld!(ui, "w_lattice", &mut fp.w_lattice,   fp_eff.w_lattice,   &mut lfo.w_lattice,   &mut mic.w_lattice,   0.0_f32, 2.0_f32);
                                sld!(ui, "w_motif  ", &mut fp.w_motif,     fp_eff.w_motif,     &mut lfo.w_motif,     &mut mic.w_motif,     0.0_f32, 2.0_f32);
                                sld!(ui, "w_band   ", &mut fp.w_band,      fp_eff.w_band,      &mut lfo.w_band,      &mut mic.w_band,      0.0_f32, 2.0_f32);
                                    }); // end Field collapsible

                                // ── FEEDBACK / SELF-SIMILARITY ────────────────────────────
                                egui::CollapsingHeader::new(egui::RichText::new("🔁 Feedback")
                                    .color(egui::Color32::from_rgb(220, 140, 80)))
                                    .id_source("sec_feedback").default_open(true)
                                    .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    let fb_label = if fp.fb_enabled { "■ ON" } else { "□ OFF" };
                                    if ui.small_button(fb_label).clicked() {
                                        fp.fb_enabled = !fp.fb_enabled;
                                        if fp.fb_enabled { req.fb_reset = true; }
                                    }
                                    if ui.small_button("RESET").clicked() { req.fb_reset = true; }
                                    let auto_lbl = if cur_fb_auto { "■ AUTO" } else { "□ AUTO" };
                                    let auto_btn = egui::Button::new(
                                        egui::RichText::new(auto_lbl).color(
                                            if cur_fb_auto { egui::Color32::from_rgb(255, 180, 80) }
                                            else { egui::Color32::from_gray(180) }
                                        )
                                    );
                                    if ui.add(auto_btn).clicked() { req.fb_auto_toggle = true; }
                                    ui.label(egui::RichText::new("mirror").small());
                                    let mirror_labels = ["none","H","V","HV","quad","3fold","6fold","8fold","tri","pin","4fold","12fold"];
                                    if ui.small_button(mirror_labels[fp.fb_mirror.min(11) as usize]).clicked() {
                                        fp.fb_mirror = (fp.fb_mirror + 1) % 12;
                                    }
                                    let blend_labels = ["lerp","add","scrn","mul","over","diff","lite","burn","dodge","excl","hrd","sft"];
                                    if ui.small_button(blend_labels[fp.fb_blend_mode.min(11) as usize]).clicked() {
                                        fp.fb_blend_mode = (fp.fb_blend_mode + 1) % 12;
                                    }
                                });
                                sld!(ui, "fb_zoom  ", &mut fp.fb_zoom,        fp_eff.fb_zoom,        &mut lfo.fb_zoom,        &mut mic.fb_zoom,        0.90_f32, 1.10_f32);
                                sld!(ui, "fb_decay ", &mut fp.fb_decay,       fp_eff.fb_decay,       &mut lfo.fb_decay,       &mut mic.fb_decay,       0.30_f32, 0.99_f32);
                                sld!(ui, "fb_inject", &mut fp.fb_inject,      fp_eff.fb_inject,      &mut lfo.fb_inject,      &mut mic.fb_inject,      0.00_f32, 1.00_f32);
                                sld!(ui, "fb_sat   ", &mut fp.fb_saturation,  fp_eff.fb_saturation,  &mut lfo.fb_saturation,  &mut mic.fb_saturation,  0.00_f32, 2.00_f32);
                                sld!(ui, "fb_bright", &mut fp.fb_brightness,  fp_eff.fb_brightness,  &mut lfo.fb_brightness,  &mut mic.fb_brightness,  0.00_f32, 2.00_f32);
                                sld!(ui, "fb_rotate", &mut fp.fb_rotation,    fp_eff.fb_rotation,    &mut lfo.fb_rotation,    &mut mic.fb_rotation,   -0.30_f32, 0.30_f32);
                                sld!(ui, "fb_hue   ", &mut fp.fb_color_shift, fp_eff.fb_color_shift, &mut lfo.fb_color_shift, &mut mic.fb_color_shift,-1.00_f32, 1.00_f32);
                                sld!(ui, "fb_pan_x ", &mut fp.fb_offset_x,    fp_eff.fb_offset_x,    &mut lfo.fb_offset_x,    &mut mic.fb_offset_x,   -0.10_f32, 0.10_f32);
                                sld!(ui, "fb_pan_y ", &mut fp.fb_offset_y,    fp_eff.fb_offset_y,    &mut lfo.fb_offset_y,    &mut mic.fb_offset_y,   -0.10_f32, 0.10_f32);
                                sld!(ui, "fb_motion", &mut fp.fb_motion_blur, fp_eff.fb_motion_blur, &mut lfo.fb_motion_blur, &mut mic.fb_motion_blur,  0.00_f32, 0.95_f32);
                                if fp.fb_mirror >= 5 {
                                    sld!(ui, "fold_ang ", &mut fp.fb_fold_angle, fp_eff.fb_fold_angle, &mut lfo.fb_fold_angle, &mut mic.fb_fold_angle, -3.14_f32, 3.14_f32);
                                }
                                    }); // end Feedback collapsible

                                // ── MODULATION (LFO A/B side-by-side + mic) ───────────────
                                egui::CollapsingHeader::new(egui::RichText::new("🌊 Modulation")
                                    .color(egui::Color32::from_rgb(150, 200, 200)))
                                    .id_source("sec_modulation").default_open(false)
                                    .show(ui, |ui| {
                                        // Compact two-column LFO row.
                                        // Each column shows: wave button, rate / depth / phase sliders.
                                        let lfo_col = |ui: &mut egui::Ui, name: &str, color: egui::Color32, eng: &mut LfoEngine| {
                                            ui.vertical(|ui| {
                                                ui.set_min_width(130.0);
                                                ui.horizontal(|ui| {
                                                    ui.label(egui::RichText::new(name).strong().color(color));
                                                    if ui.small_button(eng.wave.label()).clicked() {
                                                        eng.wave = eng.wave.next();
                                                    }
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.label(egui::RichText::new("r").small().monospace());
                                                    ui.add(egui::Slider::new(&mut eng.rate, 0.01..=4.0)
                                                        .show_value(true).suffix(" Hz"));
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.label(egui::RichText::new("d").small().monospace());
                                                    ui.add(egui::Slider::new(&mut eng.depth, 0.0..=1.0).show_value(false));
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.label(egui::RichText::new("φ").small().monospace());
                                                    ui.add(egui::Slider::new(&mut eng.phase, 0.0..=1.0).show_value(false));
                                                });
                                            });
                                        };
                                        ui.horizontal(|ui| {
                                            lfo_col(ui, "LFO A", egui::Color32::from_rgb(80, 220, 120), &mut lfo.a);
                                            ui.separator();
                                            lfo_col(ui, "LFO B", egui::Color32::from_rgb(120, 180, 255), &mut lfo.b);
                                        });

                                        ui.add_space(4.0);

                                        // Mic: one-row band meters + depth slider when live
                                        let mic_col = if mic_active {
                                            egui::Color32::from_rgb(255, 140, 80)
                                        } else { egui::Color32::from_gray(100) };
                                        ui.horizontal(|ui| {
                                            ui.label(egui::RichText::new(if mic_active { "🎤 MIC" } else { "🎤 — no device" })
                                                .small().color(mic_col));
                                            if mic_active {
                                                ui.add(egui::Slider::new(&mut mic.depth, 0.0..=2.0)
                                                    .show_value(true).text("depth"));
                                            }
                                        });
                                        if mic_active {
                                            ui.horizontal(|ui| {
                                                let mut band = |ui: &mut egui::Ui, lbl: &str, c: egui::Color32, v: f32| {
                                                    ui.label(egui::RichText::new(lbl).small().color(c).monospace());
                                                    ui.add(egui::ProgressBar::new(v).desired_width(28.0));
                                                };
                                                band(ui, "A", egui::Color32::from_rgb(220,220,220), cur_bands.amplitude);
                                                band(ui, "B", egui::Color32::from_rgb(255,80,80),   cur_bands.bass);
                                                band(ui, "M", egui::Color32::from_rgb(80,220,120),  cur_bands.mid);
                                                band(ui, "T", egui::Color32::from_rgb(80,160,255),  cur_bands.treble);
                                            });
                                            ui.label(egui::RichText::new("Tip: click · on a slider to route a band.")
                                                .small().color(egui::Color32::from_gray(130)));
                                        }

                                        // ── MIDI status row
                                        ui.add_space(4.0);
                                        let midi_col = if midi_port_name.is_some() {
                                            egui::Color32::from_rgb(180, 220, 255)
                                        } else { egui::Color32::from_gray(100) };
                                        ui.horizontal(|ui| {
                                            ui.label(egui::RichText::new(
                                                if let Some(ref p) = midi_port_name {
                                                    format!("🎹 MIDI: {}", p)
                                                } else {
                                                    "🎹 MIDI — no port".to_string()
                                                }
                                            ).small().color(midi_col));
                                            if let Some((cc, v)) = midi_last_cc {
                                                ui.label(egui::RichText::new(format!("CC{cc}={v}"))
                                                    .small().monospace()
                                                    .color(egui::Color32::from_gray(150)));
                                            }
                                        });
                                        if midi_port_name.is_some() {
                                            ui.label(egui::RichText::new(
                                                "CC1–9 → field · CC20–27 → feedback · Notes 36/38/40/41/43/45 → triggers"
                                            ).small().color(egui::Color32::from_gray(130)));
                                        }
                                    }); // end Modulation collapsible

                                ui.separator();

                                // K-path readout (toggle is in the top toolbar)
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("k =").small()
                                        .color(egui::Color32::from_gray(150)));
                                    ui.monospace(cur_kpt_label.to_string());
                                });

                                ui.separator();

                                // Crystal nav (transport: ATOMS/FIELD, TOUR, ?-help are all in toolbar)
                                ui.horizontal(|ui| {
                                    if ui.button("◀ PREV").clicked() { req.prev = true; }
                                    if ui.button("NEXT ▶").clicked() { req.next = true; }
                                });

                                ui.separator();

                            });
                    }

                    // ── Consolidated bottom sequencer panel ─────────────
                    // One source of truth: transport, grid, presets, and (when a
                    // step is selected) an inline editor — all in a single dock.
                    egui::TopBottomPanel::bottom("sequencer_panel")
                        .resizable(false)
                        .show(ctx, |ui| {
                            ui.add_space(2.0);
                            // ── Status line (replaces the old bottom OSD)
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new(format!(
                                    "{cur_crystal_name}  ·  {cur_sys_name}  ·  {cur_mode_name}  ·  k={cur_kpt_label}"
                                )).monospace().size(11.0)
                                  .color(egui::Color32::from_rgba_unmultiplied(210, 210, 255, 210)));
                            });

                            // ── Inline step editor — only when a step is selected
                            if let Some((edit_idx, ref mut ep)) = seq_selected_edit {
                                let is_muted = cur_seq_steps.get(edit_idx)
                                    .map(|&(_, m, _, _)| m).unwrap_or(false);
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgba_unmultiplied(20, 16, 36, 200))
                                    .rounding(4.0)
                                    .inner_margin(egui::vec2(8.0, 6.0))
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.label(egui::RichText::new(format!("◆ STEP {}", edit_idx + 1))
                                                .strong().color(egui::Color32::from_rgb(255, 210, 60)));
                                            egui::ComboBox::from_id_salt("step_mode_combo")
                                                .selected_text(MODE_NAMES[ep.mode as usize])
                                                .width(160.0)
                                                .show_ui(ui, |ui| {
                                                    for (i, name) in MODE_NAMES.iter().enumerate() {
                                                        ui.selectable_value(&mut ep.mode, i as u32, *name);
                                                    }
                                                });
                                            let mute_col = if is_muted {
                                                egui::Color32::from_rgb(255, 80, 80)
                                            } else { egui::Color32::from_gray(160) };
                                            if ui.add(egui::Button::new(
                                                egui::RichText::new(if is_muted { "MUTED" } else { "MUTE" }).color(mute_col)
                                            )).clicked() { req.seq_mute_step = Some(edit_idx); }
                                            if ui.small_button("RAND").clicked() { req.seq_randomize = Some(edit_idx); }
                                            if ui.small_button("✕").on_hover_text("Close step editor").clicked() {
                                                req.seq_select_step = Some(edit_idx); // toggles off
                                            }
                                        });

                                        macro_rules! esl {
                                            ($ui:expr, $label:expr, $val:expr, $min:expr, $max:expr) => {
                                                $ui.horizontal(|ui| {
                                                    ui.label(egui::RichText::new($label).small().monospace());
                                                    ui.add(egui::Slider::new($val, $min..=$max).show_value(true));
                                                });
                                            }
                                        }
                                        // Two columns to keep the editor short.
                                        ui.horizontal(|ui| {
                                            ui.vertical(|ui| {
                                                ui.set_min_width(220.0);
                                                esl!(ui, "kscale   ", &mut ep.kscale,      0.1_f32, 5.0_f32);
                                                esl!(ui, "speed    ", &mut ep.speed,       0.0_f32, 2.0_f32);
                                                esl!(ui, "field_mix", &mut ep.field_mix,   0.0_f32, 1.0_f32);
                                                esl!(ui, "iso_level", &mut ep.iso_level,   0.0_f32, 1.0_f32);
                                                esl!(ui, "color_sft", &mut ep.color_shift, 0.0_f32, 1.0_f32);
                                            });
                                            ui.separator();
                                            ui.vertical(|ui| {
                                                ui.set_min_width(220.0);
                                                esl!(ui, "zoom     ", &mut ep.zoom,        0.2_f32, 5.0_f32);
                                                esl!(ui, "w_lattice", &mut ep.w_lattice,   0.0_f32, 2.0_f32);
                                                esl!(ui, "w_motif  ", &mut ep.w_motif,     0.0_f32, 2.0_f32);
                                                esl!(ui, "w_band   ", &mut ep.w_band,      0.0_f32, 2.0_f32);
                                            });
                                        });

                                        // Per-step timing/curve/prob overrides
                                        if let Some((mut dm, curve_ov, mut pr)) = seq_selected_extras {
                                            ui.horizontal(|ui| {
                                                ui.label(egui::RichText::new("dur×").small().monospace());
                                                if ui.add(egui::Slider::new(&mut dm, 0.25_f32..=4.0_f32)
                                                    .show_value(true)).changed() {
                                                    req.seq_step_dur_mul = Some((edit_idx, dm));
                                                }
                                                ui.label(egui::RichText::new("curve").small().monospace());
                                                let lbl = match curve_ov {
                                                    Some(c) => c.label(),
                                                    None    => "(global)",
                                                };
                                                if ui.small_button(lbl).clicked() {
                                                    req.seq_step_curve_cycle = Some(edit_idx);
                                                }
                                                ui.label(egui::RichText::new("prob").small().monospace());
                                                if ui.add(egui::Slider::new(&mut pr, 0.0_f32..=1.0_f32)
                                                    .show_value(true)).changed() {
                                                    req.seq_step_prob = Some((edit_idx, pr));
                                                }
                                            });
                                        }
                                    });
                                ui.add_space(4.0);
                            }

                            // ── Transport row
                            ui.horizontal(|ui| {
                                // Play / stop
                                let seq_col = if cur_seq_active {
                                    egui::Color32::from_rgb(140, 255, 140)
                                } else { egui::Color32::from_gray(220) };
                                if ui.add(egui::Button::new(
                                    egui::RichText::new(if cur_seq_active { "■ STOP" } else { "▶ PLAY" }).color(seq_col)
                                )).clicked() { req.seq_toggle = true; }

                                // Manual + step arrows
                                let man_col = if cur_seq_manual {
                                    egui::Color32::from_rgb(255, 200, 60)
                                } else { egui::Color32::from_gray(150) };
                                if ui.add(egui::Button::new(
                                    egui::RichText::new("MANUAL").color(man_col)
                                )).clicked() { req.seq_manual_toggle = true; }
                                if cur_seq_manual {
                                    if ui.small_button("◀").clicked() { req.seq_manual_step = Some(-1); }
                                    if ui.small_button("▶").clicked() { req.seq_manual_step = Some(1); }
                                }
                                ui.separator();

                                // Play mode + curve
                                let pm_col = egui::Color32::from_rgb(180, 200, 255);
                                if ui.add(egui::Button::new(
                                    egui::RichText::new(cur_seq_play_mode.label()).color(pm_col)
                                )).on_hover_text("Play mode: Forward / Reverse / PingPong / Random").clicked() {
                                    req.seq_play_mode_toggle = true;
                                }
                                if ui.small_button(cur_seq_curve.label())
                                    .on_hover_text("Global transition curve").clicked() {
                                    req.seq_curve = Some(cur_seq_curve.next());
                                }

                                // Step duration slider
                                let mut dur = cur_seq_dur;
                                ui.label(egui::RichText::new("step").small());
                                if ui.add(egui::Slider::new(&mut dur, 0.5_f32..=8.0_f32)
                                    .show_value(true).suffix("s")).changed() {
                                    req.seq_dur = Some(dur);
                                }
                                ui.separator();

                                // Step count + add/remove/capture
                                let n = cur_seq_steps.len();
                                ui.label(egui::RichText::new(format!("{n}/32")).small()
                                    .color(egui::Color32::from_gray(160)));
                                let can_add = n < 32;
                                let can_del = n > 1 && cur_seq_selected.is_some();
                                if ui.add_enabled(can_add, egui::Button::new("+")).clicked() {
                                    req.seq_add_step = true;
                                }
                                if ui.add_enabled(can_del, egui::Button::new("−"))
                                    .on_hover_text("Remove selected step").clicked() {
                                    req.seq_remove_step = true;
                                }
                                if ui.add_enabled(can_add, egui::Button::new(
                                    egui::RichText::new("📸 CAPTURE")
                                        .color(egui::Color32::from_rgb(255, 200, 120))
                                )).on_hover_text("Append a new step from the current live state").clicked() {
                                    req.seq_capture = true;
                                }
                                ui.separator();

                                // Presets
                                for (i, (name, _)) in SEQ_PRESETS.iter().enumerate() {
                                    if ui.small_button(*name).clicked() { req.seq_preset = Some(i); }
                                }
                            });

                            // ── Step grid
                            ui.add_space(4.0);
                            egui::Frame::none()
                                .fill(egui::Color32::from_rgba_unmultiplied(12, 10, 24, 220))
                                .rounding(4.0)
                                .inner_margin(egui::vec2(8.0, 6.0))
                                .show(ui, |ui| {
                                    let n = cur_seq_steps.len();
                                    let avail_w = ui.available_width() - 16.0;
                                    let cell_w = (avail_w / n.max(1) as f32 - 3.0)
                                        .min(48.0).max(16.0);
                                    let cell_h = 32.0_f32;

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
                                });
                            ui.add_space(2.0);
                        });

                    // (OSD is now folded into the bottom sequencer panel header above.)

                    // ── Keymap help overlay ───────────────────────────
                    if cur_show_keymap {
                        let mut open = true;
                        egui::Window::new("Keyboard shortcuts")
                            .id(egui::Id::new("keymap_overlay"))
                            .open(&mut open)
                            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                            .resizable(false)
                            .collapsible(false)
                            .default_width(320.0)
                            .show(ctx, |ui| {
                                egui::Grid::new("keymap_grid")
                                    .num_columns(2).spacing(egui::vec2(16.0, 4.0))
                                    .show(ui, |ui| {
                                        let kt = egui::Color32::from_rgb(220, 220, 255);
                                        let row = |ui: &mut egui::Ui, k: &str, d: &str| {
                                            ui.label(egui::RichText::new(k).monospace().color(kt));
                                            ui.label(d);
                                            ui.end_row();
                                        };
                                        ui.label(egui::RichText::new("VIEW")
                                            .small().color(egui::Color32::from_gray(150)));
                                        ui.label(""); ui.end_row();
                                        row(ui, "Tab",   "toggle Atoms / Field render mode");
                                        row(ui, "Space", "toggle auto-rotate camera");
                                        row(ui, "1 / 2 / 3", "supercell size (Atoms mode)");
                                        row(ui, "M", "cycle field mode");
                                        ui.label(egui::RichText::new("FIELD")
                                            .small().color(egui::Color32::from_gray(150)));
                                        ui.label(""); ui.end_row();
                                        row(ui, "[  ]", "color shift down / up");
                                        row(ui, "R", "randomize field G-vectors");
                                        row(ui, "K", "snap to next k-point");
                                        row(ui, "P", "toggle k-path walk");
                                        ui.label(egui::RichText::new("FEEDBACK / AUTO")
                                            .small().color(egui::Color32::from_gray(150)));
                                        ui.label(""); ui.end_row();
                                        row(ui, "A", "toggle FB AUTO (auto-pilot)");
                                        row(ui, "T", "toggle TOUR");
                                        ui.label(egui::RichText::new("UI")
                                            .small().color(egui::Color32::from_gray(150)));
                                        ui.label(""); ui.end_row();
                                        row(ui, "? / /", "toggle this help");
                                        row(ui, "Esc", "exit");
                                        ui.label(egui::RichText::new("MIDI (any channel)")
                                            .small().color(egui::Color32::from_gray(150)));
                                        ui.label(""); ui.end_row();
                                        row(ui, "CC 1–9",   "field params (kscale, speed, …, w_band)");
                                        row(ui, "CC 20–27", "feedback (decay, zoom, …, motion_blur)");
                                        row(ui, "Note 36 C2", "toggle FB AUTO");
                                        row(ui, "Note 38 D2", "toggle TOUR");
                                        row(ui, "Note 40 E2", "toggle SEQ");
                                        row(ui, "Note 41 F2", "SEQ prev step (manual)");
                                        row(ui, "Note 43 G2", "SEQ next step (manual)");
                                        row(ui, "Note 45 A2", "📸 CAPTURE step");
                                    });
                                ui.add_space(4.0);
                                ui.label(egui::RichText::new("Click · / A / B / M / T on a slider to route mic / LFO source.")
                                    .small().color(egui::Color32::from_gray(150)));
                            });
                        if !open { req.keymap_toggle = true; }
                    }
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

                // ── MIDI Note triggers — route them into the same UiReq path so
                //    they behave exactly like clicking the corresponding button.
                for n in midi_notes {
                    match n {
                        midi::note::FB_AUTO_TOGGLE => req.fb_auto_toggle = true,
                        midi::note::TOUR_TOGGLE    => req.tour_toggle = true,
                        midi::note::SEQ_TOGGLE     => req.seq_toggle = true,
                        midi::note::SEQ_PREV       => req.seq_manual_step = Some(-1),
                        midi::note::SEQ_NEXT       => req.seq_manual_step = Some(1),
                        midi::note::SEQ_CAPTURE    => req.seq_capture = true,
                        _ => {}
                    }
                }

                if req.panel_toggle { self.panel_open = !cur_panel_open; }
                if req.keymap_toggle { self.show_keymap = !cur_show_keymap; }
                if req.fb_auto_toggle {
                    self.fb_auto = !cur_fb_auto;
                    // Turning auto on triggers a clean feedback restart and ensures fb_enabled
                    if self.fb_auto {
                        self.field_params.fb_enabled = true;
                        req.fb_reset = true;
                    }
                }

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
                    self.lfo = match self.tour_style {
                        TourStyle::Curated => tour_lfo_preset(),
                        TourStyle::Random  => tour_lfo_preset_fb_heavy(),
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
                    if i < SEQ_PRESETS.len() {
                        self.sequencer.load_preset(SEQ_PRESETS[i].1);
                        self.sequencer.active = true;
                        self.render_mode = RenderMode::Field;
                    }
                }
                if let Some(c) = req.seq_curve { self.sequencer.curve = c; }
                if let Some(d) = req.seq_dur { self.sequencer.step_dur = d; }
                if req.seq_play_mode_toggle {
                    self.sequencer.play_mode = self.sequencer.play_mode.next();
                    self.sequencer.pp_dir = 1; // reset PingPong direction on mode change
                }
                if req.seq_capture {
                    // Snapshot the *effective* state (post-LFO/mic, post-fb_auto) as a new step.
                    if let Some(new_idx) = self.sequencer.capture(fp_eff.clone()) {
                        self.sequencer.selected = Some(new_idx);
                    }
                }
                if let Some((i, v)) = req.seq_step_dur_mul {
                    if let Some(s) = self.sequencer.steps.get_mut(i) {
                        s.dur_mul = v.clamp(0.25, 4.0);
                    }
                }
                if let Some(i) = req.seq_step_curve_cycle {
                    if let Some(s) = self.sequencer.steps.get_mut(i) {
                        s.curve_override = match s.curve_override {
                            None                          => Some(TranCurve::Linear),
                            Some(TranCurve::Linear)       => Some(TranCurve::EaseInOut),
                            Some(TranCurve::EaseInOut)    => Some(TranCurve::Snap),
                            Some(TranCurve::Snap)         => Some(TranCurve::Bounce),
                            Some(TranCurve::Bounce)       => None,
                        };
                    }
                }
                if let Some((i, v)) = req.seq_step_prob {
                    if let Some(s) = self.sequencer.steps.get_mut(i) {
                        s.prob = v.clamp(0.0, 1.0);
                    }
                }
                if req.fb_reset {
                    if let Some(gpu) = &mut self.gpu { gpu.fb_clear = true; }
                }
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
            ..FieldParams::default()
        })).collect::<Vec<_>>();
        let from_params = steps[0].params.clone();
        Sequencer {
            active: true, manual: true,
            cur: 0, step_dur: 2.0, step_timer: 2.0, // fully arrived at step 0
            curve: TranCurve::Linear,
            selected: None,
            play_mode: SeqPlayMode::Forward,
            pp_dir: 1, rng_seed: 0x9E3779B9,
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

    // ── Per-step duration multiplier ──────────────────────────────────────
    #[test]
    fn per_step_dur_mul_extends_step() {
        let mut seq = make_seq(2);
        seq.manual = false;
        seq.step_dur = 1.0;
        seq.steps[0].dur_mul = 2.0; // step 0 lasts 2.0s effectively
        seq.step_timer = 0.0;
        seq.tick(1.5);
        assert_eq!(seq.cur, 0, "should not advance before effective dur");
        seq.tick(0.6); // total 2.1 > 2.0
        assert_eq!(seq.cur, 1, "should advance when effective dur exceeded");
    }

    #[test]
    fn per_step_dur_mul_shortens_step() {
        let mut seq = make_seq(2);
        seq.manual = false;
        seq.step_dur = 1.0;
        seq.steps[0].dur_mul = 0.5; // step 0 lasts 0.5s
        seq.step_timer = 0.0;
        seq.tick(0.4);
        assert_eq!(seq.cur, 0);
        seq.tick(0.2); // total 0.6 > 0.5
        assert_eq!(seq.cur, 1);
    }

    // ── Per-step curve override ───────────────────────────────────────────
    #[test]
    fn per_step_curve_override_applied() {
        let mut seq = make_seq(2);
        seq.curve = TranCurve::Linear;
        seq.steps[1].curve_override = Some(TranCurve::Snap);
        seq.manual_step(1); // cur=1, from=steps[0], timer=0
        seq.step_timer = seq.step_dur * 0.5;
        let p = seq.current_params();
        // SNAP returns from until t >= 1.0
        assert!((p.kscale - seq.from_params.kscale).abs() < 1e-5,
            "snap override at t=0.5 should output from_params");
    }

    #[test]
    fn no_curve_override_uses_global() {
        let mut seq = make_seq(2);
        seq.curve = TranCurve::Linear;
        seq.manual_step(1);
        seq.step_timer = seq.step_dur * 0.5;
        let p = seq.current_params();
        let expected = (seq.from_params.kscale + seq.steps[seq.cur].params.kscale) * 0.5;
        assert!((p.kscale - expected).abs() < 1e-5,
            "linear at t=0.5 should be exact midpoint");
    }

    // ── Play modes ────────────────────────────────────────────────────────
    #[test]
    fn play_mode_reverse_steps_backward() {
        let mut seq = make_seq(4);
        seq.play_mode = SeqPlayMode::Reverse;
        seq.cur = 2;
        seq.advance_next();
        assert_eq!(seq.cur, 1);
        seq.advance_next();
        assert_eq!(seq.cur, 0);
        seq.advance_next();
        assert_eq!(seq.cur, 3, "Reverse wraps from 0 to last");
    }

    #[test]
    fn play_mode_pingpong_reverses_at_ends() {
        let mut seq = make_seq(3);
        seq.play_mode = SeqPlayMode::PingPong;
        seq.pp_dir = 1;
        seq.cur = 0;
        seq.advance_next(); assert_eq!(seq.cur, 1);
        seq.advance_next(); assert_eq!(seq.cur, 2);
        seq.advance_next(); assert_eq!(seq.cur, 1, "should reverse off the end");
        seq.advance_next(); assert_eq!(seq.cur, 0);
        seq.advance_next(); assert_eq!(seq.cur, 1, "should reverse off the start");
    }

    #[test]
    fn play_mode_pingpong_with_one_step_stays_put() {
        let mut seq = make_seq(1);
        seq.play_mode = SeqPlayMode::PingPong;
        seq.advance_next();
        assert_eq!(seq.cur, 0);
    }

    #[test]
    fn play_mode_random_avoids_immediate_repeat() {
        let mut seq = make_seq(4);
        seq.play_mode = SeqPlayMode::Random;
        seq.rng_seed = 12345;
        for _ in 0..50 {
            let prev = seq.cur;
            seq.advance_next();
            assert_ne!(seq.cur, prev, "Random must not repeat current step");
        }
    }

    #[test]
    fn play_mode_random_visits_multiple_steps() {
        let mut seq = make_seq(8);
        seq.play_mode = SeqPlayMode::Random;
        seq.rng_seed = 7;
        let mut visited = [false; 8];
        for _ in 0..200 {
            seq.advance_next();
            visited[seq.cur] = true;
        }
        let count = visited.iter().filter(|&&v| v).count();
        assert!(count >= 6, "Random should hit at least 6/8 steps, hit {count}");
    }

    // ── Probability gating ────────────────────────────────────────────────
    #[test]
    fn prob_zero_step_is_always_skipped() {
        let mut seq = make_seq(4);
        seq.play_mode = SeqPlayMode::Forward;
        seq.steps[1].prob = 0.0; // step 1 never plays
        seq.cur = 0;
        seq.advance_next();
        assert_eq!(seq.cur, 2, "prob=0 must be skipped");
    }

    #[test]
    fn prob_one_step_always_plays() {
        let mut seq = make_seq(4);
        seq.steps[1].prob = 1.0;
        seq.cur = 0;
        seq.advance_next();
        assert_eq!(seq.cur, 1);
    }

    #[test]
    fn prob_half_step_lands_roughly_half_the_time() {
        let mut seq = make_seq(4);
        seq.steps[1].prob = 0.5;
        seq.rng_seed = 42;
        let mut plays = 0;
        for _ in 0..200 {
            seq.cur = 0;
            seq.play_mode = SeqPlayMode::Forward;
            seq.advance_next();
            if seq.cur == 1 { plays += 1; }
        }
        // generous bounds — just rules out 0% and 100%
        assert!(plays > 30 && plays < 170,
            "prob=0.5 should fire roughly half the time (got {plays}/200)");
    }

    #[test]
    fn all_prob_zero_does_not_hang() {
        let mut seq = make_seq(3);
        for s in seq.steps.iter_mut() { s.prob = 0.0; }
        seq.advance_next(); // must terminate
        // cur may be anywhere; the contract is "no hang"
        assert!(seq.cur < seq.steps.len());
    }

    // ── effective_step_dur / effective_curve ──────────────────────────────
    #[test]
    fn effective_step_dur_floor() {
        let mut seq = make_seq(1);
        seq.step_dur = 0.001;
        seq.steps[0].dur_mul = 0.001;
        assert!(seq.effective_step_dur() >= 0.05,
            "effective_step_dur must clamp above 0.05 to avoid /0");
    }

    // ── CAPTURE ───────────────────────────────────────────────────────────
    #[test]
    fn capture_appends_step() {
        let mut seq = make_seq(3);
        let mut p = FieldParams::default();
        p.kscale = 7.7;
        let idx = seq.capture(p).expect("capture should fit");
        assert_eq!(idx, 3);
        assert_eq!(seq.steps.len(), 4);
        assert!((seq.steps[3].params.kscale - 7.7).abs() < 1e-5);
        // captured step defaults: not muted, dur_mul=1, prob=1
        assert_eq!(seq.steps[3].muted, false);
        assert!((seq.steps[3].dur_mul - 1.0).abs() < 1e-5);
        assert!((seq.steps[3].prob - 1.0).abs() < 1e-5);
    }

    #[test]
    fn capture_blocked_at_32() {
        let mut seq = make_seq(32);
        let r = seq.capture(FieldParams::default());
        assert!(r.is_none(), "capture must refuse beyond 32 steps");
        assert_eq!(seq.steps.len(), 32);
    }
}

// ── Tests for parameter routing, LFO, lerp_fp, and fb_auto ───────────────
#[cfg(test)]
mod routing_tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool { (a - b).abs() < eps }

    // LfoWave::sample must stay in [-1, 1] for every wave at any phase.
    #[test]
    fn lfo_wave_sample_in_bipolar_range() {
        for w in [LfoWave::Sine, LfoWave::Triangle, LfoWave::Saw,
                   LfoWave::Square, LfoWave::Pulse, LfoWave::Steps] {
            for k in 0..1000 {
                let p = (k as f32) * 0.013;
                let s = w.sample(p);
                assert!(s >= -1.0 - 1e-4 && s <= 1.0 + 1e-4,
                    "{:?} out of range at phase {p}: {s}", w);
            }
        }
    }

    #[test]
    fn lfo_sine_periodic() {
        // Sine should repeat every 1 unit of phase (TAU is folded by .fract()).
        let a = LfoWave::Sine.sample(0.25);
        let b = LfoWave::Sine.sample(1.25);
        assert!(approx_eq(a, b, 1e-4), "sine should be 1-periodic in phase");
        // and zero at 0 and 0.5
        assert!(approx_eq(LfoWave::Sine.sample(0.0), 0.0, 1e-4));
        assert!(approx_eq(LfoWave::Sine.sample(0.5), 0.0, 1e-4));
    }

    #[test]
    fn lfo_triangle_endpoints() {
        // Triangle: 1 at 0.25, -1 at 0.75, 0 at 0/0.5
        assert!(approx_eq(LfoWave::Triangle.sample(0.0), -1.0, 1e-4));
        assert!(approx_eq(LfoWave::Triangle.sample(0.25), 0.0, 1e-4));
        assert!(approx_eq(LfoWave::Triangle.sample(0.5), 1.0, 1e-4));
        assert!(approx_eq(LfoWave::Triangle.sample(0.75), 0.0, 1e-4));
    }

    #[test]
    fn lfo_square_two_levels() {
        for k in 0..50 {
            let p = (k as f32) / 100.0; // 0..0.5
            assert!(approx_eq(LfoWave::Square.sample(p), 1.0, 1e-4));
        }
        for k in 50..100 {
            let p = (k as f32) / 100.0; // 0.5..1
            assert!(approx_eq(LfoWave::Square.sample(p), -1.0, 1e-4));
        }
    }

    // apply_modulation clamps every output param into the documented UI range.
    #[test]
    fn apply_modulation_clamps_ranges() {
        let fp = FieldParams::default();
        let mut lfo = LfoParams::default();
        let mic = MicParams::default();
        let bands = audio::AudioBands::default();
        // Crank everything: route all targets to LFO A, huge depth, square wave (±1).
        for src in [
            &mut lfo.kscale, &mut lfo.speed, &mut lfo.field_mix, &mut lfo.iso_level,
            &mut lfo.color_shift, &mut lfo.zoom, &mut lfo.w_lattice, &mut lfo.w_motif,
            &mut lfo.w_band, &mut lfo.fb_zoom, &mut lfo.fb_decay, &mut lfo.fb_offset_x,
            &mut lfo.fb_offset_y, &mut lfo.fb_rotation, &mut lfo.fb_color_shift,
            &mut lfo.fb_saturation, &mut lfo.fb_brightness, &mut lfo.fb_inject,
            &mut lfo.fb_fold_angle, &mut lfo.fb_motion_blur,
        ] { *src = LfoSrc::A; }
        lfo.a.depth = 10.0;             // wildly more than any param range
        lfo.a.wave = LfoWave::Square;   // always ±1
        lfo.a.phase = 0.0;

        // phase=0 → square at t=0 is +1
        let eff_pos = apply_modulation(&fp, &lfo, &mic, &bands, 0.0);
        // pick a t where square is -1 (phase >= 0.5 in cycle)
        let t_neg = 0.6 / lfo.a.rate.max(1e-6);
        let eff_neg = apply_modulation(&fp, &lfo, &mic, &bands, t_neg);

        let check = |label: &str, v: f32, lo: f32, hi: f32| {
            assert!(v >= lo - 1e-4 && v <= hi + 1e-4, "{label}={v} outside [{lo},{hi}]");
        };
        for eff in [&eff_pos, &eff_neg] {
            check("kscale", eff.kscale, 0.1, 5.0);
            check("speed", eff.speed, 0.0, 2.0);
            check("field_mix", eff.field_mix, 0.0, 1.0);
            check("iso_level", eff.iso_level, 0.0, 1.0);
            check("color_shift", eff.color_shift, 0.0, 1.0);
            check("zoom", eff.zoom, 0.2, 5.0);
            check("w_lattice", eff.w_lattice, 0.0, 2.0);
            check("w_motif",   eff.w_motif,   0.0, 2.0);
            check("w_band",    eff.w_band,    0.0, 2.0);
            check("fb_zoom",        eff.fb_zoom,        0.90, 1.10);
            check("fb_decay",       eff.fb_decay,       0.30, 0.99);
            check("fb_offset_x",    eff.fb_offset_x,   -0.10, 0.10);
            check("fb_offset_y",    eff.fb_offset_y,   -0.10, 0.10);
            check("fb_rotation",    eff.fb_rotation,   -0.30, 0.30);
            check("fb_color_shift", eff.fb_color_shift,-1.00, 1.00);
            check("fb_saturation",  eff.fb_saturation,  0.00, 2.00);
            check("fb_brightness",  eff.fb_brightness,  0.00, 2.00);
            check("fb_inject",      eff.fb_inject,      0.00, 1.00);
            check("fb_fold_angle",  eff.fb_fold_angle, -3.14, 3.14);
            check("fb_motion_blur", eff.fb_motion_blur, 0.00, 0.95);
        }
    }

    // apply_modulation with all-off LFO and Off mic must pass fp through unchanged.
    #[test]
    fn apply_modulation_passes_through_when_off() {
        let fp = FieldParams {
            kscale: 1.7, speed: 0.42, field_mix: 0.6,
            iso_level: 0.31, color_shift: 0.77,
            ..FieldParams::default()
        };
        let lfo = LfoParams::default();
        let mic = MicParams::default();
        let bands = audio::AudioBands::default();
        let eff = apply_modulation(&fp, &lfo, &mic, &bands, 1.23);
        assert!(approx_eq(eff.kscale, fp.kscale, 1e-5));
        assert!(approx_eq(eff.speed,  fp.speed,  1e-5));
        assert!(approx_eq(eff.field_mix, fp.field_mix, 1e-5));
        assert!(approx_eq(eff.color_shift, fp.color_shift, 1e-5));
    }

    // lerp_fp must wrap color_shift via the *short* arc (hue is circular).
    #[test]
    fn lerp_fp_color_shift_takes_short_arc_forward() {
        let mut a = FieldParams::default(); a.color_shift = 0.9;
        let mut b = FieldParams::default(); b.color_shift = 0.1;
        // Short path: 0.9 → 1.0/0.0 → 0.1 (forward through hue wheel).
        // At t=0.25 the unwrapped delta is (b - a) - 1.0 = -0.8, so
        // result = 0.9 + 0.25*(-0.8 + 1.0) = 0.9 + 0.05 = 0.95... wait the implementation
        // gives short-way delta of +0.2 (since |d|=0.8>0.5 → d-1=-0.8, but our code uses
        // the d<-0.5 branch as d+1=+0.2 for d<-0.5). Let's verify d:
        // d = 0.1 - 0.9 = -0.8 → matches `d < -0.5` → cs_delta = -0.8 + 1 = +0.2
        // so result_t=0.25 = (0.9 + 0.2*0.25).rem_euclid(1) = 0.95.
        // And at t=0.75: 0.9 + 0.2*0.75 = 1.05 → rem_euclid → 0.05.
        let r25 = lerp_fp(&a, &b, 0.25);
        let r75 = lerp_fp(&a, &b, 0.75);
        assert!(approx_eq(r25.color_shift, 0.95, 1e-4),
            "forward wrap @t=0.25 should be ~0.95, got {}", r25.color_shift);
        assert!(approx_eq(r75.color_shift, 0.05, 1e-4),
            "forward wrap @t=0.75 should be ~0.05, got {}", r75.color_shift);
    }

    #[test]
    fn lerp_fp_color_shift_no_wrap_for_short_delta() {
        let mut a = FieldParams::default(); a.color_shift = 0.2;
        let mut b = FieldParams::default(); b.color_shift = 0.4;
        let r = lerp_fp(&a, &b, 0.5);
        // Plain linear mid: 0.3
        assert!(approx_eq(r.color_shift, 0.3, 1e-5));
    }

    #[test]
    fn lerp_fp_endpoints() {
        let mut a = FieldParams::default(); a.kscale = 1.0; a.zoom = 0.5;
        let mut b = FieldParams::default(); b.kscale = 5.0; b.zoom = 2.0;
        let r0 = lerp_fp(&a, &b, 0.0);
        let r1 = lerp_fp(&a, &b, 1.0);
        assert!(approx_eq(r0.kscale, 1.0, 1e-5));
        assert!(approx_eq(r0.zoom,   0.5, 1e-5));
        assert!(approx_eq(r1.kscale, 5.0, 1e-5));
        assert!(approx_eq(r1.zoom,   2.0, 1e-5));
    }

    #[test]
    fn lerp_fp_discrete_switches_at_half() {
        let mut a = FieldParams::default(); a.mode = 1; a.fb_blend_mode = 0;
        let mut b = FieldParams::default(); b.mode = 7; b.fb_blend_mode = 3;
        assert_eq!(lerp_fp(&a, &b, 0.0).mode, 1);
        assert_eq!(lerp_fp(&a, &b, 0.49).mode, 1);
        assert_eq!(lerp_fp(&a, &b, 0.5).mode, 7,  "discrete field switches at t=0.5");
        assert_eq!(lerp_fp(&a, &b, 1.0).mode, 7);
        assert_eq!(lerp_fp(&a, &b, 0.49).fb_blend_mode, 0);
        assert_eq!(lerp_fp(&a, &b, 0.5).fb_blend_mode,  3);
    }

    // fb_auto_params must override every fb_* field of base, and only those.
    #[test]
    fn fb_auto_overrides_only_fb_fields() {
        let base = FieldParams {
            mode: 5, kscale: 2.5, speed: 0.7, field_mix: 0.6,
            iso_level: 0.4, color_shift: 0.3, zoom: 1.5,
            w_lattice: 1.7, w_motif: 0.9, w_band: 0.6,
            fb_enabled: false, fb_mirror: 0, fb_blend_mode: 0,
            ..FieldParams::default()
        };
        let auto = fb_auto_params(3.7, &base);
        // Non-feedback fields preserved exactly:
        assert_eq!(auto.mode, base.mode);
        assert!(approx_eq(auto.kscale,    base.kscale,    1e-5));
        assert!(approx_eq(auto.speed,     base.speed,     1e-5));
        assert!(approx_eq(auto.field_mix, base.field_mix, 1e-5));
        assert!(approx_eq(auto.iso_level, base.iso_level, 1e-5));
        assert!(approx_eq(auto.color_shift, base.color_shift, 1e-5));
        assert!(approx_eq(auto.zoom,      base.zoom,      1e-5));
        assert!(approx_eq(auto.w_lattice, base.w_lattice, 1e-5));
        assert!(approx_eq(auto.w_motif,   base.w_motif,   1e-5));
        assert!(approx_eq(auto.w_band,    base.w_band,    1e-5));
        // Feedback always enabled in auto mode:
        assert!(auto.fb_enabled, "fb_auto must force fb_enabled");
        // fb_* must stay in valid UI ranges (smoke-check a few)
        assert!(auto.fb_zoom >= 0.90 && auto.fb_zoom <= 1.10);
        assert!(auto.fb_decay >= 0.30 && auto.fb_decay <= 0.99);
        assert!(auto.fb_motion_blur >= 0.0 && auto.fb_motion_blur <= 0.95);
        assert!(auto.fb_inject >= 0.0 && auto.fb_inject <= 1.0);
        assert!((auto.fb_mirror as usize) < 12);
        assert!((auto.fb_blend_mode as usize) < 12);
    }

    #[test]
    fn fb_auto_evolves_over_time() {
        let base = FieldParams::default();
        let a = fb_auto_params(0.5, &base);
        let b = fb_auto_params(9.0, &base);
        // Different scenes — at least one of (mirror, zoom, decay) should differ.
        let differs = a.fb_mirror != b.fb_mirror
            || !approx_eq(a.fb_zoom, b.fb_zoom, 1e-3)
            || !approx_eq(a.fb_decay, b.fb_decay, 1e-3);
        assert!(differs, "fb_auto must evolve across scenes (a={:?}, b={:?})",
            (a.fb_mirror, a.fb_zoom, a.fb_decay), (b.fb_mirror, b.fb_zoom, b.fb_decay));
    }

    #[test]
    fn fb_auto_is_finite() {
        let base = FieldParams::default();
        for k in 0..500 {
            let t = (k as f32) * 0.13;
            let a = fb_auto_params(t, &base);
            for v in [a.fb_zoom, a.fb_decay, a.fb_offset_x, a.fb_offset_y,
                      a.fb_rotation, a.fb_color_shift, a.fb_saturation,
                      a.fb_brightness, a.fb_inject, a.fb_fold_angle, a.fb_motion_blur] {
                assert!(v.is_finite(), "fb_auto produced non-finite value at t={t}");
            }
        }
    }

    // ── LfoSrc enum behaviour ─────────────────────────────────────────────
    #[test]
    fn lfo_src_cycles_off_a_b() {
        let mut s = LfoSrc::Off;
        s = s.next(); assert_eq!(s, LfoSrc::A);
        s = s.next(); assert_eq!(s, LfoSrc::B);
        s = s.next(); assert_eq!(s, LfoSrc::Off);
    }

    // ── LfoEngine sampling ────────────────────────────────────────────────
    #[test]
    fn lfo_engine_phase_offset_shifts_sample() {
        let e0 = LfoEngine { rate: 1.0, depth: 1.0, wave: LfoWave::Sine, phase: 0.0 };
        let e1 = LfoEngine { rate: 1.0, depth: 1.0, wave: LfoWave::Sine, phase: 0.25 };
        // sin(2π·0) = 0; sin(2π·0.25) = 1.
        assert!(approx_eq(e0.sample(0.0),  0.0, 1e-4));
        assert!(approx_eq(e1.sample(0.0),  1.0, 1e-4));
    }

    // ── Two independent LFOs combine in apply_modulation ──────────────────
    #[test]
    fn apply_modulation_uses_two_lfos_independently() {
        let fp = FieldParams::default();
        let mic = MicParams::default();
        let bands = audio::AudioBands::default();
        let mut lfo = LfoParams::default();
        // LFO A: square at +1 → bumps kscale upward; LFO B: square at -1 → pulls speed down.
        lfo.a = LfoEngine { rate: 0.5, depth: 1.0, wave: LfoWave::Square, phase: 0.0 };
        lfo.b = LfoEngine { rate: 0.5, depth: 1.0, wave: LfoWave::Square, phase: 0.5 };
        lfo.kscale = LfoSrc::A;
        lfo.speed  = LfoSrc::B;
        let eff = apply_modulation(&fp, &lfo, &mic, &bands, 0.0);
        // A at t=0 → +1. B at t=0 with phase 0.5 → -1.
        // kscale base 1.4 + 1.0*range(4.9) clamped to 5.0
        assert!(eff.kscale > 4.5, "LFO A (square +1) should pull kscale to max; got {}", eff.kscale);
        // speed base 0.3 − range(2.0) → clamped to 0.0
        assert!(eff.speed < 0.05, "LFO B (square -1) should pull speed to min; got {}", eff.speed);
    }

    #[test]
    fn apply_modulation_off_src_does_not_modulate() {
        let fp = FieldParams { kscale: 2.0, ..FieldParams::default() };
        let mic = MicParams::default();
        let bands = audio::AudioBands::default();
        let mut lfo = LfoParams::default();
        lfo.a.depth = 1.0; lfo.a.wave = LfoWave::Square; lfo.a.phase = 0.0;
        lfo.kscale = LfoSrc::Off; // explicitly off
        let eff = apply_modulation(&fp, &lfo, &mic, &bands, 0.0);
        assert!((eff.kscale - 2.0).abs() < 1e-5,
            "LfoSrc::Off must not change the param");
    }

    // ── tour_lfo_preset wiring ────────────────────────────────────────────
    #[test]
    fn tour_lfo_preset_has_both_engines_configured() {
        let p = tour_lfo_preset();
        assert!(p.a.depth > 0.0 && p.a.rate > 0.0);
        assert!(p.b.depth > 0.0 && p.b.rate > 0.0);
        // At least one target routed to A and one to B
        let any_a = [p.kscale, p.fb_saturation, p.color_shift].into_iter().any(|s| s == LfoSrc::A);
        let any_b = [p.fb_zoom, p.fb_color_shift, p.fb_fold_angle].into_iter().any(|s| s == LfoSrc::B);
        assert!(any_a && any_b,
            "tour_lfo_preset should route some targets to A and some to B");
    }

    #[test]
    fn tour_lfo_preset_fb_heavy_drives_feedback() {
        let p = tour_lfo_preset_fb_heavy();
        let fb_targets = [
            p.fb_zoom, p.fb_decay, p.fb_color_shift, p.fb_saturation, p.fb_brightness,
            p.fb_rotation, p.fb_offset_x, p.fb_offset_y, p.fb_fold_angle, p.fb_motion_blur,
        ];
        let active = fb_targets.iter().filter(|&&s| s != LfoSrc::Off).count();
        assert!(active >= 6,
            "fb_heavy preset should route at least 6 feedback params to an LFO, got {active}");
    }
}
