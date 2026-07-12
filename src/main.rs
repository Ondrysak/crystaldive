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
mod preset;
use crystals::{all_crystals, all_groups, CrystalDef};
use poscar::Crystal;
use reciprocal::GpuField;
use renderer::{FieldUniform, GpuState};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};

// ── field parameters ──────────────────────────────────────────────────────

/// Per-mode parameter bank: 16 generic float slots, reinterpreted per render
/// mode. Slots 0..8 are the canonical crystal-field generator params (their
/// indices below mirror the WGSL `MP_*` consts in `prelude.wgsl`); slots 9..15
/// are free for each mode. The slot count must match `FieldUniform`'s 4×vec4
/// pack and the WGSL `array<vec4<f32>, 4>`.
pub const MP_SLOTS: usize = 16;
pub const MP_KSCALE: usize = 0;
pub const MP_SPEED: usize = 1;
pub const MP_FIELD_MIX: usize = 2;
pub const MP_ISO_LEVEL: usize = 3;
pub const MP_COLOR_SHIFT: usize = 4;
pub const MP_ZOOM: usize = 5;
pub const MP_W_LATTICE: usize = 6;
pub const MP_W_MOTIF: usize = 7;
pub const MP_W_BAND: usize = 8;

/// Default values for the canonical generator slots (0..8); free slots are 0.
pub fn default_mp() -> [f32; MP_SLOTS] {
    let mut m = [0.0f32; MP_SLOTS];
    m[MP_KSCALE] = 1.4;
    m[MP_SPEED] = 0.3;
    m[MP_FIELD_MIX] = 0.55;
    m[MP_ISO_LEVEL] = 0.5;
    m[MP_COLOR_SHIFT] = 0.0;
    m[MP_ZOOM] = 1.0;
    m[MP_W_LATTICE] = 1.0;
    m[MP_W_MOTIF] = 0.6;
    m[MP_W_BAND] = 0.4;
    m
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct FieldParams {
    pub mode: u32,
    #[serde(default = "default_mp")]
    pub mp: [f32; MP_SLOTS],
    // feedback
    pub fb_enabled: bool,
    pub fb_mirror: u32,
    pub fb_zoom: f32,
    pub fb_offset_x: f32,
    pub fb_offset_y: f32,
    pub fb_rotation: f32,
    pub fb_decay: f32,
    pub fb_color_shift: f32,
    pub fb_inject: f32,
    pub fb_fold_angle: f32,
    pub fb_saturation: f32,
    pub fb_brightness: f32,
    pub fb_blend_mode: u32,
    pub fb_motion_blur: f32,
}

impl Default for FieldParams {
    fn default() -> Self {
        Self {
            mode: 4,
            mp: default_mp(),
            fb_enabled: false,
            fb_mirror: 0,
            fb_zoom: 0.98,
            fb_offset_x: 0.0,
            fb_offset_y: 0.0,
            fb_rotation: 0.0,
            fb_decay: 0.85,
            fb_color_shift: 0.0,
            fb_inject: 1.0,
            fb_fold_angle: 0.0,
            fb_saturation: 1.0,
            fb_brightness: 1.0,
            fb_blend_mode: 0,
            fb_motion_blur: 0.0,
        }
    }
}

use crystal_viz::modes::{mode_params, slot_range, ModeArea, ModeInfo, MODES, MODE_NAMES};

// ── LFO ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Default, Debug, serde::Serialize, serde::Deserialize)]
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
            Self::Square => {
                if x < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            Self::Pulse => {
                if x < 0.18 {
                    1.0
                } else {
                    -0.35
                }
            }
            Self::Steps => {
                let n = (x * 8.0).floor();
                bz_hash(n + phase.floor() * 17.0) * 2.0 - 1.0
            }
        }
    }
}

pub fn bz_hash(n: f32) -> f32 {
    (n * 127.1 + 19.19).sin().fract().abs()
}

const TAU: f32 = std::f32::consts::PI * 2.0;

/// Which LFO bank drives a target parameter.
#[derive(Clone, Copy, PartialEq, Default, Debug, serde::Serialize, serde::Deserialize)]
pub enum LfoSrc {
    #[default]
    Off,
    A,
    B,
}

impl LfoSrc {
    fn next(self) -> Self {
        match self {
            Self::Off => Self::A,
            Self::A => Self::B,
            Self::B => Self::Off,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Off => "~",
            Self::A => "A",
            Self::B => "B",
        }
    }
    fn color(self) -> egui::Color32 {
        match self {
            Self::Off => egui::Color32::from_gray(90),
            Self::A => egui::Color32::from_rgb(80, 220, 120),
            Self::B => egui::Color32::from_rgb(120, 180, 255),
        }
    }
}

/// One LFO bank: a waveform, rate, depth, and phase offset.
/// Two independent banks (A and B) live inside `LfoParams`.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct LfoEngine {
    pub rate: f32,  // cycles per second
    pub depth: f32, // 0..1 — scaled by each param's range
    pub wave: LfoWave,
    pub phase: f32, // phase offset in cycles (0..1), lets B de-sync from A
}

impl LfoEngine {
    fn sample(&self, t: f32) -> f32 {
        self.wave.sample(self.rate * t + self.phase)
    }
}

fn lfo_mp_default() -> [LfoSrc; MP_SLOTS] {
    [LfoSrc::Off; MP_SLOTS]
}

/// Reset the active mode's declared slots to their per-mode default values.
/// Called on a *manual* mode switch so a freshly-picked mode starts sensible;
/// sequencer/preset mode changes carry their own `mp` and are not touched.
fn apply_mode_defaults(fp: &mut FieldParams) {
    for p in crystal_viz::modes::mode_params(fp.mode) {
        fp.mp[p.slot] = p.default;
    }
}

/// Seed a mode's *free* slots (9..15) with its declared defaults. The preset/
/// random/tour generators only fill the canonical slots 0..8, so without this a
/// mode that reads a bespoke free slot (e.g. Lorenz's sigma/rho) would see 0.
/// Canonical slots are left untouched (the generator owns them).
fn fill_free_slot_defaults(fp: &mut FieldParams) {
    for p in crystal_viz::modes::mode_params(fp.mode) {
        if p.slot >= 9 {
            fp.mp[p.slot] = p.default;
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct LfoParams {
    pub a: LfoEngine,
    pub b: LfoEngine,
    // per-mode param bank — each slot picks LfoSrc::{Off, A, B}
    #[serde(default = "lfo_mp_default")]
    pub mp: [LfoSrc; MP_SLOTS],
    // feedback params
    pub fb_zoom: LfoSrc,
    pub fb_decay: LfoSrc,
    pub fb_offset_x: LfoSrc,
    pub fb_offset_y: LfoSrc,
    pub fb_rotation: LfoSrc,
    pub fb_color_shift: LfoSrc,
    pub fb_saturation: LfoSrc,
    pub fb_brightness: LfoSrc,
    pub fb_inject: LfoSrc,
    pub fb_fold_angle: LfoSrc,
    pub fb_motion_blur: LfoSrc,
    /// Total modulation budget (0..1): caps the summed, range-normalized
    /// LFO + mic swing across every routed target. When routes ask for more
    /// than the budget allows, all of them are scaled down proportionally —
    /// "everything moves, nothing explodes".
    #[serde(default = "flux_default")]
    pub flux: f32,
}

fn flux_default() -> f32 {
    0.5
}

impl Default for LfoParams {
    fn default() -> Self {
        // Routed out of the box: gentle hue/zoom breathing on A, texture sway
        // on B. Depths are small — the point is "alive at first launch", the
        // flux budget keeps any later knob-twisting bounded.
        let mut mp = lfo_mp_default();
        mp[MP_COLOR_SHIFT] = LfoSrc::A;
        mp[MP_ZOOM] = LfoSrc::A;
        mp[MP_W_MOTIF] = LfoSrc::B;
        Self {
            a: LfoEngine {
                rate: 0.11,
                depth: 0.10,
                wave: LfoWave::Sine,
                phase: 0.0,
            },
            b: LfoEngine {
                rate: 0.045,
                depth: 0.14,
                wave: LfoWave::Triangle,
                phase: 0.25,
            },
            mp,
            fb_zoom: LfoSrc::Off,
            fb_decay: LfoSrc::Off,
            fb_offset_x: LfoSrc::Off,
            fb_offset_y: LfoSrc::Off,
            fb_rotation: LfoSrc::B,
            fb_color_shift: LfoSrc::Off,
            fb_saturation: LfoSrc::Off,
            fb_brightness: LfoSrc::Off,
            fb_inject: LfoSrc::Off,
            fb_fold_angle: LfoSrc::Off,
            fb_motion_blur: LfoSrc::Off,
            flux: flux_default(),
        }
    }
}

impl LfoParams {
    /// Engines configured but every route off — the base for generators that
    /// build routing incrementally (trip levels) and for routing-free tests.
    pub fn unrouted() -> Self {
        Self {
            mp: lfo_mp_default(),
            fb_zoom: LfoSrc::Off,
            fb_decay: LfoSrc::Off,
            fb_offset_x: LfoSrc::Off,
            fb_offset_y: LfoSrc::Off,
            fb_rotation: LfoSrc::Off,
            fb_color_shift: LfoSrc::Off,
            fb_saturation: LfoSrc::Off,
            fb_brightness: LfoSrc::Off,
            fb_inject: LfoSrc::Off,
            fb_fold_angle: LfoSrc::Off,
            fb_motion_blur: LfoSrc::Off,
            ..Self::default()
        }
    }
}

// ── Mic modulation ────────────────────────────────────────────────────────

/// Which audio band (or none) drives a parameter.
#[derive(Clone, Copy, PartialEq, Default)]
pub enum MicSrc {
    #[default]
    Off,
    Amp,
    Bass,
    Mid,
    Treble,
}

impl MicSrc {
    fn next(self) -> Self {
        match self {
            Self::Off => Self::Amp,
            Self::Amp => Self::Bass,
            Self::Bass => Self::Mid,
            Self::Mid => Self::Treble,
            Self::Treble => Self::Off,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Off => "·",
            Self::Amp => "A",
            Self::Bass => "B",
            Self::Mid => "M",
            Self::Treble => "T",
        }
    }
    fn color(self) -> egui::Color32 {
        match self {
            Self::Off => egui::Color32::from_gray(70),
            Self::Amp => egui::Color32::from_rgb(220, 220, 220),
            Self::Bass => egui::Color32::from_rgb(255, 80, 80),
            Self::Mid => egui::Color32::from_rgb(80, 220, 120),
            Self::Treble => egui::Color32::from_rgb(80, 160, 255),
        }
    }
    fn value(self, b: &audio::AudioBands) -> f32 {
        match self {
            Self::Off => 0.0,
            Self::Amp => b.amplitude,
            Self::Bass => b.bass,
            Self::Mid => b.mid,
            Self::Treble => b.treble,
        }
    }
}

#[derive(Clone)]
pub struct MicParams {
    // per-mode param bank — each slot picks a MicSrc band
    pub mp: [MicSrc; MP_SLOTS],
    // feedback params
    pub fb_zoom: MicSrc,
    pub fb_decay: MicSrc,
    pub fb_offset_x: MicSrc,
    pub fb_offset_y: MicSrc,
    pub fb_rotation: MicSrc,
    pub fb_color_shift: MicSrc,
    pub fb_saturation: MicSrc,
    pub fb_brightness: MicSrc,
    pub fb_inject: MicSrc,
    pub fb_fold_angle: MicSrc,
    pub fb_motion_blur: MicSrc,
    pub depth: f32,
}

impl Default for MicParams {
    fn default() -> Self {
        Self {
            mp: [MicSrc::Off; MP_SLOTS],
            fb_zoom: MicSrc::Off,
            fb_decay: MicSrc::Off,
            fb_offset_x: MicSrc::Off,
            fb_offset_y: MicSrc::Off,
            fb_rotation: MicSrc::Off,
            fb_color_shift: MicSrc::Off,
            fb_saturation: MicSrc::Off,
            fb_brightness: MicSrc::Off,
            fb_inject: MicSrc::Off,
            fb_fold_angle: MicSrc::Off,
            fb_motion_blur: MicSrc::Off,
            depth: 0.5,
        }
    }
}

// ── Combined LFO + Mic modulation ─────────────────────────────────────────

const TOUR_MODES: [u32; 14] = [22, 20, 14, 21, 10, 17, 27, 4, 28, 15, 18, 12, 23, 29];

// ── Trip level ────────────────────────────────────────────────────────────
//
// 0 = stillness (no LFOs routed, single mode, no feedback layer, slow walks)
// 9 = total madness (steps/pulse LFOs, fast mode swaps, heavy fractal feedback)
// Every other knob (scene length, walk duration, LFO rate/depth/wave, feedback
// routing) is interpolated smoothly between the two.

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TripLevel(u8);

impl Default for TripLevel {
    fn default() -> Self {
        Self(4)
    }
}

impl TripLevel {
    pub const MAX_VAL: u8 = 9;

    pub fn new(v: u8) -> Self {
        Self(v.min(Self::MAX_VAL))
    }
    pub fn get(self) -> u8 {
        self.0
    }
    pub fn inc(self) -> Self {
        Self((self.0 + 1).min(Self::MAX_VAL))
    }
    pub fn dec(self) -> Self {
        Self(self.0.saturating_sub(1))
    }

    fn t(self) -> f32 {
        self.0 as f32 / Self::MAX_VAL as f32
    }
    fn lerp(self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.t()
    }

    fn wave_a(self) -> LfoWave {
        match self.0 {
            0..=2 => LfoWave::Sine,
            3..=5 => LfoWave::Triangle,
            6..=7 => LfoWave::Saw,
            8 => LfoWave::Pulse,
            _ => LfoWave::Steps,
        }
    }
    fn wave_b(self) -> LfoWave {
        match self.0 {
            0..=1 => LfoWave::Sine,
            2..=4 => LfoWave::Triangle,
            5..=6 => LfoWave::Saw,
            7 => LfoWave::Pulse,
            _ => LfoWave::Square,
        }
    }

    fn scene_len(self) -> f32 {
        self.lerp(28.0, 1.8)
    }
    fn walk_dur(self) -> f32 {
        self.lerp(28.0, 2.0)
    }
    fn fade_dur(self) -> f32 {
        self.lerp(3.0, 0.8)
    }
    fn lfo_a_rate(self) -> f32 {
        self.lerp(0.03, 1.00)
    }
    fn lfo_b_rate(self) -> f32 {
        self.lerp(0.02, 0.65)
    }
    fn lfo_a_depth(self) -> f32 {
        self.lerp(0.0, 0.70)
    }
    fn lfo_b_depth(self) -> f32 {
        self.lerp(0.0, 0.55)
    }
    fn fb_enabled(self) -> bool {
        self.0 >= 3
    }
    fn fb_strength(self) -> f32 {
        if self.0 < 3 {
            0.0
        } else {
            let t = (self.0 - 3) as f32 / (Self::MAX_VAL - 3) as f32;
            0.35 + 0.65 * t
        }
    }

    fn label(self) -> &'static str {
        match self.0 {
            0 => "0 still",
            1 => "1 drift",
            2 => "2 calm",
            3 => "3 sway",
            4 => "4 flow",
            5 => "5 swirl",
            6 => "6 churn",
            7 => "7 wobble",
            8 => "8 strobe",
            _ => "9 madness",
        }
    }
}

// ── Sequencer ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TranCurve {
    Linear,
    EaseInOut,
    Silk,
    Snap,
    Bounce,
    Over,
}

impl TranCurve {
    fn apply(self, t: f32) -> f32 {
        match self {
            Self::Linear => t,
            Self::EaseInOut => t * t * (3.0 - 2.0 * t),
            Self::Silk => {
                // Double smoothstep: long soft ends, brisk middle — buttery.
                let s = t * t * (3.0 - 2.0 * t);
                s * s * (3.0 - 2.0 * s)
            }
            Self::Snap => {
                if t >= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Bounce => {
                // quartic ease-in-out: quick rush then slow settle
                if t < 0.5 {
                    8.0 * t * t * t * t
                } else {
                    let u = t - 1.0;
                    1.0 - 8.0 * u * u * u * u
                }
            }
            Self::Over => {
                // Ease-out-back: overshoot ~10 % past target, then settle.
                // Downstream range clamps keep the excursion safe.
                const C1: f32 = 1.70158;
                const C3: f32 = C1 + 1.0;
                let u = t - 1.0;
                1.0 + C3 * u * u * u + C1 * u * u
            }
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Linear => "LINEAR",
            Self::EaseInOut => "EASE",
            Self::Silk => "SILK",
            Self::Snap => "SNAP",
            Self::Bounce => "BOUNCE",
            Self::Over => "OVER",
        }
    }
    fn next(self) -> Self {
        match self {
            Self::Linear => Self::EaseInOut,
            Self::EaseInOut => Self::Silk,
            Self::Silk => Self::Snap,
            Self::Snap => Self::Bounce,
            Self::Bounce => Self::Over,
            Self::Over => Self::Linear,
        }
    }
}

/// Elektron-style cycle condition: step plays on pass `k` of every `n` visits.
/// `(1,1)` = always. Ratios cycle through `TRIG_CONDS` in the UI.
pub const TRIG_CONDS: [(u8, u8); 7] = [(1, 1), (1, 2), (2, 2), (1, 3), (1, 4), (4, 4), (1, 8)];

fn cond_default() -> (u8, u8) {
    (1, 1)
}

pub fn cond_label(cond: (u8, u8)) -> String {
    if cond == (1, 1) {
        "--".to_string()
    } else {
        format!("{}:{}", cond.0, cond.1)
    }
}

pub fn cond_passes(cond: (u8, u8), visits: u32) -> bool {
    let (k, n) = cond;
    if n <= 1 {
        return true;
    }
    visits.wrapping_sub(1) % n as u32 == (k.max(1) - 1) as u32
}

fn cond_next(cond: (u8, u8)) -> (u8, u8) {
    let i = TRIG_CONDS.iter().position(|&c| c == cond).unwrap_or(0);
    TRIG_CONDS[(i + 1) % TRIG_CONDS.len()]
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SeqStep {
    pub params: FieldParams,
    pub muted: bool,
    /// Per-step duration multiplier in [0.25, 4.0]. 1.0 = global step_dur.
    pub dur_mul: f32,
    /// Per-step transition curve override (None = use Sequencer.curve).
    pub curve_override: Option<TranCurve>,
    /// Probability the step plays when advance lands on it (0..1, 1.0 = always).
    pub prob: f32,
    /// Elektron cycle condition `(k, n)`: play on visit k of every n.
    #[serde(default = "cond_default")]
    pub cond: (u8, u8),
    /// Runtime visit counter for `cond` (not persisted).
    #[serde(skip)]
    pub visits: u32,
}

impl SeqStep {
    pub fn new(p: FieldParams) -> Self {
        Self {
            params: p,
            muted: false,
            dur_mul: 1.0,
            curve_override: None,
            prob: 1.0,
            cond: cond_default(),
            visits: 0,
        }
    }
}

/// Step traversal order for the sequencer.
#[derive(Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum SeqPlayMode {
    #[default]
    Forward,
    Reverse,
    PingPong,
    Random,
}

impl SeqPlayMode {
    fn next(self) -> Self {
        match self {
            Self::Forward => Self::Reverse,
            Self::Reverse => Self::PingPong,
            Self::PingPong => Self::Random,
            Self::Random => Self::Forward,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Forward => "FWD",
            Self::Reverse => "REV",
            Self::PingPong => "PP",
            Self::Random => "RND",
        }
    }
}

fn lerp_fp(a: &FieldParams, b: &FieldParams, t: f32) -> FieldParams {
    let l = |x: f32, y: f32| x + (y - x) * t;
    let d = b.mp[MP_COLOR_SHIFT] - a.mp[MP_COLOR_SHIFT];
    let cs_delta = if d > 0.5 {
        d - 1.0
    } else if d < -0.5 {
        d + 1.0
    } else {
        d
    };
    // Discrete fields snap to `b` immediately at t=0 (Digitakt-style trig: when
    // the sequencer advances to step N, that step's mode/mirror/blend is what
    // we see for the whole step duration — no half-step desync against the
    // visual highlight). Continuous fields still glide smoothly across the step.
    let mut mp = [0.0f32; MP_SLOTS];
    for i in 0..MP_SLOTS {
        mp[i] = if i == MP_COLOR_SHIFT {
            // Hue wraps on the unit circle — interpolate the shortest arc.
            (a.mp[MP_COLOR_SHIFT] + cs_delta * t).rem_euclid(1.0)
        } else {
            l(a.mp[i], b.mp[i])
        };
    }
    FieldParams {
        mode: b.mode,
        mp,
        fb_enabled: b.fb_enabled,
        fb_mirror: b.fb_mirror,
        fb_zoom: l(a.fb_zoom, b.fb_zoom),
        fb_offset_x: l(a.fb_offset_x, b.fb_offset_x),
        fb_offset_y: l(a.fb_offset_y, b.fb_offset_y),
        fb_rotation: l(a.fb_rotation, b.fb_rotation),
        fb_decay: l(a.fb_decay, b.fb_decay),
        fb_color_shift: l(a.fb_color_shift, b.fb_color_shift),
        fb_inject: l(a.fb_inject, b.fb_inject),
        fb_fold_angle: l(a.fb_fold_angle, b.fb_fold_angle),
        fb_saturation: l(a.fb_saturation, b.fb_saturation),
        fb_brightness: l(a.fb_brightness, b.fb_brightness),
        fb_blend_mode: b.fb_blend_mode,
        fb_motion_blur: l(a.fb_motion_blur, b.fb_motion_blur),
    }
}

/// The always-alive micro-motion lane. Slot-aware, deliberately tiny, and
/// phase-scattered across incommensurate rates so it never settles or repeats.
/// Every excursion is bounded by construction (≤ a few % of a knob's span at
/// amt = 1) and the result still passes through `apply_modulation`'s range
/// clamps downstream — controlled motion, not chaos.
fn apply_drift(fp: &mut FieldParams, clock: f32, amt: f32, step_idx: usize) {
    if amt <= 0.001 {
        return;
    }
    let ph = step_idx as f32 * 0.618;
    let osc = |rate: f32, phase: f32| (TAU * (rate * clock) + phase).sin();
    // Hue crawl: slow monotonic rotation — the surest "something is happening".
    fp.mp[MP_COLOR_SHIFT] = (fp.mp[MP_COLOR_SHIFT] + amt * 0.010 * clock).rem_euclid(1.0);
    // Breathing: proportional wobble keeps scale relationships intact.
    fp.mp[MP_ZOOM] *= 1.0 + amt * 0.045 * osc(0.050, ph);
    fp.mp[MP_KSCALE] *= 1.0 + amt * 0.030 * osc(0.017, ph * 2.0);
    fp.mp[MP_SPEED] *= 1.0 + amt * 0.100 * osc(0.023, 1.7);
    fp.mp[MP_W_MOTIF] += amt * 0.120 * osc(0.031, ph);
    fp.mp[MP_W_BAND] += amt * 0.100 * osc(0.041, ph + 2.1);
    if fp.fb_enabled {
        fp.fb_rotation += amt * 0.030 * osc(0.019, 0.0);
        fp.fb_zoom += amt * 0.006 * osc(0.043, ph);
    }
}

// Compact builder used by presets. The 9 positional args map to the canonical
// generator slots (0..8); free slots (9..15) stay at their defaults.
fn sp(
    mode: u32,
    ks: f32,
    sp: f32,
    fm: f32,
    il: f32,
    cs: f32,
    zm: f32,
    wl: f32,
    wm: f32,
    wb: f32,
) -> SeqStep {
    let mut mp = default_mp();
    mp[MP_KSCALE] = ks;
    mp[MP_SPEED] = sp;
    mp[MP_FIELD_MIX] = fm;
    mp[MP_ISO_LEVEL] = il;
    mp[MP_COLOR_SHIFT] = cs;
    mp[MP_ZOOM] = zm;
    mp[MP_W_LATTICE] = wl;
    mp[MP_W_MOTIF] = wm;
    mp[MP_W_BAND] = wb;
    let mut fp = FieldParams {
        mode,
        mp,
        ..FieldParams::default()
    };
    fill_free_slot_defaults(&mut fp);
    SeqStep::new(fp)
}

fn seq_preset_phase_space() -> Vec<SeqStep> {
    vec![
        sp(1, 1.5, 0.35, 0.30, 0.55, 0.00, 1.0, 1.5, 0.3, 0.5), // BZ SLICE
        sp(2, 2.0, 0.45, 0.50, 0.60, 0.12, 1.1, 0.8, 1.2, 0.8), // FERMI
        sp(4, 1.8, 0.30, 0.65, 0.40, 0.25, 0.9, 1.2, 1.8, 0.6), // NODAL
        sp(6, 2.5, 0.55, 0.40, 0.50, 0.37, 1.3, 0.5, 0.8, 1.8), // STRIPES
        sp(5, 1.2, 0.80, 0.55, 0.45, 0.50, 1.0, 1.2, 0.6, 0.8), // PHASE
        sp(12, 2.0, 0.40, 0.70, 0.50, 0.62, 0.9, 0.6, 1.0, 1.5), // KIKUCHI
        sp(9, 1.0, 0.25, 0.45, 0.60, 0.75, 0.8, 1.0, 1.2, 0.9), // MOIRE
        sp(16, 1.8, 0.50, 0.35, 0.70, 0.87, 1.2, 0.7, 0.5, 2.0), // CDW
    ]
}

fn seq_preset_quantum() -> Vec<SeqStep> {
    vec![
        sp(10, 1.2, 0.20, 0.50, 0.55, 0.00, 1.2, 1.0, 0.6, 0.4), // WANNIER
        sp(3, 1.5, 0.30, 0.55, 0.50, 0.12, 1.0, 1.8, 0.4, 0.3),  // DENSITY
        sp(20, 2.0, 0.40, 0.50, 0.55, 0.25, 0.9, 0.8, 1.0, 1.2), // BERRY
        sp(23, 1.5, 0.30, 0.45, 0.60, 0.37, 1.0, 1.0, 0.5, 0.8), // NEMATIC
        sp(15, 1.8, 0.35, 0.30, 0.50, 0.50, 1.1, 1.2, 0.8, 0.5), // SPIN TEXTURE
        sp(14, 2.2, 0.45, 0.60, 0.45, 0.62, 1.3, 1.5, 0.6, 0.7), // BAND SURFACE
        sp(21, 1.0, 0.35, 0.60, 0.55, 0.75, 1.5, 0.5, 1.2, 0.6), // STM
        sp(22, 1.5, 0.25, 0.50, 0.50, 0.87, 1.0, 0.8, 0.9, 1.0), // VORTEX KNOT
    ]
}

fn seq_preset_geometric() -> Vec<SeqStep> {
    vec![
        sp(0, 1.5, 0.30, 0.50, 0.40, 0.00, 1.2, 1.0, 0.6, 0.8), // 3D ISO
        sp(8, 1.2, 0.35, 0.30, 0.50, 0.12, 1.0, 0.8, 1.0, 1.2), // NONEUC
        sp(7, 2.0, 0.80, 0.55, 0.45, 0.25, 0.9, 0.6, 0.8, 1.8), // WARP
        sp(22, 1.8, 0.45, 0.40, 0.55, 0.37, 1.3, 1.2, 0.5, 1.0), // VORTEX KNOT
        sp(17, 1.5, 0.55, 0.60, 0.65, 0.50, 1.1, 0.7, 1.2, 0.8), // QUASICRYSTAL
        sp(24, 1.2, 0.35, 0.45, 0.55, 0.62, 1.1, 1.5, 0.6, 0.5), // ABRIKOSOV
        sp(19, 1.8, 0.40, 0.55, 0.50, 0.75, 1.0, 0.8, 1.0, 1.8), // DOMAIN WALL
        sp(34, 2.0, 0.50, 0.35, 0.60, 0.87, 1.4, 0.5, 1.5, 1.0), // STRAIN FIELD
    ]
}

fn seq_preset_chromatic() -> Vec<SeqStep> {
    let modes: [u32; 16] = [27, 22, 20, 23, 10, 3, 0, 28, 9, 17, 29, 12, 5, 15, 14, 8];
    modes
        .iter()
        .enumerate()
        .map(|(i, &mode)| {
            let phi = i as f32 / 16.0;
            let mut mp = default_mp();
            mp[MP_KSCALE] = 1.0 + 1.2 * (phi * TAU).sin().abs();
            mp[MP_SPEED] = 0.2 + 0.6 * (phi * TAU * 0.7).cos().abs();
            mp[MP_FIELD_MIX] = 0.3 + 0.5 * (phi * TAU * 1.3).sin().abs();
            mp[MP_ISO_LEVEL] = 0.3 + 0.4 * (phi * TAU * 0.5).cos().abs();
            mp[MP_COLOR_SHIFT] = phi;
            mp[MP_ZOOM] = 0.8 + 0.6 * (phi * TAU * 1.1).sin().abs();
            mp[MP_W_LATTICE] = 0.5 + 1.2 * (phi * TAU).cos().abs();
            mp[MP_W_MOTIF] = 0.3 + 1.0 * (phi * TAU * 1.7).sin().abs();
            mp[MP_W_BAND] = 0.4 + 1.2 * (phi * TAU * 0.9).cos().abs();
            let mut fp = FieldParams {
                mode,
                mp,
                ..FieldParams::default()
            };
            fill_free_slot_defaults(&mut fp);
            SeqStep::new(fp)
        })
        .collect()
}

/// Default pattern: built to show the sequencer's articulation vocabulary in
/// one loop — varied step lengths, cycle conditions, curve accents, and two
/// feedback scenes — while every step stays readable on its own.
fn seq_preset_showcase() -> Vec<SeqStep> {
    let mut steps = vec![
        sp(4, 1.8, 0.30, 0.65, 0.40, 0.00, 0.9, 1.2, 1.8, 0.6), // NODAL — home base
        sp(2, 2.0, 0.45, 0.50, 0.60, 0.13, 1.1, 0.8, 1.2, 0.8), // FERMI — quick accent
        sp(9, 1.0, 0.25, 0.45, 0.60, 0.28, 0.8, 1.0, 1.2, 0.9), // MOIRE — long dwell
        sp(22, 1.8, 0.45, 0.40, 0.55, 0.42, 1.3, 1.2, 0.5, 1.0), // VORTEX KNOT — snap hit
        sp(17, 1.5, 0.55, 0.60, 0.65, 0.55, 1.1, 0.7, 1.2, 0.8), // QUASICRYSTAL — fb bloom
        sp(6, 2.5, 0.55, 0.40, 0.50, 0.68, 1.3, 0.5, 0.8, 1.8), // STRIPES — every 2nd pass
        sp(20, 2.0, 0.40, 0.50, 0.55, 0.80, 0.9, 0.8, 1.0, 1.2), // BERRY — overshoot jump
        sp(12, 2.0, 0.40, 0.70, 0.50, 0.92, 0.9, 0.6, 1.0, 1.5), // KIKUCHI — fb mirror coda
    ];
    // Quick accent: half-length, hard cut in.
    steps[1].dur_mul = 0.5;
    steps[1].curve_override = Some(TranCurve::Snap);
    // Long dwell: double-length silk glide.
    steps[2].dur_mul = 2.0;
    // Snap hit that only lands on the first of every two passes.
    steps[3].dur_mul = 0.5;
    steps[3].curve_override = Some(TranCurve::Snap);
    steps[3].cond = (1, 2);
    // Feedback bloom scene.
    steps[4].params.fb_enabled = true;
    steps[4].params.fb_mirror = 5;
    steps[4].params.fb_decay = 0.88;
    steps[4].params.fb_inject = 0.55;
    steps[4].params.fb_zoom = 1.012;
    // Alternating answer to the snap hit: second of every two passes.
    steps[5].cond = (2, 2);
    // Big harmonic jump with overshoot-settle.
    steps[6].curve_override = Some(TranCurve::Over);
    // Mirrored feedback coda, longer dwell.
    steps[7].dur_mul = 1.5;
    steps[7].params.fb_enabled = true;
    steps[7].params.fb_mirror = 3;
    steps[7].params.fb_decay = 0.90;
    steps[7].params.fb_inject = 0.60;
    steps[7].params.fb_rotation = 0.05;
    steps
}

const SEQ_PRESETS: [(&str, fn() -> Vec<SeqStep>); 5] = [
    ("SHOWCASE", seq_preset_showcase),
    ("PHASE", seq_preset_phase_space),
    ("QUANTUM", seq_preset_quantum),
    ("GEO", seq_preset_geometric),
    ("CHROMA", seq_preset_chromatic),
];

pub struct Sequencer {
    pub active: bool,
    pub manual: bool,
    pub steps: Vec<SeqStep>,
    pub cur: usize,
    pub step_dur: f32,
    pub step_timer: f32,
    pub curve: TranCurve,
    pub selected: Option<usize>,
    pub play_mode: SeqPlayMode,
    pub pp_dir: i8,               // direction for PingPong: +1 or -1 (0 = not started)
    pub rng_seed: u32,            // LCG state for Random mode and prob gates
    pub from_params: FieldParams, // pub so preset::apply can rewind to step 0 cleanly
    /// Fraction of the step spent morphing (Elektron slide-trig feel):
    /// 0.25 = quick morph then hold; 1.0 = classic wall-to-wall glide.
    pub morph: f32,
    /// Depth of the built-in micro-motion lane (0 = off). Keeps the image
    /// alive during holds — bounded by construction, never chaotic.
    pub drift: f32,
    /// Free-running clock driving the drift lane (advances with tick).
    pub clock: f32,
}

impl Sequencer {
    fn new() -> Self {
        let steps = seq_preset_showcase();
        let from_params = steps[0].params.clone();
        Self {
            active: false,
            manual: false,
            steps,
            cur: 0,
            step_dur: 2.6,
            step_timer: 0.0,
            curve: TranCurve::Silk,
            selected: None,
            play_mode: SeqPlayMode::Forward,
            pp_dir: 1,
            rng_seed: 0x9E3779B9,
            from_params,
            morph: 0.6,
            drift: 0.5,
            clock: 0.0,
        }
    }

    /// Duration for the *current* step (respects per-step dur_mul).
    fn effective_step_dur(&self) -> f32 {
        let mul = self.steps.get(self.cur).map(|s| s.dur_mul).unwrap_or(1.0);
        (self.step_dur * mul).max(0.05)
    }

    /// Transition curve for the current step (per-step override else global).
    fn effective_curve(&self) -> TranCurve {
        self.steps
            .get(self.cur)
            .and_then(|s| s.curve_override)
            .unwrap_or(self.curve)
    }

    fn current_params(&self) -> FieldParams {
        let dur = self.effective_step_dur();
        let raw = (self.step_timer / dur).clamp(0.0, 1.0);
        // Morph window: complete the transition within the first `morph`
        // fraction of the step, then hold the destination.
        let m = (raw / self.morph.clamp(0.05, 1.0)).clamp(0.0, 1.0);
        let t = self.effective_curve().apply(m);
        let mut fp = lerp_fp(&self.from_params, &self.steps[self.cur].params, t);
        apply_drift(&mut fp, self.clock, self.drift, self.cur);
        fp
    }

    fn tick(&mut self, dt: f32) {
        if !self.active {
            return;
        }
        self.clock += dt;
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
        ((v >> 8) & 0x00FF_FFFF) as f32 / 16_777_216.0 // 24-bit fraction
    }

    /// Compute the next raw index using the play mode (no muted/prob filtering).
    fn raw_next_index(&mut self) -> usize {
        let n = self.steps.len();
        if n <= 1 {
            return 0;
        }
        match self.play_mode {
            SeqPlayMode::Forward => (self.cur + 1) % n,
            SeqPlayMode::Reverse => self.cur.checked_sub(1).unwrap_or(n - 1),
            SeqPlayMode::PingPong => {
                if self.pp_dir == 0 {
                    self.pp_dir = 1;
                }
                let cur = self.cur as i32;
                let mut nx = cur + self.pp_dir as i32;
                if nx < 0 || nx >= n as i32 {
                    self.pp_dir = -self.pp_dir;
                    nx = cur + self.pp_dir as i32;
                    if nx < 0 || nx >= n as i32 {
                        nx = cur;
                    }
                }
                nx as usize
            }
            SeqPlayMode::Random => {
                let mut nx = (self.rng_next() as usize) % n;
                if nx == self.cur {
                    nx = (nx + 1) % n;
                }
                nx
            }
        }
    }

    fn advance_next(&mut self) {
        let n = self.steps.len();
        if n == 0 {
            return;
        }
        // Try up to 2n candidates: skip muted, condition-failed, and
        // probability-failed steps.
        for _ in 0..(2 * n) {
            let cand = self.raw_next_index();
            self.cur = cand;
            if self.steps[cand].muted {
                continue;
            }
            // Elektron cycle condition: count this playhead visit, then gate.
            self.steps[cand].visits = self.steps[cand].visits.wrapping_add(1);
            if !cond_passes(self.steps[cand].cond, self.steps[cand].visits) {
                continue;
            }
            if self.steps[cand].prob < 0.999 {
                if self.rng_unit() > self.steps[cand].prob {
                    continue;
                }
            }
            return;
        }
        // All blocked (every step muted or prob=0): leave self.cur where it last landed.
    }

    fn manual_step(&mut self, dir: i32) {
        if self.steps.is_empty() {
            return;
        }
        self.from_params = self.current_params();
        self.step_timer = 0.0;
        let n = self.steps.len();
        if dir > 0 {
            self.advance_next();
        } else {
            let mut prev = self.cur.checked_sub(1).unwrap_or(n - 1);
            for _ in 0..n {
                if !self.steps[prev].muted {
                    break;
                }
                prev = prev.checked_sub(1).unwrap_or(n - 1);
            }
            self.cur = prev;
        }
    }

    fn add_step_after(&mut self, after: usize, seed: f32) {
        if self.steps.len() >= 32 {
            return;
        }
        let new_step = self
            .steps
            .get(after)
            .cloned()
            .unwrap_or_else(|| SeqStep::new(randomize_fp(seed)));
        let pos = (after + 1).min(self.steps.len());
        self.steps.insert(pos, new_step);
        if self.cur >= pos {
            self.cur += 1;
        }
        if let Some(sel) = self.selected {
            if sel >= pos {
                self.selected = Some(sel + 1);
            }
        }
    }

    fn remove_step(&mut self, idx: usize) {
        if self.steps.len() <= 1 || idx >= self.steps.len() {
            return;
        }
        self.steps.remove(idx);
        // shift cur down if removed step was before it; clamp only if cur was the removed step
        if self.cur > idx {
            self.cur -= 1;
        } else {
            self.cur = self.cur.min(self.steps.len() - 1);
        }
        self.selected = match self.selected {
            Some(s) if s == idx => None,
            Some(s) if s > idx => Some(s - 1),
            s => s,
        };
    }

    fn load_preset(&mut self, make: fn() -> Vec<SeqStep>) {
        self.steps = make();
        self.cur = 0;
        self.step_timer = 0.0;
        self.selected = None;
        if !self.steps.is_empty() {
            self.from_params = self.steps[0].params.clone();
        }
    }

    /// Append a captured FieldParams as a fresh step at the end. Returns the new
    /// step's index, or None if the 32-step cap was hit.
    fn capture(&mut self, params: FieldParams) -> Option<usize> {
        if self.steps.len() >= 32 {
            return None;
        }
        self.steps.push(SeqStep::new(params));
        Some(self.steps.len() - 1)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn total_dur_mul(&self) -> f32 {
        self.steps
            .iter()
            .map(|s| s.dur_mul.max(1e-6))
            .sum::<f32>()
            .max(1e-6)
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
    let f = h6 - h6.floor();
    let q = 1.0 - f;
    let (r, g, b) = match hi {
        0 => (1.0, f, 0.0),
        1 => (q, 1.0, 0.0),
        2 => (0.0, 1.0, f),
        3 => (0.0, q, 1.0),
        4 => (f, 0.0, 1.0),
        _ => (1.0, 0.0, q),
    };
    ((r * 200.0) as u8, (g * 200.0) as u8, (b * 200.0) as u8)
}

pub fn randomize_fp(seed: f32) -> FieldParams {
    let r = |s: f32| tour_rand(seed + s * 13.17);
    let fb_on = r(11.0) > 0.55; // ~45% chance of feedback
    let mirror_roll = r(12.0);
    let fb_mirror = if mirror_roll < 0.25 {
        0u32
    } else if mirror_roll < 0.45 {
        1
    } else if mirror_roll < 0.60 {
        3
    } else if mirror_roll < 0.72 {
        5
    } else if mirror_roll < 0.82 {
        6
    } else if mirror_roll < 0.90 {
        7
    } else {
        9
    };
    let mut mp = default_mp();
    mp[MP_KSCALE] = 0.3 + r(2.0) * 3.5;
    mp[MP_SPEED] = r(3.0) * 1.6;
    mp[MP_FIELD_MIX] = r(4.0);
    mp[MP_ISO_LEVEL] = 0.05 + r(5.0) * 0.9;
    mp[MP_COLOR_SHIFT] = r(6.0);
    mp[MP_ZOOM] = 0.4 + r(7.0) * 1.9;
    mp[MP_W_LATTICE] = r(8.0) * 2.0;
    mp[MP_W_MOTIF] = r(9.0) * 2.0;
    mp[MP_W_BAND] = r(10.0) * 2.0;
    FieldParams {
        mode: ((r(1.0) * MODE_NAMES.len() as f32) as u32).min(MODE_NAMES.len() as u32 - 1),
        mp,
        fb_enabled: fb_on,
        fb_mirror,
        fb_zoom: 0.93 + r(13.0) * 0.09,
        fb_decay: 0.60 + r(14.0) * 0.38,
        fb_color_shift: (r(15.0) - 0.5) * 0.6,
        fb_inject: 0.5 + r(16.0) * 0.5,
        fb_saturation: 0.7 + r(17.0) * 0.6,
        fb_brightness: 0.8 + r(18.0) * 0.4,
        fb_rotation: (r(19.0) - 0.5) * 0.06,
        fb_offset_x: (r(20.0) - 0.5) * 0.04,
        fb_offset_y: (r(21.0) - 0.5) * 0.04,
        fb_fold_angle: (r(22.0) - 0.5) * 2.0,
        fb_blend_mode: (r(23.0) * 4.0) as u32, // modes 0-3 most useful
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

/// Build the curated tour LFO preset for a given trip level.
///
/// At level 0 the LFOs are present but nothing is routed (silent). Each level
/// step wires more targets (color first, then geometry, then feedback), and
/// engine rate/depth/wave all scale up via `TripLevel`.
pub fn tour_lfo_preset_for_level(level: TripLevel) -> LfoParams {
    let mut p = LfoParams::unrouted();
    p.a = LfoEngine {
        rate: level.lfo_a_rate(),
        depth: level.lfo_a_depth(),
        wave: level.wave_a(),
        phase: 0.0,
    };
    p.b = LfoEngine {
        rate: level.lfo_b_rate(),
        depth: level.lfo_b_depth(),
        wave: level.wave_b(),
        phase: 0.33,
    };
    let l = level.get();
    if l >= 1 {
        p.mp[MP_COLOR_SHIFT] = LfoSrc::A;
        p.fb_color_shift = LfoSrc::B;
    }
    if l >= 2 {
        p.mp[MP_FIELD_MIX] = LfoSrc::A;
        p.fb_zoom = LfoSrc::B;
    }
    if l >= 3 {
        p.mp[MP_ISO_LEVEL] = LfoSrc::A;
        p.fb_fold_angle = LfoSrc::B;
        p.fb_saturation = LfoSrc::A;
    }
    if l >= 4 {
        p.mp[MP_KSCALE] = LfoSrc::A;
        p.mp[MP_SPEED] = LfoSrc::A;
        p.mp[MP_ZOOM] = LfoSrc::A;
        p.mp[MP_W_LATTICE] = LfoSrc::A;
        p.mp[MP_W_MOTIF] = LfoSrc::A;
        p.mp[MP_W_BAND] = LfoSrc::A;
        p.fb_rotation = LfoSrc::B;
    }
    if l >= 5 {
        p.fb_offset_x = LfoSrc::B;
        p.fb_offset_y = LfoSrc::B;
        p.fb_decay = LfoSrc::B;
    }
    if l >= 6 {
        p.fb_brightness = LfoSrc::A;
        p.fb_motion_blur = LfoSrc::B;
    }
    if l >= 7 {
        p.fb_inject = LfoSrc::A;
    }
    p
}

#[cfg(test)]
fn tour_lfo_preset() -> LfoParams {
    tour_lfo_preset_for_level(TripLevel::default())
}

fn tour_lfo_preset_fb_heavy() -> LfoParams {
    let mut mp = lfo_mp_default();
    mp[MP_SPEED] = LfoSrc::A;
    mp[MP_COLOR_SHIFT] = LfoSrc::A;
    LfoParams {
        a: LfoEngine {
            rate: 0.07,
            depth: 0.25,
            wave: LfoWave::Sine,
            phase: 0.0,
        },
        b: LfoEngine {
            rate: 0.21,
            depth: 0.18,
            wave: LfoWave::Triangle,
            phase: 0.5,
        },
        mp,
        fb_zoom: LfoSrc::A,
        fb_decay: LfoSrc::B,
        fb_color_shift: LfoSrc::A,
        fb_saturation: LfoSrc::A,
        fb_brightness: LfoSrc::B,
        fb_rotation: LfoSrc::A,
        fb_offset_x: LfoSrc::B,
        fb_offset_y: LfoSrc::B,
        fb_inject: LfoSrc::Off,
        fb_fold_angle: LfoSrc::A,
        fb_motion_blur: LfoSrc::B,
        flux: flux_default(),
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
    let local = (t / SCENE_LEN).fract();
    let drift = t * 0.13;
    let punch = (TAU * local).sin().max(0.0).powf(1.6);
    let snap = if local < 0.18 {
        (1.0 - local / 0.18).powf(2.0)
    } else {
        0.0
    };

    // Per-scene discrete picks
    let mirrors: [u32; 9] = [0, 1, 3, 5, 6, 7, 8, 9, 11];
    let blends: [u32; 7] = [0, 1, 2, 3, 5, 6, 9];
    let mi = (tour_rand(scene_f + 11.0) * mirrors.len() as f32) as usize % mirrors.len();
    let bi = (tour_rand(scene_f + 23.0) * blends.len() as f32) as usize % blends.len();

    // Continuous, energy-bounded variation. Keeps things "trippy" but stable.
    let fb_zoom = (0.965 + 0.045 * (TAU * (local * 0.35 + drift * 0.27)).sin() + 0.020 * snap)
        .clamp(0.93, 1.05);
    let fb_decay = (0.82 + 0.14 * (TAU * (local * 0.55 + drift * 0.19)).cos()).clamp(0.50, 0.98);
    let fb_inject = (0.65 + 0.35 * punch).clamp(0.0, 1.0);
    let fb_color_shift = 0.55 * (TAU * (drift * 0.17 + local * 0.41)).sin();
    let fb_saturation = (1.05 + 0.45 * (TAU * (local * 0.7 + drift * 0.11)).sin()).clamp(0.0, 2.0);
    let fb_brightness = (1.0 + 0.18 * (TAU * (local * 0.4 + drift * 0.09)).cos()).clamp(0.5, 1.5);
    let fb_rotation = 0.06 * (TAU * (drift * 0.09 + local * 0.27)).sin();
    let fb_offset_x = 0.035 * (TAU * (drift * 0.14 + local * 0.42)).sin();
    let fb_offset_y = 0.035 * (TAU * (drift * 0.11 + local * 0.36)).cos();
    let fb_fold_angle =
        2.4 * (TAU * (drift * 0.06 + local * 0.22)).sin() + (tour_rand(scene_f + 37.0) - 0.5) * 1.6;
    // Motion blur on for ~60% of scenes, lightly modulated.
    let mb_on = tour_rand(scene_f + 53.0) > 0.40;
    let fb_motion_blur = if mb_on {
        (0.18 + 0.22 * (TAU * (local * 0.30 + drift * 0.05)).sin().abs()).clamp(0.0, 0.65)
    } else {
        0.0
    };

    FieldParams {
        fb_enabled: true,
        fb_mirror: mirrors[mi],
        fb_blend_mode: blends[bi],
        fb_zoom,
        fb_decay,
        fb_inject,
        fb_color_shift,
        fb_saturation,
        fb_brightness,
        fb_rotation,
        fb_offset_x,
        fb_offset_y,
        fb_fold_angle,
        fb_motion_blur,
        ..base.clone()
    }
}

fn tour_random_field_params(t: f32, crystal_idx: usize, level: TripLevel) -> FieldParams {
    // Random tour cadence: at level 0 a "scene" stretches to ~25s and morphs slowly,
    // at level 9 it snaps every ~1.5s with chaotic bursts. Level 0 also collapses to
    // a single curated mode so the user truly sees stillness.
    let scene_len = level.lerp(25.0, 1.5);
    let scene_f = (t / scene_len).floor();
    let local = (t / scene_len).fract();
    let seed = scene_f + crystal_idx as f32 * 37.0;
    let ease = local * local * (3.0 - 2.0 * local);
    let burst_scale = level.lerp(0.0, 1.0);
    let burst = if local < 0.22 {
        (1.0 - local / 0.22).powf(2.0) * burst_scale
    } else {
        0.0
    };

    let mode_a = (tour_rand(seed + 1.0) * MODE_NAMES.len() as f32).floor() as u32;
    let mode_b = (tour_rand(seed + 2.0) * MODE_NAMES.len() as f32).floor() as u32;
    // Below level 2 we lock to a single mode per crystal so transitions are not the focus.
    let mode = if level.get() < 2 {
        TOUR_MODES[crystal_idx % TOUR_MODES.len()]
    } else if local < 0.72 {
        mode_a
    } else {
        mode_b
    };

    let morph = |slot: f32, min: f32, max: f32| -> f32 {
        let a = tour_rand(seed + slot);
        let b = tour_rand(seed + slot + 19.0);
        min + (max - min) * (a + (b - a) * ease)
    };

    let fb_on = level.fb_enabled() && tour_rand(seed + 77.0) > 0.45;
    let mirror_roll = tour_rand(seed + 88.0);
    let fb_mirror = if mirror_roll < 0.3 {
        0u32
    } else if mirror_roll < 0.5 {
        1
    } else if mirror_roll < 0.65 {
        5
    } else if mirror_roll < 0.78 {
        6
    } else if mirror_roll < 0.88 {
        7
    } else {
        9
    };
    let mut mp = default_mp();
    mp[MP_KSCALE] = (morph(3.0, 0.25, 3.8) + burst * 0.55).clamp(0.1, 5.0);
    mp[MP_SPEED] = (morph(4.0, 0.05, 1.75) + burst * 0.25).clamp(0.0, 2.0);
    mp[MP_FIELD_MIX] = morph(5.0, 0.0, 1.0).clamp(0.0, 1.0);
    mp[MP_ISO_LEVEL] = morph(6.0, 0.05, 0.96).clamp(0.0, 1.0);
    mp[MP_COLOR_SHIFT] = (morph(7.0, 0.0, 1.0) + t * 0.025).rem_euclid(1.0);
    mp[MP_ZOOM] = (morph(8.0, 0.35, 2.15) + burst * 0.25).clamp(0.2, 5.0);
    mp[MP_W_LATTICE] = morph(9.0, 0.0, 2.0).clamp(0.0, 2.0);
    mp[MP_W_MOTIF] = morph(10.0, 0.0, 2.0).clamp(0.0, 2.0);
    mp[MP_W_BAND] = morph(11.0, 0.0, 2.0).clamp(0.0, 2.0);
    FieldParams {
        mode: mode.min((MODE_NAMES.len() - 1) as u32),
        mp,
        fb_enabled: fb_on,
        fb_mirror,
        fb_zoom: morph(12.0, 0.93, 1.02),
        fb_decay: morph(13.0, 0.65, 0.97),
        fb_color_shift: morph(14.0, -0.3, 0.3),
        fb_inject: morph(15.0, 0.5, 1.0),
        fb_saturation: morph(16.0, 0.7, 1.4),
        fb_brightness: morph(17.0, 0.85, 1.15),
        fb_rotation: morph(18.0, -0.03, 0.03),
        fb_offset_x: morph(19.0, -0.02, 0.02),
        fb_offset_y: morph(20.0, -0.02, 0.02),
        fb_fold_angle: morph(21.0, -1.57, 1.57),
        fb_blend_mode: (tour_rand(seed + 99.0) * 4.0) as u32,
        fb_motion_blur: if tour_rand(seed + 101.0) > 0.55 {
            morph(22.0, 0.0, 0.55)
        } else {
            0.0
        },
        ..FieldParams::default()
    }
}

fn tour_field_params(
    t: f32,
    crystal_idx: usize,
    style: TourStyle,
    level: TripLevel,
) -> FieldParams {
    if style == TourStyle::Random {
        return tour_random_field_params(t, crystal_idx, level);
    }

    // Curated cadence: scene_len shrinks from ~25s (still) to ~1.8s (madness).
    // drift, punch, snap intensity also scale with level so visual movement is
    // gentle at the bottom and aggressive at the top. fb_enabled is gated by
    // level — feedback only kicks in at 3+.
    let scene_len = level.scene_len();
    let scene = (t / scene_len).floor() as usize;
    let local = (t / scene_len).fract();
    let drift_rate = level.lerp(0.04, 0.45);
    let drift = t * drift_rate + crystal_idx as f32 * 0.31;
    let punch_amt = level.lerp(0.0, 1.0);
    let punch = (TAU * local).sin().max(0.0).powf(1.8) * punch_amt;
    let snap_amt = level.lerp(0.0, 1.0);
    let snap = if local < 0.16 {
        (1.0 - local / 0.16).powf(2.0) * snap_amt
    } else {
        0.0
    };

    let fb_on = level.fb_enabled() && (scene % 3) != 0;
    let fb_strength = level.fb_strength();
    let fb_mirrors = [0u32, 1, 3, 5, 6, 7, 8, 9];
    let fb_mirror = fb_mirrors[scene % fb_mirrors.len()];

    // Mode-cycling rate: at level 0 the curated mode list is sampled by crystal
    // only (no scene rotation), so the user sees a single mode per crystal. As
    // level rises, the scene index folds back in to swap modes every scene_len.
    let mode_scene = if level.get() < 2 { 0 } else { scene };
    let mut mp = default_mp();
    mp[MP_KSCALE] = (1.05 + 0.72 * (TAU * drift).sin().abs() + 0.45 * snap).clamp(0.1, 5.0);
    mp[MP_SPEED] =
        (0.22 + 0.82 * punch + 0.18 * (TAU * (drift * 0.37)).sin().abs()).clamp(0.0, 2.0);
    mp[MP_FIELD_MIX] = (0.50 + 0.38 * (TAU * (local + drift * 0.11)).sin()).clamp(0.0, 1.0);
    mp[MP_ISO_LEVEL] = (0.42 + 0.36 * (TAU * (local * 0.5 + drift * 0.19)).cos()).clamp(0.0, 1.0);
    mp[MP_COLOR_SHIFT] = (drift * 0.22 + 0.08 * (TAU * local).sin()).rem_euclid(1.0);
    mp[MP_ZOOM] = (0.78 + 0.36 * (TAU * (local * 0.75)).sin().abs() + 0.22 * snap).clamp(0.2, 5.0);
    mp[MP_W_LATTICE] = (0.75 + 0.65 * (TAU * (local + 0.10)).sin().abs()).clamp(0.0, 2.0);
    mp[MP_W_MOTIF] = (0.38 + 0.92 * (TAU * (local * 0.7 + 0.35)).sin().abs()).clamp(0.0, 2.0);
    mp[MP_W_BAND] =
        (0.48 + 1.05 * (TAU * (local * 1.2 + drift * 0.07)).cos().abs()).clamp(0.0, 2.0);
    FieldParams {
        mode: TOUR_MODES[(mode_scene + crystal_idx) % TOUR_MODES.len()],
        mp,
        fb_enabled: fb_on,
        fb_mirror,
        fb_zoom: (0.96 + 0.04 * (TAU * (local * 0.3 + drift * 0.07)).sin()).clamp(0.90, 1.10),
        fb_decay: (0.82 + 0.12 * (TAU * (local * 0.5)).cos()).clamp(0.30, 0.99),
        fb_color_shift: 0.08 * fb_strength * (TAU * (drift * 0.11 + local * 0.3)).sin(),
        fb_inject: (0.4 + 0.6 * fb_strength * punch).clamp(0.0, 1.0),
        fb_saturation: (1.0 + 0.3 * (TAU * (local * 0.7 + drift * 0.13)).sin()).clamp(0.0, 2.0),
        fb_brightness: (1.0 + 0.15 * (TAU * local).cos()).clamp(0.0, 2.0),
        fb_rotation: 0.015 * fb_strength * (TAU * (drift * 0.07 + local * 0.25)).sin(),
        fb_offset_x: 0.012 * fb_strength * (TAU * (drift * 0.13 + local * 0.4)).sin(),
        fb_offset_y: 0.012 * fb_strength * (TAU * (drift * 0.09 + local * 0.35)).cos(),
        fb_fold_angle: 1.2 * fb_strength * (TAU * (drift * 0.05 + local * 0.2)).sin(),
        fb_blend_mode: (scene % 4) as u32,
        // Curated tour: motion blur cycles in 1/4 of scenes (smooth tail feel)
        fb_motion_blur: if (scene % 4) == 2 {
            (0.20 + 0.18 * fb_strength * (TAU * (local * 0.4)).sin()).clamp(0.0, 0.6)
        } else {
            0.0
        },
        ..FieldParams::default()
    }
}

/// Range-normalized modulation delta for one target (before the flux budget).
#[inline]
fn mod_norm(
    lfo: &LfoParams,
    lfo_src: LfoSrc,
    lfo_a_s: f32,
    lfo_b_s: f32,
    mic_src: MicSrc,
    mic_depth: f32,
    bands: &audio::AudioBands,
) -> f32 {
    let lfo_d = match lfo_src {
        LfoSrc::Off => 0.0,
        LfoSrc::A => lfo.a.depth * lfo_a_s,
        LfoSrc::B => lfo.b.depth * lfo_b_s,
    };
    lfo_d + mic_src.value(bands) * mic_depth
}

/// Flux budget in summed normalized units. flux 1.0 admits a total swing of
/// three full knob-spans across all routes; flux 0.33 ≈ one span.
fn flux_budget(flux: f32) -> f32 {
    flux.clamp(0.0, 1.0) * 3.0
}

/// Returns a FieldParams with LFO and mic deltas applied additively from the
/// base, plus the fraction (0..1+) of the flux budget the routes asked for.
fn apply_modulation_ex(
    fp: &FieldParams,
    lfo: &LfoParams,
    mic: &MicParams,
    bands: &audio::AudioBands,
    t: f32,
) -> (FieldParams, f32) {
    let lfo_a_s = lfo.a.sample(t);
    let lfo_b_s = lfo.b.sample(t);

    // Pass 1 — normalized deltas for every target.
    const FB_TARGETS: usize = 11;
    let mut mp_d = [0.0f32; MP_SLOTS];
    for i in 0..MP_SLOTS {
        mp_d[i] = mod_norm(
            lfo, lfo.mp[i], lfo_a_s, lfo_b_s, mic.mp[i], mic.depth, bands,
        );
    }
    let fb_ranges: [(f32, f32); FB_TARGETS] = [
        (0.90, 1.10),
        (0.30, 0.99),
        (-0.10, 0.10),
        (-0.10, 0.10),
        (-0.30, 0.30),
        (-1.00, 1.00),
        (0.00, 2.00),
        (0.00, 2.00),
        (0.00, 1.00),
        (-3.14, 3.14),
        (0.00, 0.95),
    ];
    let fb_routes = [
        lfo.fb_zoom,
        lfo.fb_decay,
        lfo.fb_offset_x,
        lfo.fb_offset_y,
        lfo.fb_rotation,
        lfo.fb_color_shift,
        lfo.fb_saturation,
        lfo.fb_brightness,
        lfo.fb_inject,
        lfo.fb_fold_angle,
        lfo.fb_motion_blur,
    ];
    let fb_mics = [
        mic.fb_zoom,
        mic.fb_decay,
        mic.fb_offset_x,
        mic.fb_offset_y,
        mic.fb_rotation,
        mic.fb_color_shift,
        mic.fb_saturation,
        mic.fb_brightness,
        mic.fb_inject,
        mic.fb_fold_angle,
        mic.fb_motion_blur,
    ];
    let mut fb_d = [0.0f32; FB_TARGETS];
    for i in 0..FB_TARGETS {
        fb_d[i] = mod_norm(
            lfo,
            fb_routes[i],
            lfo_a_s,
            lfo_b_s,
            fb_mics[i],
            mic.depth,
            bands,
        );
    }

    // Pass 2 — budget: scale everything down proportionally if over cap.
    let total: f32 = mp_d.iter().chain(fb_d.iter()).map(|d| d.abs()).sum();
    let budget = flux_budget(lfo.flux);
    let scale = if total > budget && total > 1e-6 {
        budget / total
    } else {
        1.0
    };
    let load = if budget > 1e-6 { total / budget } else { 0.0 };

    let mut mp = [0.0f32; MP_SLOTS];
    for i in 0..MP_SLOTS {
        let (lo, hi) = slot_range(fp.mode, i);
        mp[i] = (fp.mp[i] + mp_d[i] * scale * (hi - lo)).clamp(lo, hi);
    }
    let fb = |i: usize, base: f32| -> f32 {
        let (lo, hi) = fb_ranges[i];
        (base + fb_d[i] * scale * (hi - lo)).clamp(lo, hi)
    };
    let out = FieldParams {
        mode: fp.mode,
        mp,
        fb_enabled: fp.fb_enabled,
        fb_mirror: fp.fb_mirror,
        fb_blend_mode: fp.fb_blend_mode,
        fb_zoom: fb(0, fp.fb_zoom),
        fb_decay: fb(1, fp.fb_decay),
        fb_offset_x: fb(2, fp.fb_offset_x),
        fb_offset_y: fb(3, fp.fb_offset_y),
        fb_rotation: fb(4, fp.fb_rotation),
        fb_color_shift: fb(5, fp.fb_color_shift),
        fb_saturation: fb(6, fp.fb_saturation),
        fb_brightness: fb(7, fp.fb_brightness),
        fb_inject: fb(8, fp.fb_inject),
        fb_fold_angle: fb(9, fp.fb_fold_angle),
        fb_motion_blur: fb(10, fp.fb_motion_blur),
    };
    (out, load)
}

/// Returns a FieldParams with LFO and mic deltas applied additively from the base.
fn apply_modulation(
    fp: &FieldParams,
    lfo: &LfoParams,
    mic: &MicParams,
    bands: &audio::AudioBands,
    t: f32,
) -> FieldParams {
    apply_modulation_ex(fp, lfo, mic, bands, t).0
}

// ── Tour state machine ────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum TourPhase {
    Settling,
    Walking,
    Fading,
}

struct Tour {
    pub active: bool,
    pub crystal_idx: usize,
    phase: TourPhase,
    phase_timer: f32,
    pub prev_field: Option<GpuField>,
    pub prev_crystal_idx: Option<usize>,
    previous_params: Option<FieldParams>,
    last_params: Option<FieldParams>,
    param_fade_elapsed: f32,
    param_fade_duration: f32,
    param_fade_style: CrossfadeStyle,
    param_fade_cursor: usize,
}

const SETTLE_DUR: f32 = 1.2;

impl Tour {
    fn new() -> Self {
        Self {
            active: false,
            crystal_idx: 0,
            phase: TourPhase::Settling,
            phase_timer: 0.0,
            prev_field: None,
            prev_crystal_idx: None,
            previous_params: None,
            last_params: None,
            param_fade_elapsed: 0.0,
            param_fade_duration: 0.0,
            param_fade_style: CrossfadeStyle::Silk,
            param_fade_cursor: 0,
        }
    }

    /// Start an image-level transition from a stable snapshot. The renderer
    /// shades this source independently, so `mode` can change without a
    /// dispatch cut.
    fn begin_param_transition(&mut self, from: FieldParams, duration: f32) {
        self.previous_params = Some(from);
        self.param_fade_elapsed = 0.0;
        self.param_fade_duration = duration.max(0.01);
        self.param_fade_style = CrossfadeStyle::next(&mut self.param_fade_cursor);
    }

    /// Track discrete mode changes in an otherwise continuously generated tour.
    /// The first frame after a switch keeps the previous image fully visible;
    /// later frames use a smoothstep blend to reveal the destination.
    fn transition_params(
        &mut self,
        target: FieldParams,
        dt: f32,
        level: TripLevel,
    ) -> Option<(FieldParams, f32)> {
        if self.previous_params.is_none()
            && self
                .last_params
                .as_ref()
                .is_some_and(|last| last.mode != target.mode)
        {
            let duration = (level.scene_len() * 0.18).clamp(0.18, 0.75);
            self.begin_param_transition(self.last_params.clone().unwrap(), duration);
        }
        self.last_params = Some(target);

        let previous = self.previous_params.clone()?;
        self.param_fade_elapsed += dt;
        let raw = (self.param_fade_elapsed / self.param_fade_duration).clamp(0.0, 1.0);
        let mix = self.param_fade_style.apply(raw);
        if raw >= 1.0 {
            self.previous_params = None;
            None
        } else {
            Some((previous, mix))
        }
    }

    fn reset_param_transition(&mut self) {
        self.previous_params = None;
        self.last_params = None;
        self.param_fade_elapsed = 0.0;
        self.param_fade_duration = 0.0;
        self.param_fade_style = CrossfadeStyle::Silk;
        self.param_fade_cursor = 0;
        self.prev_field = None;
        self.prev_crystal_idx = None;
    }

    /// Returns true when it's time to snapshot and load the next crystal.
    /// Walk and fade durations scale with `level` — level 0 lingers on each
    /// crystal for ~28s, level 9 swaps every ~2s.
    fn tick(&mut self, dt: f32, level: TripLevel) -> bool {
        if !self.active {
            return false;
        }
        let walk_dur = level.walk_dur();
        let fade_dur = level.fade_dur();
        self.phase_timer += dt;
        match self.phase {
            TourPhase::Settling => {
                if self.phase_timer >= SETTLE_DUR {
                    self.phase = TourPhase::Walking;
                    self.phase_timer = 0.0;
                }
            }
            TourPhase::Walking => {
                if self.phase_timer >= walk_dur {
                    self.phase = TourPhase::Fading;
                    self.phase_timer = 0.0;
                    return true;
                }
            }
            TourPhase::Fading => {
                if self.phase_timer >= fade_dur {
                    self.prev_field = None;
                    self.prev_crystal_idx = None;
                    self.phase = TourPhase::Settling;
                    self.phase_timer = 0.0;
                }
            }
        }
        false
    }

    fn is_walking(&self) -> bool {
        self.phase == TourPhase::Walking
    }
}

/// Temporal personalities for scene melts. All styles preserve exact 0 → 1
/// endpoints; only the rate of reveal changes, so the renderer can keep using
/// its physically simple two-image blend.
#[derive(Clone, Copy, Debug, PartialEq)]
enum CrossfadeStyle {
    Silk,
    Rush,
    Drift,
    Ripple,
}

impl CrossfadeStyle {
    const ALL: [Self; 4] = [Self::Silk, Self::Rush, Self::Drift, Self::Ripple];

    fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            // Long soft ends, brisk middle.
            Self::Silk => {
                let s = t * t * (3.0 - 2.0 * t);
                s * s * (3.0 - 2.0 * s)
            }
            // The new scene blooms in quickly, then settles.
            Self::Rush => 1.0 - (1.0 - t).powi(3),
            // Hold the outgoing image before a deliberate melt.
            Self::Drift => t * t * t,
            // A small forward pulse gives busy scenes a living hand-off.
            Self::Ripple => {
                let s = t * t * (3.0 - 2.0 * t);
                (s + 0.10 * (std::f32::consts::PI * t).sin()).clamp(0.0, 1.0)
            }
        }
    }

    fn next(cursor: &mut usize) -> Self {
        let style = Self::ALL[*cursor % Self::ALL.len()];
        *cursor = cursor.wrapping_add(1);
        style
    }
}

/// A short, image-level bridge for any non-tour scene discontinuity. It keeps
/// the last fully rendered field uniform and, when a mode or G-field changes,
/// feeds it to the renderer as the outgoing pass. Continuous LFO/mic motion
/// remains in the incoming pass and therefore never starts a fade by itself.
const DEFAULT_SCENE_FADE_DUR: f32 = 0.48;

struct SceneTransition {
    last: Option<FieldUniform>,
    previous: Option<FieldUniform>,
    previous_field: Option<GpuField>,
    elapsed: f32,
    style: CrossfadeStyle,
    style_cursor: usize,
}

impl SceneTransition {
    fn new() -> Self {
        Self {
            last: None,
            previous: None,
            previous_field: None,
            elapsed: 0.0,
            style: CrossfadeStyle::Silk,
            style_cursor: 0,
        }
    }

    /// Begin from the last image. `previous_field` is present only when the
    /// underlying reciprocal field was regenerated (crystal load/randomize).
    fn begin(&mut self, previous_field: Option<GpuField>) {
        let Some(previous) = self.last else { return };
        self.previous = Some(previous);
        self.previous_field = previous_field;
        self.elapsed = 0.0;
        self.style = CrossfadeStyle::next(&mut self.style_cursor);
    }

    /// Store `target` and return an outgoing uniform + smooth fade mix when a
    /// transition is active. A mode change starts one automatically; callers
    /// explicitly call `begin` for same-mode G-field regeneration.
    fn track(&mut self, target: FieldUniform, dt: f32) -> Option<(FieldUniform, f32)> {
        if self.last.is_some_and(|last| last.mode != target.mode) {
            self.begin(None);
        }
        self.last = Some(target);

        let mut previous = self.previous?;
        self.elapsed += dt;
        let raw = (self.elapsed / DEFAULT_SCENE_FADE_DUR).clamp(0.0, 1.0);
        let mix = self.style.apply(raw);
        if raw >= 1.0 {
            self.previous = None;
            self.previous_field = None;
            None
        } else {
            // Keep animated outgoing modes moving during the melt rather than
            // freezing them at the frame where the scene changed.
            previous.time = target.time;
            Some((previous, mix))
        }
    }

    fn record(&mut self, target: FieldUniform) {
        self.last = Some(target);
        self.previous = None;
        self.previous_field = None;
        self.elapsed = 0.0;
    }
}

fn clone_gpu_field(field: &GpuField) -> GpuField {
    GpuField {
        gvecs: field.gvecs.clone(),
        amps: field.amps.clone(),
        phases: field.phases.clone(),
        b_mat: field.b_mat,
        count: field.count,
    }
}

// ── App ───────────────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum RenderMode {
    Atoms,
    Field,
}

/// Mutations requested by the egui UI within one frame.
#[derive(Default)]
struct UiReq {
    switch_to: Option<usize>,
    prev: bool,
    next: bool,
    screenshot: bool,
    tour_toggle: bool,
    tour_style_toggle: bool,
    trip_level_inc: bool,
    trip_level_dec: bool,
    trip_level_set: Option<u8>,
    kpath_toggle: bool,
    panel_toggle: bool,
    sequencer_panel_toggle: bool,
    render_mode: Option<RenderMode>,
    supercell: Option<usize>,
    auto_rotate_toggle: bool,
    seq_toggle: bool,
    seq_manual_toggle: bool,
    seq_manual_step: Option<i32>,   // +1 / -1
    seq_select_step: Option<usize>, // select for editor (toggle)
    seq_mute_step: Option<usize>,   // toggle mute
    seq_randomize: Option<usize>,
    seq_add_step: bool,
    seq_remove_step: bool,
    seq_preset: Option<usize>,
    seq_curve: Option<TranCurve>,
    seq_dur: Option<f32>,
    fb_reset: bool,
    fb_auto_toggle: bool,
    keymap_toggle: bool,
    mode_info_toggle: bool,
    seq_play_mode_toggle: bool,
    seq_capture: bool,
    seq_step_dur_mul: Option<(usize, f32)>,
    seq_step_curve_cycle: Option<usize>,
    seq_step_prob: Option<(usize, f32)>,
    seq_step_cond_cycle: Option<usize>,
    seq_morph: Option<f32>,
    seq_drift: Option<f32>,
    // Preset I/O
    preset_load_bundled: Option<usize>,
    preset_random: bool,
    preset_show_json: bool,
    preset_copy_json: bool,
    preset_generate_loop: bool,
    preset_apply_json: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    preset_save_path: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    preset_load_path: Option<String>,
}

struct App {
    gpu: Option<GpuState>,
    #[cfg(target_arch = "wasm32")]
    event_proxy: Option<winit::event_loop::EventLoopProxy<UserEvent>>,
    start_crystal: Crystal,
    start: Instant,
    prev_t: f32,

    dragging: bool,
    last_mouse: Option<(f64, f64)>,
    auto_rotate: bool,

    render_mode: RenderMode,
    field_params: FieldParams,
    mouse_norm: [f32; 2],
    mouse_btn_down: bool,

    kpath_active: bool,
    kpt_idx: usize,

    tour: Tour,
    tour_style: TourStyle,
    trip_level: TripLevel,
    sequencer: Sequencer,
    all_crystals: Vec<&'static CrystalDef>,
    scene_transition: SceneTransition,

    panel_open: bool,
    sequencer_panel_open: bool,
    search_str: String,
    lfo: LfoParams,
    mic_params: MicParams,
    audio: Option<audio::AudioCapture>,
    midi_in: Option<midi::MidiCapture>,
    fb_auto: bool,
    show_keymap: bool,
    show_mode_info: bool,
    mode_area: ModeArea,
    // Preset editor modal
    preset_editor_open: bool,
    preset_editor_text: String,
    preset_status: String,
    preset_random_seed: u32,
    #[cfg(not(target_arch = "wasm32"))]
    preset_path_input: String,
    /// Dev hook: capture the composed frame once `t` passes the given second
    /// mark (CRYSTALVIZ_SCREENSHOT + CRYSTALVIZ_SCREENSHOT_DELAY).
    #[cfg(not(target_arch = "wasm32"))]
    capture_after: Option<(f32, String)>,
}

enum UserEvent {
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    GpuReady(GpuState),
}

impl App {
    fn new(crystal: Crystal) -> Self {
        let all = all_crystals();
        #[cfg(not(target_arch = "wasm32"))]
        let render_mode = match std::env::var("CRYSTALVIZ_VIEW").as_deref() {
            Ok("atoms") => RenderMode::Atoms,
            _ => RenderMode::Field,
        };
        #[cfg(target_arch = "wasm32")]
        let render_mode = RenderMode::Field;
        #[cfg(not(target_arch = "wasm32"))]
        let sequencer = {
            let mut s = Sequencer::new();
            if std::env::args().any(|arg| arg == "--play") {
                s.active = true;
            }
            s
        };
        #[cfg(target_arch = "wasm32")]
        let sequencer = Sequencer::new();
        #[cfg(not(target_arch = "wasm32"))]
        let capture_after: Option<(f32, String)> = std::env::var("CRYSTALVIZ_SCREENSHOT_DELAY")
            .ok()
            .and_then(|d| d.parse::<f32>().ok())
            .zip(std::env::var("CRYSTALVIZ_SCREENSHOT").ok());
        #[cfg(not(target_arch = "wasm32"))]
        let sequencer_panel_open =
            std::env::args().any(|arg| arg == "--timeline" || arg == "--play");
        #[cfg(target_arch = "wasm32")]
        let sequencer_panel_open = false;
        Self {
            gpu: None,
            start_crystal: crystal,
            #[cfg(target_arch = "wasm32")]
            event_proxy: None,
            start: Instant::now(),
            prev_t: 0.0,
            dragging: false,
            last_mouse: None,
            auto_rotate: true,
            render_mode,
            field_params: FieldParams::default(),
            mouse_norm: [0.5, 0.5],
            mouse_btn_down: false,
            kpath_active: false,
            kpt_idx: 0,
            tour: Tour::new(),
            tour_style: TourStyle::default(),
            trip_level: TripLevel::default(),
            sequencer,
            all_crystals: all,
            scene_transition: SceneTransition::new(),
            panel_open: !sequencer_panel_open,
            sequencer_panel_open,
            search_str: String::new(),
            lfo: LfoParams::default(),
            mic_params: MicParams::default(),
            audio: audio::AudioCapture::start(),
            midi_in: midi::MidiCapture::start(),
            fb_auto: false,
            show_keymap: false,
            show_mode_info: false,
            mode_area: MODES[FieldParams::default().mode as usize].area(),
            preset_editor_open: false,
            preset_editor_text: String::new(),
            preset_status: String::new(),
            preset_random_seed: 0,
            #[cfg(not(target_arch = "wasm32"))]
            preset_path_input: String::new(),
            #[cfg(not(target_arch = "wasm32"))]
            capture_after,
        }
    }

    fn time(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }

    fn load_crystal_def(&mut self, def: &CrystalDef) {
        let Some(gpu) = self.gpu.as_mut() else { return };
        let crystal = def.to_crystal();
        let new_field = GpuField::from_crystal(&crystal, 3);
        gpu.gpu_field = new_field;
        gpu.gpu_field.seed_kpoint([0.0, 0.0, 0.0], 1.0);
        gpu.update_field();
        let sys = symmetry::detect(&crystal.lattice);
        gpu.set_crystal(crystal);
        gpu.kpath = Some(kpoints::build_kpath(sys));
        self.kpt_idx = 0;
        self.kpath_active = false;
    }

    fn switch_to(&mut self, idx: usize) {
        if let Some(gpu) = &self.gpu {
            self.scene_transition
                .begin(Some(clone_gpu_field(&gpu.gpu_field)));
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
            if let Ok(needle) = std::env::var("CRYSTALVIZ_CRYSTAL") {
                let needle = needle.to_ascii_lowercase();
                if let Some(idx) = self
                    .all_crystals
                    .iter()
                    .position(|def| def.name.to_ascii_lowercase().contains(&needle))
                {
                    self.switch_to(idx);
                } else {
                    log::warn!("Unknown CRYSTALVIZ_CRYSTAL value: {needle}");
                }
            }
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
        let egui_consumed = self
            .gpu
            .as_mut()
            .map(|g| g.handle_ui_event(&event))
            .unwrap_or(false);

        let auto_rotate = self.auto_rotate;
        let render_mode = self.render_mode;
        let t = self.time();
        let Some(gpu) = self.gpu.as_mut() else { return };

        #[cfg(not(target_arch = "wasm32"))]
        if let Some((deadline, _)) = self.capture_after {
            if t >= deadline {
                let (_, path) = self.capture_after.take().unwrap();
                gpu.request_screenshot(path);
            }
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{KeyCode, PhysicalKey};
                if event.state != ElementState::Pressed {
                    return;
                }
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                    PhysicalKey::Code(KeyCode::KeyM) => {
                        self.field_params.mode =
                            (self.field_params.mode + 1) % MODE_NAMES.len() as u32;
                    }
                    PhysicalKey::Code(KeyCode::KeyK) => {
                        if let Some(kp) = &gpu.kpath {
                            let n = kp.n_points();
                            self.kpt_idx = (self.kpt_idx + 1) % n;
                            let coords = kp.snap_to(self.kpt_idx);
                            self.scene_transition
                                .begin(Some(clone_gpu_field(&gpu.gpu_field)));
                            gpu.gpu_field.seed_kpoint(coords, 1.0);
                            gpu.update_field();
                        }
                        self.kpath_active = false;
                    }
                    PhysicalKey::Code(KeyCode::KeyP) => {
                        self.kpath_active = !self.kpath_active;
                        if self.kpath_active {
                            if let Some(kp) = &mut gpu.kpath {
                                kp.reset();
                            }
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        if render_mode == RenderMode::Field {
                            self.scene_transition
                                .begin(Some(clone_gpu_field(&gpu.gpu_field)));
                            gpu.gpu_field.randomize();
                            gpu.update_field();
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyT) => {
                        self.tour.active = !self.tour.active;
                        if self.tour.active {
                            self.render_mode = RenderMode::Field;
                            self.kpath_active = true;
                            self.lfo = tour_lfo_preset_for_level(self.trip_level);
                            if let Some(kp) = &mut gpu.kpath {
                                kp.reset();
                            }
                        }
                    }
                    // Minus / Equals: step the trip level down/up
                    PhysicalKey::Code(KeyCode::Minus) => {
                        self.trip_level = self.trip_level.dec();
                        if self.tour.active {
                            self.lfo = tour_lfo_preset_for_level(self.trip_level);
                        }
                    }
                    PhysicalKey::Code(KeyCode::Equal) => {
                        self.trip_level = self.trip_level.inc();
                        if self.tour.active {
                            self.lfo = tour_lfo_preset_for_level(self.trip_level);
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
                        self.field_params.mp[MP_COLOR_SHIFT] =
                            (self.field_params.mp[MP_COLOR_SHIFT] + 0.05) % 1.0;
                    }
                    PhysicalKey::Code(KeyCode::BracketLeft) => {
                        self.field_params.mp[MP_COLOR_SHIFT] =
                            (self.field_params.mp[MP_COLOR_SHIFT] - 0.05).rem_euclid(1.0);
                    }
                    _ => {}
                }
            }

            WindowEvent::Resized(size) => gpu.resize(size),

            WindowEvent::MouseInput { state, button, .. } if !egui_consumed => {
                if button == MouseButton::Left {
                    let pressed = state == ElementState::Pressed;
                    self.dragging = pressed;
                    self.mouse_btn_down = pressed;
                    if !pressed {
                        self.last_mouse = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let sz = gpu.size;
                self.mouse_norm = [
                    (position.x as f32 / sz.width as f32).clamp(0.0, 1.0),
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
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.05,
                };
                match render_mode {
                    RenderMode::Atoms => gpu.zoom(d),
                    RenderMode::Field => {
                        self.field_params.mp[MP_ZOOM] =
                            (self.field_params.mp[MP_ZOOM] + d * 0.15).clamp(0.2, 5.0);
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
                    let should_advance = self.tour.tick(dt, self.trip_level);
                    if should_advance {
                        let total = self.all_crystals.len();
                        let old_idx = self.tour.crystal_idx;
                        let next_idx = (old_idx + 1) % total;
                        // Snapshot both the outgoing G-field and its currently
                        // shaded parameter state. Rendering them in separate
                        // passes avoids the old dispatch cut between modes.
                        let old_params = self.tour.last_params.clone().unwrap_or_else(|| {
                            tour_field_params(t, old_idx, self.tour_style, self.trip_level)
                        });
                        self.tour
                            .begin_param_transition(old_params, self.trip_level.fade_dur());
                        self.tour.prev_crystal_idx = Some(old_idx);
                        self.tour.prev_field = Some(GpuField {
                            gvecs: gpu.gpu_field.gvecs.clone(),
                            amps: gpu.gpu_field.amps.clone(),
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
                        if let Some(kp) = &mut gpu.kpath {
                            kp.reset();
                        }
                        self.kpt_idx = 0;
                        self.kpath_active = true;
                    }

                    if self.tour.is_walking() && self.kpath_active {
                        if let Some(kp) = &mut gpu.kpath {
                            let (k, _) = kp.tick(dt);
                            gpu.gpu_field.seed_kpoint(k, 0.4);
                        }
                    }
                    gpu.update_field();
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
                let cur_render_mode = self.render_mode;
                let cur_tour_active = self.tour.active;
                let cur_tour_style = self.tour_style;
                let cur_kpath_active = self.kpath_active;
                let cur_panel_open = self.panel_open;
                let cur_sequencer_panel_open = self.sequencer_panel_open;
                let cur_fb_auto = self.fb_auto;
                let cur_show_keymap = self.show_keymap;
                let cur_show_mode_info = self.show_mode_info;
                let mut cur_mode_area = self.mode_area;
                #[cfg(not(target_arch = "wasm32"))]
                let cur_surface_size = gpu.size;
                let cur_supercell = gpu.supercell;
                let cur_crystal_idx = self.tour.crystal_idx;
                let cur_trip_level = self.trip_level;
                let (tour_fp, tour_transition) = if self.tour.active {
                    let target = tour_field_params(
                        t,
                        self.tour.crystal_idx,
                        self.tour_style,
                        self.trip_level,
                    );
                    let transition =
                        self.tour
                            .transition_params(target.clone(), dt, self.trip_level);
                    (Some(target), transition)
                } else {
                    (None, None)
                };
                let seq_fp = if self.sequencer.active {
                    Some(self.sequencer.current_params())
                } else {
                    None
                };
                // Sequencer overrides tour which overrides manual field_params
                let cur_mode_idx = seq_fp
                    .as_ref()
                    .or(tour_fp.as_ref())
                    .map(|fp| fp.mode)
                    .unwrap_or(self.field_params.mode) as usize;
                let cur_mode_name = MODE_NAMES[cur_mode_idx];
                let live_mode_area = MODES[cur_mode_idx].area();
                if live_mode_area != cur_mode_area {
                    cur_mode_area = live_mode_area;
                }
                let cur_crystal_name = self.all_crystals[cur_crystal_idx].name;
                let cur_sys_name = self.all_crystals[cur_crystal_idx].system.name();
                let cur_kpt_label: String = gpu
                    .kpath
                    .as_ref()
                    .map(|kp| kp.label_at(self.kpt_idx % kp.n_points()).to_owned())
                    .unwrap_or_default();
                // Snapshot sequencer state for UI
                let cur_seq_active = self.sequencer.active;
                let cur_seq_manual = self.sequencer.manual;
                let cur_seq_selected = self.sequencer.selected;
                let cur_seq_steps: Vec<(bool, bool, bool, u32, (u8, u8), f32)> = self
                    .sequencer
                    .steps
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        (
                            i == self.sequencer.cur,
                            s.muted,
                            Some(i) == self.sequencer.selected,
                            s.params.mode,
                            s.cond,
                            s.prob,
                        )
                    })
                    .collect();
                let cur_seq_dur = self.sequencer.step_dur;
                let cur_seq_curve = self.sequencer.curve;
                let cur_seq_morph = self.sequencer.morph;
                let cur_seq_drift = self.sequencer.drift;
                let cur_seq_play_mode = self.sequencer.play_mode;
                let cur_seq_eff_dur = self.sequencer.effective_step_dur();
                let cur_seq_progress =
                    (self.sequencer.step_timer / cur_seq_eff_dur).clamp(0.0, 1.0);
                // Editor: mutable local that closure can modify; written back in mutations
                let mut seq_selected_edit: Option<(usize, FieldParams)> = self
                    .sequencer
                    .selected
                    .and_then(|i| self.sequencer.steps.get(i).map(|s| (i, s.params.clone())));
                // Per-step extras for the selected step editor: (dur_mul, curve_override, prob)
                let seq_selected_extras: Option<(f32, Option<TranCurve>, f32, (u8, u8))> = self
                    .sequencer
                    .selected
                    .and_then(|i| self.sequencer.steps.get(i))
                    .map(|s| (s.dur_mul, s.curve_override, s.prob, s.cond));
                // Snapshot field_params, lfo, mic, and current audio bands.
                let mut fp = seq_fp
                    .or(tour_fp)
                    .unwrap_or_else(|| self.field_params.clone());
                // FB AUTO: override every fb_* field with an evolving auto-pilot pattern.
                // Composes on top of the manual/tour/sequencer source, before LFO/mic.
                if self.fb_auto {
                    fp = fb_auto_params(t, &fp);
                }

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
                    } else {
                        midi::CcSnapshot::default()
                    }
                } else {
                    midi::CcSnapshot::default()
                };
                // CC: directly drive params (overrides tour/seq for the set CCs).
                midi::apply_midi_cc(&mut fp, &midi_cc_snap);

                let mut lfo = self.lfo.clone();
                let mut mic = self.mic_params.clone();
                let cur_bands = self
                    .audio
                    .as_ref()
                    .and_then(|a| a.bands.lock().ok().map(|b| b.clone()))
                    .unwrap_or_default();
                let mic_active = self.audio.is_some();

                // Apply LFO + mic modulation to get effective values for this frame.
                let (fp_eff, cur_flux_load) = apply_modulation_ex(&fp, &lfo, &mic, &cur_bands, t);

                // Build each pass's independent uniform. Tour transitions use
                // the same current modulation inputs for both images, while the
                // mode-specific base params and crystal colour stay distinct.
                let cdef = self.all_crystals[cur_crystal_idx];
                let make_field_uniform =
                    |effective: &FieldParams, crystal_idx: usize| FieldUniform {
                        mp: renderer::pack_mp(&effective.mp),
                        time: t,
                        mode: effective.mode,
                        num_g: gpu.gpu_field.count as u32,
                        aspect: gpu.size.width as f32 / gpu.size.height.max(1) as f32,
                        crystal_color: {
                            let cdef = self.all_crystals[crystal_idx];
                            [cdef.color[0], cdef.color[1], cdef.color[2], 0.0]
                        },
                        mouse: self.mouse_norm,
                        mouse_down: if self.mouse_btn_down { 1.0 } else { 0.0 },
                        fb_enabled: if effective.fb_enabled { 1 } else { 0 },
                        fb_mirror: effective.fb_mirror,
                        fb_zoom: effective.fb_zoom,
                        fb_offset_x: effective.fb_offset_x,
                        fb_offset_y: effective.fb_offset_y,
                        fb_rotation: effective.fb_rotation,
                        fb_decay: effective.fb_decay,
                        fb_color_shift: effective.fb_color_shift,
                        fb_inject: effective.fb_inject,
                        fb_fold_angle: effective.fb_fold_angle,
                        fb_saturation: effective.fb_saturation,
                        fb_brightness: effective.fb_brightness,
                        fb_blend_mode: effective.fb_blend_mode,
                        fb_motion_blur: effective.fb_motion_blur,
                        _pad: [0.0; 3],
                    };
                let field_params_uniform = make_field_uniform(&fp_eff, cur_crystal_idx);
                let tour_field_transition = if self.sequencer.active {
                    None
                } else {
                    tour_transition.map(|(mut previous, mix)| {
                        if self.fb_auto {
                            previous = fb_auto_params(t, &previous);
                        }
                        midi::apply_midi_cc(&mut previous, &midi_cc_snap);
                        let (previous_eff, _) =
                            apply_modulation_ex(&previous, &lfo, &mic, &cur_bands, t);
                        let previous_crystal_idx =
                            self.tour.prev_crystal_idx.unwrap_or(cur_crystal_idx);
                        (make_field_uniform(&previous_eff, previous_crystal_idx), mix)
                    })
                };
                let uses_tour_transition = tour_field_transition.is_some();
                let default_field_transition = if self.tour.active {
                    // Tour owns its longer crystal/mode transition timing.
                    self.scene_transition.record(field_params_uniform);
                    None
                } else if render_mode == RenderMode::Field {
                    self.scene_transition.track(field_params_uniform, dt)
                } else {
                    None
                };
                let field_transition = tour_field_transition.or(default_field_transition);

                // Snapshot search string
                let mut search = self.search_str.clone();

                // Preset editor snapshot
                let cur_preset_editor_open = self.preset_editor_open;
                let mut preset_editor_text = self.preset_editor_text.clone();
                let cur_preset_status = self.preset_status.clone();
                #[cfg(not(target_arch = "wasm32"))]
                let mut preset_path_input = self.preset_path_input.clone();

                // Crystal list (static data, no borrows)
                let groups = all_groups();
                let all_defs = all_crystals();

                // Accumulator for mutations
                let mut req = UiReq::default();
                let req_panel_open = cur_panel_open;

                // The closure now captures &mut fp.* freely — no conflict with match below.
                let ui_fn = |ctx: &egui::Context| {
                    // ── Command bar ───────────────────────────────────
                    egui::TopBottomPanel::top("command_bar")
                        .exact_height(42.0)
                        .show(ctx, |ui| {
                            ui.add_space(5.0);
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new("CRYSTAL / DIVE")
                                        .strong()
                                        .size(13.0)
                                        .color(egui::Color32::from_rgb(137, 225, 215)),
                                );
                                ui.separator();

                                let view_label = match cur_render_mode {
                                    RenderMode::Atoms => "ATOMS",
                                    RenderMode::Field => "FIELD",
                                };
                                if ui
                                    .button(view_label)
                                    .on_hover_text("Switch between structure and field views")
                                    .clicked()
                                {
                                    req.render_mode = Some(match cur_render_mode {
                                        RenderMode::Atoms => RenderMode::Field,
                                        RenderMode::Field => RenderMode::Atoms,
                                    });
                                }

                                let inspector_text =
                                    egui::RichText::new("INSPECTOR").color(if req_panel_open {
                                        egui::Color32::from_rgb(137, 225, 215)
                                    } else {
                                        egui::Color32::from_gray(145)
                                    });
                                if ui.add(egui::Button::new(inspector_text)).clicked() {
                                    req.panel_toggle = true;
                                }

                                let timeline_text = egui::RichText::new("TIMELINE").color(
                                    if cur_sequencer_panel_open {
                                        egui::Color32::from_rgb(247, 197, 106)
                                    } else {
                                        egui::Color32::from_gray(145)
                                    },
                                );
                                if ui
                                    .add(egui::Button::new(timeline_text))
                                    .on_hover_text("Open the sequencer workspace")
                                    .clicked()
                                {
                                    req.sequencer_panel_toggle = true;
                                }

                                ui.separator();
                                let tour_color = if cur_tour_active {
                                    egui::Color32::from_rgb(137, 225, 215)
                                } else {
                                    egui::Color32::from_gray(180)
                                };
                                if ui
                                    .add(egui::Button::new(
                                        egui::RichText::new(if cur_tour_active {
                                            "STOP TOUR"
                                        } else {
                                            "TOUR"
                                        })
                                        .color(tour_color),
                                    ))
                                    .clicked()
                                {
                                    req.tour_toggle = true;
                                }
                                if ui
                                    .small_button(cur_tour_style.label())
                                    .on_hover_text("Tour direction")
                                    .clicked()
                                {
                                    req.tour_style_toggle = true;
                                }

                                if ui.small_button("−").clicked() {
                                    req.trip_level_dec = true;
                                }
                                ui.label(
                                    egui::RichText::new(format!("FLOW {}", cur_trip_level.label()))
                                        .monospace()
                                        .color(egui::Color32::from_rgb(247, 197, 106)),
                                );
                                if ui.small_button("+").clicked() {
                                    req.trip_level_inc = true;
                                }

                                if ui
                                    .small_button(if cur_kpath_active {
                                        "K PATH ON"
                                    } else {
                                        "K PATH"
                                    })
                                    .clicked()
                                {
                                    req.kpath_toggle = true;
                                }
                                if ui
                                    .small_button(if cur_fb_auto { "AUTO ON" } else { "AUTO" })
                                    .clicked()
                                {
                                    req.fb_auto_toggle = true;
                                }

                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui
                                            .small_button("?")
                                            .on_hover_text("Keyboard shortcuts")
                                            .clicked()
                                        {
                                            req.keymap_toggle = true;
                                        }
                                        if ui
                                            .small_button("CAPTURE")
                                            .on_hover_text("Save the complete frame, including UI")
                                            .clicked()
                                        {
                                            req.screenshot = true;
                                        }
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "{}  /  {}  /  {}",
                                                cur_crystal_name, cur_sys_name, cur_mode_name
                                            ))
                                            .small()
                                            .color(egui::Color32::from_gray(155)),
                                        );
                                    },
                                );
                            });
                        });

                    // Side panel
                    if req_panel_open {
                        egui::SidePanel::left("inspector")
                            .default_width(326.0)
                            .width_range(300.0..=380.0)
                            .resizable(true)
                            .show(ctx, |ui| {
                                ui.add_space(8.0);
                                ui.label(
                                    egui::RichText::new("LIVE INSPECTOR")
                                        .small()
                                        .strong()
                                        .color(egui::Color32::from_rgb(137, 225, 215)),
                                );
                                ui.add_space(4.0);
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
                                egui::ScrollArea::vertical()
                                    .id_salt("inspector_scroll")
                                    .auto_shrink([false, false])
                                    .show(ui, |ui| {

                                // Shared slider macro for both Field and Feedback sections.
                                // Defined here in the side-panel scope so sibling CollapsingHeader
                                // closures (Field, Feedback) can both reach it.
                                macro_rules! sld {
                                    ($ui:expr, $label:expr, $val:expr, $eff:expr,
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
                                egui::CollapsingHeader::new(egui::RichText::new("CRYSTAL LIBRARY")
                                    .strong().color(egui::Color32::from_gray(175)))
                                    .id_salt("sec_library").default_open(false)
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

                                if cur_render_mode == RenderMode::Field {
                                // Field params (collapsible, open by default)
                                egui::CollapsingHeader::new(egui::RichText::new("FIELD")
                                    .strong().color(egui::Color32::from_rgb(137, 225, 215)))
                                    .id_salt("sec_field").default_open(true)
                                    .show(ui, |ui| {
                                        let mode_before = fp.mode;
                                        ui.horizontal_wrapped(|ui| {
                                            for area in ModeArea::ALL {
                                                let selected = cur_mode_area == area;
                                                let color = if selected {
                                                    egui::Color32::from_rgb(255, 220, 120)
                                                } else {
                                                    egui::Color32::from_rgb(150, 155, 180)
                                                };
                                                if ui.selectable_label(
                                                    selected,
                                                    egui::RichText::new(area.label()).small().color(color),
                                                ).on_hover_text(area.title()).clicked() {
                                                    cur_mode_area = area;
                                                    if MODES[fp.mode as usize].area() != area {
                                                        if let Some(mode) = MODES.iter().find(|m| m.area() == area) {
                                                            fp.mode = mode.idx as u32;
                                                        }
                                                    }
                                                }
                                            }
                                        });
                                        ui.add_space(2.0);
                                        // Compact mode picker: ComboBox + arrow buttons
                                        ui.horizontal(|ui| {
                                            ui.label(egui::RichText::new("mode").small().monospace());
                                            egui::ComboBox::from_id_salt("mode_combo")
                                                .selected_text(MODE_NAMES[fp.mode as usize])
                                                .width(160.0)
                                                .show_ui(ui, |ui| {
                                                    ui.label(egui::RichText::new(cur_mode_area.title())
                                                        .small().color(egui::Color32::from_gray(150)));
                                                    ui.separator();
                                                    for mode in MODES.iter().filter(|m| m.area() == cur_mode_area) {
                                                        ui.selectable_value(&mut fp.mode, mode.idx as u32, mode.name);
                                                    }
                                                });
                                            let area_modes: Vec<u32> = MODES.iter()
                                                .filter(|m| m.area() == cur_mode_area)
                                                .map(|m| m.idx as u32)
                                                .collect();
                                            if ui.small_button("◀").clicked() && !area_modes.is_empty() {
                                                let pos = area_modes.iter()
                                                    .position(|&mode| mode == fp.mode)
                                                    .unwrap_or(0);
                                                fp.mode = area_modes[(pos + area_modes.len() - 1) % area_modes.len()];
                                            }
                                            if ui.small_button("▶").clicked() && !area_modes.is_empty() {
                                                let pos = area_modes.iter()
                                                    .position(|&mode| mode == fp.mode)
                                                    .unwrap_or(0);
                                                fp.mode = area_modes[(pos + 1) % area_modes.len()];
                                            }
                                            let info_btn = egui::Button::new(
                                                egui::RichText::new("ⓘ").color(
                                                    if cur_show_mode_info {
                                                        egui::Color32::from_rgb(255, 220, 120)
                                                    } else {
                                                        egui::Color32::from_rgb(180, 180, 200)
                                                    }
                                                )
                                            ).small();
                                            if ui.add(info_btn)
                                                .on_hover_text("Mode info: equations + slider semantics")
                                                .clicked()
                                            {
                                                req.mode_info_toggle = true;
                                            }
                                        });
                                        // A manual mode switch loads the new mode's slot defaults.
                                        if fp.mode != mode_before {
                                            apply_mode_defaults(&mut fp);
                                        }
                                        ui.add_space(2.0);
                                // Per-mode tailored sliders: one named slider per slot the
                                // active mode declares (see modes::mode_params).
                                for p in mode_params(fp.mode) {
                                    let s = p.slot;
                                    sld!(ui, format!("{:<9}", p.name),
                                        &mut fp.mp[s], fp_eff.mp[s],
                                        &mut lfo.mp[s], &mut mic.mp[s],
                                        p.min, p.max);
                                }
                                    }); // end Field collapsible

                                // ── FEEDBACK / SELF-SIMILARITY ────────────────────────────
                                egui::CollapsingHeader::new(egui::RichText::new("FEEDBACK")
                                    .strong().color(egui::Color32::from_rgb(247, 197, 106)))
                                    .id_salt("sec_feedback").default_open(false)
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
                                egui::CollapsingHeader::new(egui::RichText::new("MODULATION")
                                    .strong().color(egui::Color32::from_rgb(159, 186, 220)))
                                    .id_salt("sec_modulation").default_open(false)
                                    .show(ui, |ui| {
                                        // Compact two-column LFO row.
                                        // Each column shows: wave button, rate / depth / phase sliders.
                                        let lfo_row = |ui: &mut egui::Ui, name: &str, color: egui::Color32, eng: &mut LfoEngine| {
                                            ui.vertical(|ui| {
                                                ui.horizontal(|ui| {
                                                    ui.label(egui::RichText::new(name).strong().color(color));
                                                    if ui.small_button(eng.wave.label()).clicked() {
                                                        eng.wave = eng.wave.next();
                                                    }
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.label(egui::RichText::new("r").small().monospace());
                                                    ui.add_sized(
                                                        [ui.available_width(), 18.0],
                                                        egui::Slider::new(&mut eng.rate, 0.01..=4.0)
                                                            .show_value(true).suffix(" Hz"),
                                                    );
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.label(egui::RichText::new("d").small().monospace());
                                                    ui.add_sized(
                                                        [ui.available_width(), 18.0],
                                                        egui::Slider::new(&mut eng.depth, 0.0..=1.0).show_value(false),
                                                    );
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.label(egui::RichText::new("φ").small().monospace());
                                                    ui.add_sized(
                                                        [ui.available_width(), 18.0],
                                                        egui::Slider::new(&mut eng.phase, 0.0..=1.0).show_value(false),
                                                    );
                                                });
                                            });
                                        };
                                        lfo_row(ui, "LFO A", egui::Color32::from_rgb(80, 220, 120), &mut lfo.a);
                                        ui.add_space(4.0);
                                        lfo_row(ui, "LFO B", egui::Color32::from_rgb(120, 180, 255), &mut lfo.b);

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
                                                let band = |ui: &mut egui::Ui, lbl: &str, c: egui::Color32, v: f32| {
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
                                } else {
                                    egui::CollapsingHeader::new(
                                        egui::RichText::new("STRUCTURE")
                                            .strong()
                                            .color(egui::Color32::from_rgb(137, 225, 215)),
                                    )
                                    .id_salt("sec_structure")
                                    .default_open(true)
                                    .show(ui, |ui| {
                                        egui::Grid::new("structure_metadata")
                                            .num_columns(2)
                                            .spacing(egui::vec2(16.0, 6.0))
                                            .show(ui, |ui| {
                                                ui.label(egui::RichText::new("SPACE GROUP").small()
                                                    .color(egui::Color32::from_gray(125)));
                                                ui.monospace(format!(
                                                    "{}  ·  No. {}",
                                                    cdef.space_group, cdef.sp_number
                                                ));
                                                ui.end_row();
                                                ui.label(egui::RichText::new("POINT GROUP").small()
                                                    .color(egui::Color32::from_gray(125)));
                                                ui.monospace(cdef.point_group);
                                                ui.end_row();
                                                ui.label(egui::RichText::new("CENTERING").small()
                                                    .color(egui::Color32::from_gray(125)));
                                                ui.monospace(cdef.lattice_type);
                                                ui.end_row();
                                            });

                                        ui.add_space(8.0);
                                        ui.label(
                                            egui::RichText::new("SUPERCELL")
                                                .small()
                                                .color(egui::Color32::from_gray(125)),
                                        );
                                        ui.horizontal(|ui| {
                                            for n in 1..=3 {
                                                let selected = cur_supercell == n;
                                                if ui.selectable_label(
                                                    selected,
                                                    format!("{n} × {n} × {n}"),
                                                ).clicked() {
                                                    req.supercell = Some(n);
                                                }
                                            }
                                        });

                                        ui.add_space(6.0);
                                        let rotation_label =
                                            if auto_rotate { "AUTO ROTATE ON" } else { "AUTO ROTATE" };
                                        if ui.button(rotation_label).clicked() {
                                            req.auto_rotate_toggle = true;
                                        }
                                        ui.label(
                                            egui::RichText::new(
                                                "Drag to orbit  ·  wheel to zoom  ·  Space pauses rotation",
                                            )
                                            .small()
                                            .color(egui::Color32::from_gray(120)),
                                        );
                                    });
                                }

                                ui.separator();

                                // Crystal nav (transport: ATOMS/FIELD, TOUR, ?-help are all in toolbar)
                                ui.horizontal(|ui| {
                                    if ui.button("◀ PREV").clicked() { req.prev = true; }
                                    if ui.button("NEXT ▶").clicked() { req.next = true; }
                                });

                                ui.separator();

                                    });
                            });
                    }

                    // ── Consolidated bottom sequencer panel ─────────────
                    // One source of truth: transport, grid, presets, and (when a
                    // step is selected) an inline editor — all in a single dock.
                    if cur_sequencer_panel_open {
                        egui::TopBottomPanel::bottom("sequencer_panel")
                        .resizable(true)
                        .default_height(if seq_selected_edit.is_some() { 340.0 } else { 132.0 })
                        .height_range(112.0..=390.0)
                        .show(ctx, |ui| {
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new("SEQUENCE")
                                        .strong()
                                        .color(egui::Color32::from_rgb(247, 197, 106)),
                                );
                                let state = if cur_seq_active { "PLAYING" } else { "READY" };
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{state}  ·  {} STEPS  ·  {:.1}s",
                                        cur_seq_steps.len(),
                                        cur_seq_dur
                                    ))
                                    .small()
                                    .color(egui::Color32::from_gray(145)),
                                );
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button("CLOSE").clicked() {
                                        req.sequencer_panel_toggle = true;
                                    }
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{}  /  {}",
                                            cur_crystal_name, cur_mode_name
                                        ))
                                        .small()
                                        .color(egui::Color32::from_gray(125)),
                                    );
                                });
                            });
                            ui.separator();

                            // ── Inline step editor — only when a step is selected
                            if let Some((edit_idx, ref mut ep)) = seq_selected_edit {
                                let is_muted = cur_seq_steps.get(edit_idx)
                                    .map(|&(_, m, _, _, _, _)| m).unwrap_or(false);
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgba_unmultiplied(20, 16, 36, 200))
                                    .rounding(4.0)
                                    .inner_margin(egui::vec2(8.0, 6.0))
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.label(egui::RichText::new(format!("STEP {:02}", edit_idx + 1))
                                                .strong().color(egui::Color32::from_rgb(247, 197, 106)));
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
                                            if ui.small_button("DONE").on_hover_text("Close step editor").clicked() {
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
                                        // Two columns of this step's mode-tailored params.
                                        let eparams = mode_params(ep.mode);
                                        let half = eparams.len().div_ceil(2);
                                        ui.horizontal(|ui| {
                                            ui.vertical(|ui| {
                                                ui.set_min_width(220.0);
                                                for p in &eparams[..half] {
                                                    esl!(ui, format!("{:<9}", p.name), &mut ep.mp[p.slot], p.min, p.max);
                                                }
                                            });
                                            ui.separator();
                                            ui.vertical(|ui| {
                                                ui.set_min_width(220.0);
                                                for p in &eparams[half..] {
                                                    esl!(ui, format!("{:<9}", p.name), &mut ep.mp[p.slot], p.min, p.max);
                                                }
                                            });
                                        });

                                        // Per-step timing/curve/prob overrides
                                        if let Some((mut dm, curve_ov, mut pr, cond)) = seq_selected_extras {
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
                                                ui.label(egui::RichText::new("cond").small().monospace());
                                                let cond_active = cond != (1, 1);
                                                let cond_col = if cond_active {
                                                    egui::Color32::from_rgb(247, 197, 106)
                                                } else { egui::Color32::from_gray(150) };
                                                if ui.add(egui::Button::new(
                                                    egui::RichText::new(cond_label(cond)).color(cond_col).monospace()
                                                ).small()).on_hover_text(
                                                    "Cycle condition: play on pass k of every n visits (Elektron trig condition)"
                                                ).clicked() {
                                                    req.seq_step_cond_cycle = Some(edit_idx);
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

                                // Morph window: how much of the step is transition vs hold.
                                let mut morph = cur_seq_morph;
                                ui.label(egui::RichText::new("morph").small());
                                if ui.add(egui::Slider::new(&mut morph, 0.05_f32..=1.0_f32)
                                    .show_value(false))
                                    .on_hover_text("Transition window: 0.1 = hard cut then hold, 1.0 = continuous glide")
                                    .changed() {
                                    req.seq_morph = Some(morph);
                                }

                                // Drift lane: bounded always-alive micro-motion.
                                let mut drift = cur_seq_drift;
                                let drift_col = if drift > 0.001 {
                                    egui::Color32::from_rgb(137, 225, 215)
                                } else { egui::Color32::from_gray(140) };
                                ui.label(egui::RichText::new("drift").small().color(drift_col));
                                if ui.add(egui::Slider::new(&mut drift, 0.0_f32..=1.0_f32)
                                    .show_value(false))
                                    .on_hover_text("Micro-motion lane: keeps holds alive (hue crawl, breathing). Bounded — never chaotic.")
                                    .changed() {
                                    req.seq_drift = Some(drift);
                                }

                                // Flux budget: total modulation cap with live load meter.
                                let flux_load = cur_flux_load.min(1.5);
                                let flux_col = if flux_load >= 1.0 {
                                    egui::Color32::from_rgb(255, 140, 90)  // saturated: routes being scaled
                                } else {
                                    egui::Color32::from_rgb(159, 186, 220)
                                };
                                ui.label(egui::RichText::new("flux").small().color(flux_col));
                                ui.add(egui::Slider::new(&mut lfo.flux, 0.0_f32..=1.0_f32).show_value(false))
                                    .on_hover_text("Modulation budget: caps the summed LFO+mic swing across every route. \
                                                    Orange = routes asking for more than the cap; all get scaled proportionally.");
                                ui.add(egui::ProgressBar::new(flux_load / 1.5).desired_width(42.0)
                                    .fill(flux_col))
                                    .on_hover_text(format!("Modulation load: {:.0} % of budget", cur_flux_load * 100.0));
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
                                    egui::RichText::new("CAPTURE STEP")
                                        .color(egui::Color32::from_rgb(247, 197, 106))
                                )).on_hover_text("Append a new step from the current live state").clicked() {
                                    req.seq_capture = true;
                                }
                                ui.separator();

                                // Bundled-preset dropdown (full Preset, applies LFO + level too)
                                egui::ComboBox::from_id_salt("preset_combo")
                                    .selected_text("PRESET")
                                    .width(110.0)
                                    .show_ui(ui, |ui| {
                                        for (i, (name, _)) in preset::BUNDLED_PRESETS.iter().enumerate() {
                                            if ui.selectable_label(false, *name).clicked() {
                                                req.preset_load_bundled = Some(i);
                                            }
                                        }
                                    });

                                if ui.add(egui::Button::new(
                                    egui::RichText::new("RANDOMIZE")
                                        .color(egui::Color32::from_rgb(206, 153, 216))
                                )).on_hover_text("Roll a fresh random preset (steps + LFO + trip level)").clicked() {
                                    req.preset_random = true;
                                }
                                if ui.small_button("{ } JSON")
                                    .on_hover_text("Open the preset JSON editor").clicked() {
                                    req.preset_show_json = true;
                                }
                                if ui.add(egui::Button::new(
                                    egui::RichText::new("LOOP")
                                        .color(egui::Color32::from_rgb(140, 220, 255))
                                )).on_hover_text("Render a seamless ping-pong loop from the current preset").clicked() {
                                    req.preset_generate_loop = true;
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
                                        let mut cells: Vec<(usize, egui::Rect, bool, bool, bool, u32, (u8, u8), f32)> = Vec::new();
                                        for (i, &(is_cur, muted, is_sel, mode, cond, prob)) in cur_seq_steps.iter().enumerate() {
                                            let (rect, resp) = ui.allocate_exact_size(
                                                egui::vec2(cell_w, cell_h),
                                                egui::Sense::click(),
                                            );
                                            if resp.double_clicked() {
                                                req.seq_randomize = Some(i);
                                            } else if resp.clicked() {
                                                req.seq_select_step = Some(i);
                                            }
                                            cells.push((i, rect, is_cur, muted, is_sel, mode, cond, prob));
                                            ui.add_space(3.0);
                                        }
                                        cells
                                    }).inner;

                                    let painter = ui.painter();
                                    for &(_i, rect, is_cur, muted, is_sel, mode, cond, prob) in &cells {
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
                                        // Elektron-style trig-condition badge, top-right.
                                        if cond != (1, 1) {
                                            painter.text(
                                                rect.right_top() + egui::vec2(-2.0, 1.0),
                                                egui::Align2::RIGHT_TOP,
                                                cond_label(cond),
                                                egui::FontId::monospace(7.0),
                                                egui::Color32::from_rgb(247, 197, 106),
                                            );
                                        }
                                        // Probability pip, bottom-left, when < 100 %.
                                        if prob < 0.999 {
                                            painter.circle_filled(
                                                rect.left_bottom() + egui::vec2(4.0, -4.0),
                                                2.0,
                                                egui::Color32::from_rgb(206, 153, 216),
                                            );
                                        }
                                    }
                                });
                            ui.add_space(2.0);
                        });
                    }

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
                                        row(ui, "− / =", "trip level down / up (0 still → 9 madness)");
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
                        if !open {
                            req.keymap_toggle = true;
                        }
                    }

                    // ── Mode info overlay ─────────────────────────────
                    if cur_show_mode_info {
                        let mut open = true;
                        let info: &ModeInfo = &MODES[cur_mode_idx];
                        egui::Window::new(format!("Mode {} — {}", info.idx, info.name))
                            .id(egui::Id::new("mode_info_overlay"))
                            .open(&mut open)
                            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-12.0, 56.0))
                            .resizable(true)
                            .collapsible(false)
                            .default_width(380.0)
                            .show(ctx, |ui| {
                                ui.label(
                                    egui::RichText::new(info.tagline)
                                        .italics()
                                        .color(egui::Color32::from_rgb(200, 210, 230)),
                                );
                                ui.add_space(6.0);
                                ui.separator();
                                ui.add_space(4.0);
                                ui.label(
                                    egui::RichText::new("equations")
                                        .small()
                                        .color(egui::Color32::from_gray(140)),
                                );
                                ui.add_space(2.0);
                                for eq in info.equations.iter() {
                                    ui.label(
                                        egui::RichText::new(*eq)
                                            .monospace()
                                            .size(13.5)
                                            .color(egui::Color32::from_rgb(230, 230, 200)),
                                    );
                                }
                                ui.add_space(6.0);
                                ui.separator();
                                ui.add_space(4.0);
                                ui.label(
                                    egui::RichText::new("notes")
                                        .small()
                                        .color(egui::Color32::from_gray(140)),
                                );
                                ui.add_space(2.0);
                                ui.label(
                                    egui::RichText::new(info.notes)
                                        .color(egui::Color32::from_rgb(220, 220, 230)),
                                );
                            });
                        if !open {
                            req.mode_info_toggle = true;
                        }
                    }

                    // ── Preset JSON editor modal ─────────────────────
                    if cur_preset_editor_open {
                        let mut open = true;
                        egui::Window::new("Preset (JSON)")
                            .id(egui::Id::new("preset_editor"))
                            .open(&mut open)
                            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                            .resizable(true)
                            .collapsible(false)
                            .default_size(egui::vec2(540.0, 480.0))
                            .show(ctx, |ui| {
                                ui.horizontal(|ui| {
                                    if ui.button("📋 Copy to clipboard").clicked() {
                                        ui.ctx().copy_text(preset_editor_text.clone());
                                    }
                                    if ui.button("📥 Paste from clipboard").clicked() {
                                        ui.ctx()
                                            .send_viewport_cmd(egui::ViewportCommand::RequestPaste);
                                    }
                                    if ui
                                        .add(egui::Button::new(
                                            egui::RichText::new("Apply →")
                                                .color(egui::Color32::from_rgb(140, 255, 140)),
                                        ))
                                        .on_hover_text(
                                            "Parse the JSON above and load it into the sequencer",
                                        )
                                        .clicked()
                                    {
                                        req.preset_apply_json = Some(preset_editor_text.clone());
                                    }
                                });
                                ui.add_space(4.0);

                                #[cfg(not(target_arch = "wasm32"))]
                                ui.horizontal(|ui| {
                                    ui.label("Path:");
                                    ui.add(
                                        egui::TextEdit::singleline(&mut preset_path_input)
                                            .hint_text("presets/my_preset.preset.json")
                                            .desired_width(280.0),
                                    );
                                    if ui.button("💾 Save").clicked()
                                        && !preset_path_input.is_empty()
                                    {
                                        req.preset_save_path = Some(preset_path_input.clone());
                                    }
                                    if ui.button("📁 Load").clicked()
                                        && !preset_path_input.is_empty()
                                    {
                                        req.preset_load_path = Some(preset_path_input.clone());
                                    }
                                });

                                if !cur_preset_status.is_empty() {
                                    ui.label(
                                        egui::RichText::new(&cur_preset_status)
                                            .small()
                                            .color(egui::Color32::from_rgb(180, 220, 255)),
                                    );
                                }

                                ui.add_space(4.0);
                                // Paste events land into the editor text directly so
                                // Ctrl-V in the text area works the way users expect.
                                ui.ctx().input(|i| {
                                    for ev in &i.events {
                                        if let egui::Event::Paste(s) = ev {
                                            preset_editor_text = s.clone();
                                        }
                                    }
                                });
                                egui::ScrollArea::vertical()
                                    .auto_shrink([false, false])
                                    .show(ui, |ui| {
                                        ui.add_sized(
                                            ui.available_size(),
                                            egui::TextEdit::multiline(&mut preset_editor_text)
                                                .code_editor()
                                                .desired_rows(20),
                                        );
                                    });
                            });
                        if !open {
                            req.preset_show_json = true;
                        }
                    }
                };

                // ── Render ────────────────────────────────────────────
                match cur_render_mode {
                    RenderMode::Atoms => {
                        if auto_rotate {
                            gpu.orbit(0.4, 0.0);
                        }
                        match gpu.render(t, ui_fn) {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                let sz = gpu.size;
                                gpu.resize(sz);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("OOM");
                                event_loop.exit();
                            }
                            Err(e) => log::warn!("render: {e:?}"),
                        }
                    }
                    RenderMode::Field => {
                        // The transition path shades both modes independently
                        // before blending, so tour scene changes do not chop.
                        let previous_field = if uses_tour_transition {
                            self.tour.prev_field.as_ref()
                        } else {
                            self.scene_transition.previous_field.as_ref()
                        };
                        let result = if let Some((previous, mix)) = field_transition {
                            gpu.render_field_transition(
                                &previous,
                                previous_field,
                                &field_params_uniform,
                                mix,
                                ui_fn,
                            )
                        } else {
                            gpu.render_field(&field_params_uniform, ui_fn)
                        };
                        match result {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                let sz = gpu.size;
                                gpu.resize(sz);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("OOM");
                                event_loop.exit();
                            }
                            Err(e) => log::warn!("field: {e:?}"),
                        }
                    }
                }

                // ── Apply UI mutations (closure is dropped, borrows released) ──
                self.field_params = fp;
                self.lfo = lfo;
                self.mic_params = mic;
                self.search_str = search;
                self.mode_area = cur_mode_area;
                self.preset_editor_text = preset_editor_text;
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.preset_path_input = preset_path_input;
                }

                // ── MIDI Note triggers — route them into the same UiReq path so
                //    they behave exactly like clicking the corresponding button.
                for n in midi_notes {
                    match n {
                        midi::note::FB_AUTO_TOGGLE => req.fb_auto_toggle = true,
                        midi::note::TOUR_TOGGLE => req.tour_toggle = true,
                        midi::note::SEQ_TOGGLE => req.seq_toggle = true,
                        midi::note::SEQ_PREV => req.seq_manual_step = Some(-1),
                        midi::note::SEQ_NEXT => req.seq_manual_step = Some(1),
                        midi::note::SEQ_CAPTURE => req.seq_capture = true,
                        _ => {}
                    }
                }

                if req.panel_toggle {
                    self.panel_open = !cur_panel_open;
                    if self.panel_open {
                        self.sequencer_panel_open = false;
                    }
                }
                if req.sequencer_panel_toggle {
                    self.sequencer_panel_open = !cur_sequencer_panel_open;
                    if self.sequencer_panel_open {
                        self.panel_open = false;
                    }
                }
                if req.keymap_toggle {
                    self.show_keymap = !cur_show_keymap;
                }
                if req.mode_info_toggle {
                    self.show_mode_info = !cur_show_mode_info;
                }
                if req.fb_auto_toggle {
                    self.fb_auto = !cur_fb_auto;
                    // Turning auto on triggers a clean feedback restart and ensures fb_enabled
                    if self.fb_auto {
                        self.field_params.fb_enabled = true;
                        req.fb_reset = true;
                    }
                }

                if let Some(rm) = req.render_mode {
                    self.render_mode = rm;
                }
                if req.auto_rotate_toggle {
                    self.auto_rotate = !auto_rotate;
                }
                if let Some(n) = req.supercell {
                    if let Some(gpu2) = &mut self.gpu {
                        gpu2.set_supercell(n);
                    }
                }
                if req.kpath_toggle {
                    self.kpath_active = !cur_kpath_active;
                    if self.kpath_active {
                        if let Some(gpu2) = &mut self.gpu {
                            if let Some(kp) = &mut gpu2.kpath {
                                kp.reset();
                            }
                        }
                    }
                }
                if req.tour_toggle {
                    self.tour.active = !cur_tour_active;
                    self.tour.reset_param_transition();
                    if self.tour.active {
                        self.render_mode = RenderMode::Field;
                        self.kpath_active = true;
                        self.lfo = tour_lfo_preset_for_level(self.trip_level);
                        if let Some(gpu2) = &mut self.gpu {
                            if let Some(kp) = &mut gpu2.kpath {
                                kp.reset();
                            }
                        }
                    }
                }
                if req.tour_style_toggle {
                    self.tour_style = self.tour_style.next();
                    self.lfo = match self.tour_style {
                        TourStyle::Curated => tour_lfo_preset_for_level(self.trip_level),
                        TourStyle::Random => tour_lfo_preset_fb_heavy(),
                    };
                }
                // Trip-level requests: re-apply the curated preset so LFOs follow
                // the new level immediately. (Random style keeps fb_heavy preset.)
                let trip_changed =
                    req.trip_level_inc || req.trip_level_dec || req.trip_level_set.is_some();
                if req.trip_level_dec {
                    self.trip_level = self.trip_level.dec();
                }
                if req.trip_level_inc {
                    self.trip_level = self.trip_level.inc();
                }
                if let Some(v) = req.trip_level_set {
                    self.trip_level = TripLevel::new(v);
                }
                if trip_changed && self.tour.active && self.tour_style == TourStyle::Curated {
                    self.lfo = tour_lfo_preset_for_level(self.trip_level);
                }
                if req.seq_toggle {
                    self.sequencer.active = !cur_seq_active;
                    if self.sequencer.active {
                        self.render_mode = RenderMode::Field;
                    }
                }
                if req.seq_manual_toggle {
                    self.sequencer.manual = !cur_seq_manual;
                    // Enabling manual mode must activate the sequencer so seq_fp is Some(...)
                    if self.sequencer.manual {
                        self.sequencer.active = true;
                        self.render_mode = RenderMode::Field;
                    }
                }
                if let Some(dir) = req.seq_manual_step {
                    self.sequencer.manual_step(dir);
                }
                if let Some(i) = req.seq_select_step {
                    self.sequencer.selected = if self.sequencer.selected == Some(i) {
                        None
                    } else {
                        Some(i)
                    };
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
                // ── Preset I/O ────────────────────────────────────────
                if let Some(i) = req.preset_load_bundled {
                    if let Some((name, json)) = preset::BUNDLED_PRESETS.get(i) {
                        match preset::Preset::from_json(json) {
                            Ok(p) => {
                                p.apply(&mut self.sequencer, &mut self.lfo, &mut self.trip_level);
                                self.sequencer.active = true;
                                self.render_mode = RenderMode::Field;
                                self.preset_status = format!("Loaded preset '{name}'");
                            }
                            Err(e) => self.preset_status = format!("bundled '{name}' failed: {e}"),
                        }
                    }
                }
                if req.preset_random {
                    // Seed off the wall clock so each click is a fresh roll;
                    // remember the seed so the user can paste it / reproduce.
                    let seed = (t * 1_000_000.0) as u32
                        ^ self
                            .preset_random_seed
                            .wrapping_mul(1664525)
                            .wrapping_add(1013904223);
                    self.preset_random_seed = seed;
                    let p = preset::random_preset(seed);
                    self.preset_status = format!("Rolled '{}' (seed 0x{seed:08x})", p.name);
                    p.apply(&mut self.sequencer, &mut self.lfo, &mut self.trip_level);
                    self.sequencer.active = true;
                    self.render_mode = RenderMode::Field;
                }
                if req.preset_show_json {
                    self.preset_editor_open = !self.preset_editor_open;
                    if self.preset_editor_open {
                        // Pre-fill with the current state so users see what they have.
                        let cur = preset::Preset::from_state(
                            "edited",
                            &self.sequencer,
                            &self.lfo,
                            self.trip_level,
                        );
                        self.preset_editor_text = cur.to_pretty_json();
                    }
                }
                if req.preset_copy_json {
                    let cur = preset::Preset::from_state(
                        "current",
                        &self.sequencer,
                        &self.lfo,
                        self.trip_level,
                    );
                    self.preset_editor_text = cur.to_pretty_json();
                    self.preset_status =
                        "Preset JSON ready in the editor — Ctrl-C to copy".to_string();
                    self.preset_editor_open = true;
                }
                if req.preset_generate_loop {
                    #[cfg(target_arch = "wasm32")]
                    {
                        self.preset_status = "Loop rendering needs the desktop/local binary so it can write PNG frames.".to_string();
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        match spawn_loop_render(
                            &self.sequencer,
                            &self.lfo,
                            self.trip_level,
                            cur_surface_size.width,
                            cur_surface_size.height,
                        ) {
                            Ok(path) => {
                                self.preset_status =
                                    format!("Loop render started: {}", path.display());
                            }
                            Err(e) => {
                                self.preset_status = format!("Loop render failed: {e}");
                            }
                        }
                    }
                }
                if let Some(json) = req.preset_apply_json {
                    match preset::Preset::from_json(&json) {
                        Ok(p) => {
                            let nm = p.name.clone();
                            p.apply(&mut self.sequencer, &mut self.lfo, &mut self.trip_level);
                            self.sequencer.active = true;
                            self.render_mode = RenderMode::Field;
                            self.preset_status = format!("Applied '{nm}'");
                        }
                        Err(e) => self.preset_status = format!("Apply failed: {e}"),
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if let Some(path) = req.preset_save_path {
                        let cur = preset::Preset::from_state(
                            std::path::Path::new(&path)
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("preset"),
                            &self.sequencer,
                            &self.lfo,
                            self.trip_level,
                        );
                        self.preset_status = match std::fs::write(&path, cur.to_pretty_json()) {
                            Ok(_) => format!("Saved → {path}"),
                            Err(e) => format!("Save failed: {e}"),
                        };
                    }
                    if let Some(path) = req.preset_load_path {
                        self.preset_status = match std::fs::read_to_string(&path)
                            .map_err(|e| e.to_string())
                            .and_then(|s| preset::Preset::from_json(&s))
                        {
                            Ok(p) => {
                                let nm = p.name.clone();
                                p.apply(&mut self.sequencer, &mut self.lfo, &mut self.trip_level);
                                self.sequencer.active = true;
                                self.render_mode = RenderMode::Field;
                                format!("Loaded '{nm}' from {path}")
                            }
                            Err(e) => format!("Load {path} failed: {e}"),
                        };
                    }
                }
                if let Some(c) = req.seq_curve {
                    self.sequencer.curve = c;
                }
                if let Some(d) = req.seq_dur {
                    self.sequencer.step_dur = d;
                }
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
                            None => Some(TranCurve::Linear),
                            Some(c) if c == TranCurve::Over => None,
                            Some(c) => Some(c.next()),
                        };
                    }
                }
                if let Some((i, v)) = req.seq_step_prob {
                    if let Some(s) = self.sequencer.steps.get_mut(i) {
                        s.prob = v.clamp(0.0, 1.0);
                    }
                }
                if let Some(i) = req.seq_step_cond_cycle {
                    if let Some(s) = self.sequencer.steps.get_mut(i) {
                        s.cond = cond_next(s.cond);
                        s.visits = 0;
                    }
                }
                if let Some(v) = req.seq_morph {
                    self.sequencer.morph = v.clamp(0.05, 1.0);
                }
                if let Some(v) = req.seq_drift {
                    self.sequencer.drift = v.clamp(0.0, 1.0);
                }
                if req.fb_reset {
                    if let Some(gpu) = &mut self.gpu {
                        gpu.fb_clear = true;
                    }
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
                        gpu2.request_screenshot(format!("crystal-viz-{}.png", chrono_stamp()));
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
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}")
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_loop_render(
    seq: &Sequencer,
    lfo: &LfoParams,
    level: TripLevel,
    width: u32,
    height: u32,
) -> Result<std::path::PathBuf, String> {
    let stamp = chrono_stamp();
    let root = std::path::PathBuf::from("loops").join(format!("loop-{stamp}"));
    let frames = root.join("frames");
    std::fs::create_dir_all(&frames).map_err(|e| format!("create {}: {e}", frames.display()))?;

    let full_duration = 8.0_f32;
    let half_duration = full_duration * 0.5;
    let mut p = preset::Preset::from_state("current-loop", seq, lfo, level);
    p.play_mode = SeqPlayMode::Forward;
    p.step_dur = half_duration / seq.total_dur_mul();
    for step in &mut p.steps {
        step.muted = false;
        step.prob = 1.0;
    }

    let preset_path = root.join("current-loop.preset.json");
    std::fs::write(&preset_path, p.to_pretty_json())
        .map_err(|e| format!("write {}: {e}", preset_path.display()))?;

    let exe = std::env::current_exe().map_err(|e| format!("current exe: {e}"))?;
    let even = |v: u32, min: u32| (v.max(min) & !1).max(min);
    let res = format!("{}x{}", even(width, 320), even(height, 180));
    let child = std::process::Command::new(exe)
        .arg("--render")
        .arg(&preset_path)
        .arg("--out")
        .arg(&frames)
        .arg("--res")
        .arg(res)
        .arg("--fps")
        .arg("60")
        .arg("--duration")
        .arg(format!("{full_duration:.3}"))
        .arg("--pingpong-loop")
        .arg("--encode")
        .spawn()
        .map_err(|e| format!("spawn renderer: {e}"))?;

    log::info!(
        "Started loop renderer pid {} -> {}",
        child.id(),
        frames.display()
    );
    Ok(root)
}

fn default_crystal() -> Crystal {
    all_crystals()[0].to_crystal()
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();

    // ── Headless clip render mode ────────────────────────────────────────
    //
    //   crystal-viz --render preset.json --out frames/ --res 1280x720 --fps 60
    //               --duration 10 [--start 0] [--poscar POSCAR]
    //
    // Reads the preset, advances the sequencer at every 1/fps tick, and writes
    // one PNG per frame. Prints an ffmpeg encode command at the end.
    if args.iter().any(|a| a == "--list-crystals") {
        for c in crystals::all_crystals() {
            println!("{}", c.name);
        }
        std::process::exit(0);
    }
    if args.iter().any(|a| a == "--render") {
        match run_render_cli(&args) {
            Ok(_) => std::process::exit(0),
            Err(e) => {
                eprintln!("render: {e}");
                std::process::exit(1);
            }
        }
    }

    let crystal = if let Some(path) = args.get(1).filter(|a| !a.starts_with("--")) {
        poscar::Crystal::from_file(path).unwrap_or_else(|e| {
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
    println!("  T        — auto-tour on/off");
    println!("  M        — cycle render mode  P — k-path walk  K — next k-pt");
    println!("  [ / ]    — colour shift       scroll — zoom");
    println!("  − / =    — trip level 0..9");
    println!("  --render preset.json --out frames/ --res WxH --fps 60 --duration 10");
    println!("           (offline clip render — produces a PNG sequence)");

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new(crystal)).unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
fn run_render_cli(args: &[String]) -> Result<(), String> {
    use crystal_viz::bench;

    // Tiny ad-hoc CLI parser — we only have a handful of flags.
    fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
        args.windows(2)
            .find_map(|w| (w[0] == name).then_some(w[1].as_str()))
    }
    let preset_path =
        flag(args, "--render").ok_or_else(|| "missing --render <preset.json>".to_string())?;
    let out_dir = flag(args, "--out").ok_or_else(|| "missing --out <dir>".to_string())?;
    let res = flag(args, "--res").unwrap_or("1280x720");
    let fps: u32 = flag(args, "--fps")
        .unwrap_or("60")
        .parse()
        .map_err(|e| format!("--fps not a number: {e}"))?;
    let start: f32 = flag(args, "--start")
        .unwrap_or("0.0")
        .parse()
        .map_err(|e| format!("--start not a number: {e}"))?;
    let poscar_path = flag(args, "--poscar");
    let crystal_name = flag(args, "--crystal");
    let bpm = flag(args, "--bpm")
        .map(|s| s.parse::<f32>())
        .transpose()
        .map_err(|e| format!("--bpm not a number: {e}"))?;
    let bars = flag(args, "--bars")
        .map(|s| s.parse::<f32>())
        .transpose()
        .map_err(|e| format!("--bars not a number: {e}"))?;
    let pingpong_loop = args.iter().any(|a| a == "--pingpong-loop");
    let encode = args.iter().any(|a| a == "--encode");
    // --bpm + --bars together override --duration and trigger loop-snap.
    let duration: f32 = match (bpm, bars) {
        (Some(b), Some(n)) if b > 0.0 && n > 0.0 => n * 4.0 * 60.0 / b,
        _ => flag(args, "--duration")
            .unwrap_or("10.0")
            .parse()
            .map_err(|e| format!("--duration not a number: {e}"))?,
    };
    let loop_snap = bpm.is_some() && bars.is_some();
    let final_frames = (duration * fps as f32).round().max(2.0) as u32;
    let render_duration = if pingpong_loop {
        let forward_frames = final_frames / 2 + 1;
        forward_frames as f32 / fps as f32
    } else {
        duration
    };

    let (width, height): (u32, u32) = res
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        .ok_or_else(|| format!("--res must be WxH, got '{res}'"))?;

    let crystal = if let Some(name) = crystal_name {
        let needle = name.to_lowercase();
        let pick = crystals::all_crystals()
            .into_iter()
            .find(|c| c.name.to_lowercase().contains(&needle))
            .ok_or_else(|| format!("--crystal '{name}' not found (try --list-crystals)"))?;
        eprintln!("Crystal: '{}'", pick.name);
        pick.to_crystal()
    } else if let Some(p) = poscar_path {
        poscar::Crystal::from_file(p).map_err(|e| format!("poscar {p}: {e}"))?
    } else {
        default_crystal()
    };

    let json =
        std::fs::read_to_string(preset_path).map_err(|e| format!("read {preset_path}: {e}"))?;
    let p = preset::Preset::from_json(&json)?;

    eprintln!(
        "Preset: '{}' — {} steps, trip {}",
        p.name,
        p.steps.len(),
        p.trip_level.get()
    );

    // Build a sequencer + LFO state from the preset and tick it as we render.
    let mut seq = Sequencer::new();
    let mut lfo = LfoParams::default();
    let mut level = TripLevel::default();
    p.apply(&mut seq, &mut lfo, &mut level);
    seq.active = true;

    // Seamless-loop snap: when --bpm/--bars set, scale step_dur so one full
    // pass through the step list = duration, and round LFO rates to integer
    // cycles per loop so iso/color modulation wraps cleanly at t=duration.
    if loop_snap {
        let total_dur_mul: f32 = seq.steps.iter().map(|s| s.dur_mul.max(1e-6)).sum();
        if total_dur_mul > 0.0 {
            seq.step_dur = duration / total_dur_mul;
        }
        let snap = |r: f32| -> f32 {
            let cycles = (r * duration).round().max(0.0);
            cycles / duration
        };
        lfo.a.rate = snap(lfo.a.rate);
        lfo.b.rate = snap(lfo.b.rate);
        eprintln!(
            "Loop snap: dur {:.3}s, step_dur {:.3}s, LFO A {:.4}Hz ({} cyc), B {:.4}Hz ({} cyc)",
            duration,
            seq.step_dur,
            lfo.a.rate,
            (lfo.a.rate * duration).round() as i32,
            lfo.b.rate,
            (lfo.b.rate * duration).round() as i32,
        );
    }

    // The closure tracks its own `prev_t` so we can advance the sequencer by
    // the exact 1/fps delta even though render_clip just hands us absolute t.
    let mut prev_t: f32 = start;
    let mic = MicParams::default();
    let bands = audio::AudioBands::default();
    let aspect = width as f32 / height.max(1) as f32;
    let accent = crystal_color(&crystal);

    let eval = move |t: f32| -> FieldUniform {
        let dt = (t - prev_t).max(0.0);
        prev_t = t;
        seq.tick(dt);
        let fp_seq = seq.current_params();
        let fp = apply_modulation(&fp_seq, &lfo, &mic, &bands, t);
        // Headless render skips the feedback pass, so blank fb_enabled regardless.
        let mut fp = fp;
        fp.fb_enabled = false;
        field_params_to_uniform(&fp, t, aspect, &accent)
    };

    let opts = bench::ClipOpts {
        width,
        height,
        fps,
        duration: render_duration,
        start,
        out_dir: std::path::PathBuf::from(out_dir),
    };
    let out_path = opts.out_dir.clone();
    bench::render_clip(opts, &crystal, eval)?;
    if pingpong_loop {
        mirror_pingpong_frames(&out_path, final_frames)?;
        eprintln!(
            "Ping-pong loop completed: {} frames. The final frame mirrors back to frame_000000 for a clean loop.",
            final_frames,
        );
    }
    if encode {
        encode_frame_sequence(&out_path, fps)?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn mirror_pingpong_frames(out_dir: &std::path::Path, final_frames: u32) -> Result<(), String> {
    let forward_frames = final_frames / 2 + 1;
    if forward_frames < 2 {
        return Err("ping-pong loop needs at least two forward frames".to_string());
    }
    for dst in forward_frames..final_frames {
        let mirror_offset = dst - forward_frames + 1;
        let src = forward_frames.saturating_sub(1 + mirror_offset);
        let src_path = out_dir.join(format!("frame_{src:06}.png"));
        let dst_path = out_dir.join(format!("frame_{dst:06}.png"));
        std::fs::copy(&src_path, &dst_path).map_err(|e| {
            format!(
                "mirror {} -> {}: {e}",
                src_path.display(),
                dst_path.display()
            )
        })?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn encode_frame_sequence(out_dir: &std::path::Path, fps: u32) -> Result<(), String> {
    let output = out_dir.parent().unwrap_or(out_dir).join("loop.mp4");
    let input = out_dir.join("frame_%06d.png");
    let status = std::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-loglevel")
        .arg("error")
        .arg("-framerate")
        .arg(fps.to_string())
        .arg("-i")
        .arg(&input)
        .arg("-c:v")
        .arg("libx264")
        .arg("-vf")
        .arg("scale=trunc(iw/2)*2:trunc(ih/2)*2")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-crf")
        .arg("18")
        .arg("-preset")
        .arg("slow")
        .arg("-movflags")
        .arg("+faststart")
        .arg(&output)
        .status();

    match status {
        Ok(s) if s.success() => {
            eprintln!("Encoded loop video: {}", output.display());
            write_loop_preview(&output)?;
            Ok(())
        }
        Ok(s) => {
            eprintln!(
                "ffmpeg exited with status {s}; PNG frames remain at {}",
                out_dir.display()
            );
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "ffmpeg not found; PNG frames remain at {}",
                out_dir.display()
            );
            Ok(())
        }
        Err(e) => Err(format!("ffmpeg: {e}")),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn write_loop_preview(video_path: &std::path::Path) -> Result<(), String> {
    let dir = video_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let file = video_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("bad video path: {}", video_path.display()))?;
    let html = format!(
        r#"<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>CrystalDive Loop Preview</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #05050a; }}
  body {{ display: grid; place-items: center; }}
  video {{ max-width: 100vw; max-height: 100vh; background: #000; }}
</style>
<video src="{file}" autoplay loop muted controls playsinline></video>
"#
    );
    let preview = dir.join("loop-preview.html");
    std::fs::write(&preview, html).map_err(|e| format!("write {}: {e}", preview.display()))?;
    eprintln!("Loop preview page: {}", preview.display());
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn crystal_color(_c: &Crystal) -> [f32; 4] {
    // Same accent the UI uses — for the headless path we keep it neutral.
    [0.50, 0.70, 1.00, 0.0]
}

#[cfg(not(target_arch = "wasm32"))]
fn field_params_to_uniform(
    fp: &FieldParams,
    t: f32,
    aspect: f32,
    crystal_color: &[f32; 4],
) -> FieldUniform {
    FieldUniform {
        mp: renderer::pack_mp(&fp.mp),
        time: t,
        mode: fp.mode,
        num_g: 0, // num_g is populated by GpuField at upload — keep 0 here; the shader uses textureLoad with bounds-check loops
        aspect,
        crystal_color: *crystal_color,
        mouse: [0.5, 0.5],
        mouse_down: 0.0,
        fb_enabled: 0,
        fb_mirror: 0,
        fb_zoom: fp.fb_zoom,
        fb_offset_x: fp.fb_offset_x,
        fb_offset_y: fp.fb_offset_y,
        fb_rotation: fp.fb_rotation,
        fb_decay: fp.fb_decay,
        fb_color_shift: fp.fb_color_shift,
        fb_inject: fp.fb_inject,
        fb_fold_angle: fp.fb_fold_angle,
        fb_saturation: fp.fb_saturation,
        fb_brightness: fp.fb_brightness,
        fb_blend_mode: fp.fb_blend_mode,
        fb_motion_blur: fp.fb_motion_blur,
        _pad: [0.0; 3],
    }
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
        let steps = (0..n)
            .map(|i| {
                sp(
                    i as u32 % MODE_NAMES.len() as u32,
                    1.0 + i as f32 * 0.1,
                    0.5,
                    i as f32 / n as f32,
                    0.5,
                    i as f32 / n as f32,
                    1.0,
                    1.0,
                    0.5,
                    0.5,
                )
            })
            .collect::<Vec<_>>();
        let from_params = steps[0].params.clone();
        Sequencer {
            active: true,
            manual: true,
            cur: 0,
            step_dur: 2.0,
            step_timer: 2.0, // fully arrived at step 0
            curve: TranCurve::Linear,
            selected: None,
            play_mode: SeqPlayMode::Forward,
            pp_dir: 1,
            rng_seed: 0x9E3779B9,
            steps,
            from_params,
            // Legacy semantics for the existing invariants: wall-to-wall
            // glide, no drift lane. Dedicated tests cover morph/drift.
            morph: 1.0,
            drift: 0.0,
            clock: 0.0,
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
        assert_eq!(
            seq.cur, 0,
            "going back from 2, step 1 muted, should land on 0"
        );
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
        assert!(
            (seq.step_timer - 2.0).abs() < 1e-6,
            "should clamp at step_dur"
        );
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
        assert!(
            (p.mp[crate::MP_KSCALE] - seq.from_params.mp[crate::MP_KSCALE]).abs() < 1e-5,
            "at t=0 current_params should equal from_params"
        );
    }

    // Digitakt sync: when the sequencer advances to step N (manually or via tick),
    // current_params().mode must equal steps[N].mode immediately at t=0, so the
    // rendered field matches the highlight in the seq dock with no half-step lag.
    #[test]
    fn current_params_mode_snaps_to_cur_on_advance() {
        let mut seq = make_seq(4);
        // Manually step from 0 → 1 (cur=1, timer=0).
        seq.manual_step(1);
        let p_just_after = seq.current_params();
        assert_eq!(
            p_just_after.mode, seq.steps[1].params.mode,
            "after manual_step, mode must match new cur immediately (t=0)"
        );
        // Mid-step (t=0.4): same mode.
        seq.step_timer = seq.step_dur * 0.4;
        let p_mid = seq.current_params();
        assert_eq!(
            p_mid.mode, seq.steps[1].params.mode,
            "mid-step mode must equal cur's mode (no half-step lag)"
        );
    }

    // Same property when auto-tick crosses a step boundary.
    #[test]
    fn current_params_mode_snaps_on_auto_tick_boundary() {
        let mut seq = make_seq(4);
        seq.manual = false;
        seq.cur = 0;
        seq.step_timer = 0.0;
        seq.from_params = seq.steps[0].params.clone();
        // Tick just past the step boundary so cur advances to 1.
        seq.tick(seq.step_dur + 0.001);
        assert_eq!(seq.cur, 1);
        let p = seq.current_params();
        assert_eq!(
            p.mode, seq.steps[1].params.mode,
            "after auto-tick crossing boundary, mode must equal new cur's mode"
        );
    }

    // current_params at step_timer=step_dur returns the destination (t=1)
    #[test]
    fn current_params_at_t1_is_destination() {
        let mut seq = make_seq(4);
        seq.manual_step(1); // cur=1, from=steps[0], timer=0
        seq.step_timer = seq.step_dur; // t=1
        let p = seq.current_params();
        let target = &seq.steps[seq.cur].params;
        assert!(
            (p.mp[crate::MP_KSCALE] - target.mp[crate::MP_KSCALE]).abs() < 1e-5,
            "at t=1 current_params should equal target step params"
        );
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
        assert_eq!(
            seq.selected, None,
            "removing selected step clears selection"
        );
    }

    // All-muted: advance_next should not infinite-loop
    #[test]
    fn advance_next_all_muted_no_infinite_loop() {
        let mut seq = make_seq(3);
        for s in seq.steps.iter_mut() {
            s.muted = true;
        }
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
        assert!(
            (p.mp[crate::MP_KSCALE] - seq.from_params.mp[crate::MP_KSCALE]).abs() < 1e-5,
            "snap override at t=0.5 should output from_params"
        );
    }

    #[test]
    fn no_curve_override_uses_global() {
        let mut seq = make_seq(2);
        seq.curve = TranCurve::Linear;
        seq.manual_step(1);
        seq.step_timer = seq.step_dur * 0.5;
        let p = seq.current_params();
        let expected = (seq.from_params.mp[crate::MP_KSCALE]
            + seq.steps[seq.cur].params.mp[crate::MP_KSCALE])
            * 0.5;
        assert!(
            (p.mp[crate::MP_KSCALE] - expected).abs() < 1e-5,
            "linear at t=0.5 should be exact midpoint"
        );
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
        seq.advance_next();
        assert_eq!(seq.cur, 1);
        seq.advance_next();
        assert_eq!(seq.cur, 2);
        seq.advance_next();
        assert_eq!(seq.cur, 1, "should reverse off the end");
        seq.advance_next();
        assert_eq!(seq.cur, 0);
        seq.advance_next();
        assert_eq!(seq.cur, 1, "should reverse off the start");
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
        assert!(
            count >= 6,
            "Random should hit at least 6/8 steps, hit {count}"
        );
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
            if seq.cur == 1 {
                plays += 1;
            }
        }
        // generous bounds — just rules out 0% and 100%
        assert!(
            plays > 30 && plays < 170,
            "prob=0.5 should fire roughly half the time (got {plays}/200)"
        );
    }

    #[test]
    fn all_prob_zero_does_not_hang() {
        let mut seq = make_seq(3);
        for s in seq.steps.iter_mut() {
            s.prob = 0.0;
        }
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
        assert!(
            seq.effective_step_dur() >= 0.05,
            "effective_step_dur must clamp above 0.05 to avoid /0"
        );
    }

    // ── CAPTURE ───────────────────────────────────────────────────────────
    #[test]
    fn capture_appends_step() {
        let mut seq = make_seq(3);
        let mut p = FieldParams::default();
        p.mp[crate::MP_KSCALE] = 7.7;
        let idx = seq.capture(p).expect("capture should fit");
        assert_eq!(idx, 3);
        assert_eq!(seq.steps.len(), 4);
        assert!((seq.steps[3].params.mp[crate::MP_KSCALE] - 7.7).abs() < 1e-5);
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

    // ── Morph window ─────────────────────────────────────────────────────
    // After the morph fraction elapses, the step must HOLD its destination —
    // the articulation that separates trig-style steps from a constant glide.
    #[test]
    fn morph_holds_destination_after_window() {
        let mut seq = make_seq(4);
        seq.morph = 0.5;
        seq.manual_step(1); // cur=1, from=arrived step 0
        seq.step_timer = seq.step_dur * 0.75; // past the 50 % morph point
        let got = seq.current_params();
        let want = &seq.steps[1].params;
        assert!(
            (got.mp[crate::MP_KSCALE] - want.mp[crate::MP_KSCALE]).abs() < 1e-4,
            "kscale must sit exactly on the destination during the hold"
        );
        assert!((got.mp[crate::MP_W_BAND] - want.mp[crate::MP_W_BAND]).abs() < 1e-4);
    }

    #[test]
    fn morph_full_matches_classic_glide() {
        let mut a = make_seq(4);
        let mut b = make_seq(4);
        a.morph = 1.0;
        b.morph = 1.0;
        a.manual_step(1);
        b.manual_step(1);
        a.step_timer = a.step_dur * 0.5;
        b.step_timer = b.step_dur * 0.5;
        let pa = a.current_params();
        let pb = b.current_params();
        assert!((pa.mp[crate::MP_KSCALE] - pb.mp[crate::MP_KSCALE]).abs() < 1e-6);
        // And mid-glide is strictly between the endpoints.
        let from = a.from_params.mp[crate::MP_KSCALE];
        let to = a.steps[1].params.mp[crate::MP_KSCALE];
        let mid = pa.mp[crate::MP_KSCALE];
        assert!(
            (mid - from) * (to - mid) > 0.0,
            "mid-glide must sit between endpoints"
        );
    }

    // ── Trig conditions ──────────────────────────────────────────────────
    #[test]
    fn cond_one_of_two_plays_alternate_visits() {
        assert!(cond_passes((1, 2), 1));
        assert!(!cond_passes((1, 2), 2));
        assert!(cond_passes((1, 2), 3));
        assert!(!cond_passes((2, 2), 1));
        assert!(cond_passes((2, 2), 2));
        assert!(cond_passes((1, 1), 7), "(1,1) must always pass");
    }

    #[test]
    fn advance_skips_condition_failed_step() {
        let mut seq = make_seq(3);
        seq.manual = false;
        // Step 1 plays only on its 2nd visit → the first pass must skip to 2.
        seq.steps[1].cond = (2, 2);
        seq.advance_next();
        assert_eq!(seq.cur, 2, "first visit to a (2,2) step must skip it");
        // Wrap around: 2 → 0 → second visit of 1 passes.
        seq.advance_next();
        assert_eq!(seq.cur, 0);
        seq.advance_next();
        assert_eq!(seq.cur, 1, "second visit must play the (2,2) step");
    }

    // ── Drift lane ───────────────────────────────────────────────────────
    #[test]
    fn drift_moves_but_stays_bounded() {
        let mut seq = make_seq(2);
        seq.morph = 1.0;
        seq.step_timer = seq.step_dur; // fully arrived: lerp t = 1
        seq.drift = 0.0;
        let still = seq.current_params();
        seq.drift = 1.0;
        seq.clock = 7.3; // arbitrary point on the drift clock
        let alive = seq.current_params();
        // It moves…
        let moved = (alive.mp[crate::MP_ZOOM] - still.mp[crate::MP_ZOOM]).abs() > 1e-6
            || (alive.mp[crate::MP_COLOR_SHIFT] - still.mp[crate::MP_COLOR_SHIFT]).abs() > 1e-6;
        assert!(moved, "drift at full depth must actually move the image");
        // …but stays near the base: bounded micro-motion, not chaos.
        assert!(
            (alive.mp[crate::MP_ZOOM] - still.mp[crate::MP_ZOOM]).abs()
                <= still.mp[crate::MP_ZOOM] * 0.05 + 1e-4,
            "zoom drift must stay within ~5 %"
        );
        assert!(
            (alive.mp[crate::MP_KSCALE] - still.mp[crate::MP_KSCALE]).abs()
                <= still.mp[crate::MP_KSCALE] * 0.04 + 1e-4,
            "kscale drift must stay within ~3 %"
        );
    }

    #[test]
    fn drift_zero_is_exact_passthrough() {
        let mut seq = make_seq(2);
        seq.step_timer = seq.step_dur;
        seq.drift = 0.0;
        seq.clock = 123.4;
        let got = seq.current_params();
        let want = &seq.steps[0].params;
        assert!((got.mp[crate::MP_ZOOM] - want.mp[crate::MP_ZOOM]).abs() < 1e-6);
        assert!((got.mp[crate::MP_COLOR_SHIFT] - want.mp[crate::MP_COLOR_SHIFT]).abs() < 1e-6);
    }

    // ── Curves ───────────────────────────────────────────────────────────
    #[test]
    fn curves_settle_exactly_at_endpoints() {
        for c in [
            TranCurve::Linear,
            TranCurve::EaseInOut,
            TranCurve::Silk,
            TranCurve::Snap,
            TranCurve::Bounce,
            TranCurve::Over,
        ] {
            assert!((c.apply(0.0)).abs() < 1e-4, "{} must start at 0", c.label());
            assert!(
                (c.apply(1.0) - 1.0).abs() < 1e-4,
                "{} must end at 1",
                c.label()
            );
        }
    }

    #[test]
    fn over_curve_overshoots_then_settles() {
        let peak = (0..100)
            .map(|i| TranCurve::Over.apply(i as f32 / 99.0))
            .fold(f32::MIN, f32::max);
        assert!(
            peak > 1.02 && peak < 1.25,
            "OVER must overshoot a bounded amount, got peak {peak}"
        );
    }
}

// ── Tests for parameter routing, LFO, lerp_fp, and fb_auto ───────────────
#[cfg(test)]
mod routing_tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // LfoWave::sample must stay in [-1, 1] for every wave at any phase.
    #[test]
    fn lfo_wave_sample_in_bipolar_range() {
        for w in [
            LfoWave::Sine,
            LfoWave::Triangle,
            LfoWave::Saw,
            LfoWave::Square,
            LfoWave::Pulse,
            LfoWave::Steps,
        ] {
            for k in 0..1000 {
                let p = (k as f32) * 0.013;
                let s = w.sample(p);
                assert!(
                    s >= -1.0 - 1e-4 && s <= 1.0 + 1e-4,
                    "{:?} out of range at phase {p}: {s}",
                    w
                );
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
        lfo.mp = [LfoSrc::A; MP_SLOTS];
        for src in [
            &mut lfo.fb_zoom,
            &mut lfo.fb_decay,
            &mut lfo.fb_offset_x,
            &mut lfo.fb_offset_y,
            &mut lfo.fb_rotation,
            &mut lfo.fb_color_shift,
            &mut lfo.fb_saturation,
            &mut lfo.fb_brightness,
            &mut lfo.fb_inject,
            &mut lfo.fb_fold_angle,
            &mut lfo.fb_motion_blur,
        ] {
            *src = LfoSrc::A;
        }
        lfo.a.depth = 10.0; // wildly more than any param range
        lfo.a.wave = LfoWave::Square; // always ±1
        lfo.a.phase = 0.0;

        // phase=0 → square at t=0 is +1
        let eff_pos = apply_modulation(&fp, &lfo, &mic, &bands, 0.0);
        // pick a t where square is -1 (phase >= 0.5 in cycle)
        let t_neg = 0.6 / lfo.a.rate.max(1e-6);
        let eff_neg = apply_modulation(&fp, &lfo, &mic, &bands, t_neg);

        let check = |label: &str, v: f32, lo: f32, hi: f32| {
            assert!(
                v >= lo - 1e-4 && v <= hi + 1e-4,
                "{label}={v} outside [{lo},{hi}]"
            );
        };
        for eff in [&eff_pos, &eff_neg] {
            check("kscale", eff.mp[crate::MP_KSCALE], 0.1, 5.0);
            check("speed", eff.mp[crate::MP_SPEED], 0.0, 2.0);
            check("field_mix", eff.mp[crate::MP_FIELD_MIX], 0.0, 1.0);
            check("iso_level", eff.mp[crate::MP_ISO_LEVEL], 0.0, 1.0);
            check("color_shift", eff.mp[crate::MP_COLOR_SHIFT], 0.0, 1.0);
            check("zoom", eff.mp[crate::MP_ZOOM], 0.2, 5.0);
            check("w_lattice", eff.mp[crate::MP_W_LATTICE], 0.0, 2.0);
            check("w_motif", eff.mp[crate::MP_W_MOTIF], 0.0, 2.0);
            check("w_band", eff.mp[crate::MP_W_BAND], 0.0, 2.0);
            check("fb_zoom", eff.fb_zoom, 0.90, 1.10);
            check("fb_decay", eff.fb_decay, 0.30, 0.99);
            check("fb_offset_x", eff.fb_offset_x, -0.10, 0.10);
            check("fb_offset_y", eff.fb_offset_y, -0.10, 0.10);
            check("fb_rotation", eff.fb_rotation, -0.30, 0.30);
            check("fb_color_shift", eff.fb_color_shift, -1.00, 1.00);
            check("fb_saturation", eff.fb_saturation, 0.00, 2.00);
            check("fb_brightness", eff.fb_brightness, 0.00, 2.00);
            check("fb_inject", eff.fb_inject, 0.00, 1.00);
            check("fb_fold_angle", eff.fb_fold_angle, -3.14, 3.14);
            check("fb_motion_blur", eff.fb_motion_blur, 0.00, 0.95);
        }
    }

    // apply_modulation with all-off LFO and Off mic must pass fp through unchanged.
    #[test]
    fn apply_modulation_passes_through_when_off() {
        let mut fp = FieldParams::default();
        fp.mp[crate::MP_KSCALE] = 1.7;
        fp.mp[crate::MP_SPEED] = 0.42;
        fp.mp[crate::MP_FIELD_MIX] = 0.6;
        fp.mp[crate::MP_ISO_LEVEL] = 0.31;
        fp.mp[crate::MP_COLOR_SHIFT] = 0.77;
        let lfo = LfoParams::unrouted();
        let mic = MicParams::default();
        let bands = audio::AudioBands::default();
        let eff = apply_modulation(&fp, &lfo, &mic, &bands, 1.23);
        assert!(approx_eq(
            eff.mp[crate::MP_KSCALE],
            fp.mp[crate::MP_KSCALE],
            1e-5
        ));
        assert!(approx_eq(
            eff.mp[crate::MP_SPEED],
            fp.mp[crate::MP_SPEED],
            1e-5
        ));
        assert!(approx_eq(
            eff.mp[crate::MP_FIELD_MIX],
            fp.mp[crate::MP_FIELD_MIX],
            1e-5
        ));
        assert!(approx_eq(
            eff.mp[crate::MP_COLOR_SHIFT],
            fp.mp[crate::MP_COLOR_SHIFT],
            1e-5
        ));
    }

    // lerp_fp must wrap color_shift via the *short* arc (hue is circular).
    #[test]
    fn lerp_fp_color_shift_takes_short_arc_forward() {
        let mut a = FieldParams::default();
        a.mp[crate::MP_COLOR_SHIFT] = 0.9;
        let mut b = FieldParams::default();
        b.mp[crate::MP_COLOR_SHIFT] = 0.1;
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
        assert!(
            approx_eq(r25.mp[crate::MP_COLOR_SHIFT], 0.95, 1e-4),
            "forward wrap @t=0.25 should be ~0.95, got {}",
            r25.mp[crate::MP_COLOR_SHIFT]
        );
        assert!(
            approx_eq(r75.mp[crate::MP_COLOR_SHIFT], 0.05, 1e-4),
            "forward wrap @t=0.75 should be ~0.05, got {}",
            r75.mp[crate::MP_COLOR_SHIFT]
        );
    }

    #[test]
    fn lerp_fp_color_shift_no_wrap_for_short_delta() {
        let mut a = FieldParams::default();
        a.mp[crate::MP_COLOR_SHIFT] = 0.2;
        let mut b = FieldParams::default();
        b.mp[crate::MP_COLOR_SHIFT] = 0.4;
        let r = lerp_fp(&a, &b, 0.5);
        // Plain linear mid: 0.3
        assert!(approx_eq(r.mp[crate::MP_COLOR_SHIFT], 0.3, 1e-5));
    }

    #[test]
    fn lerp_fp_endpoints() {
        let mut a = FieldParams::default();
        a.mp[crate::MP_KSCALE] = 1.0;
        a.mp[crate::MP_ZOOM] = 0.5;
        let mut b = FieldParams::default();
        b.mp[crate::MP_KSCALE] = 5.0;
        b.mp[crate::MP_ZOOM] = 2.0;
        let r0 = lerp_fp(&a, &b, 0.0);
        let r1 = lerp_fp(&a, &b, 1.0);
        assert!(approx_eq(r0.mp[crate::MP_KSCALE], 1.0, 1e-5));
        assert!(approx_eq(r0.mp[crate::MP_ZOOM], 0.5, 1e-5));
        assert!(approx_eq(r1.mp[crate::MP_KSCALE], 5.0, 1e-5));
        assert!(approx_eq(r1.mp[crate::MP_ZOOM], 2.0, 1e-5));
    }

    #[test]
    fn lerp_fp_discrete_snaps_to_destination_immediately() {
        // Digitakt-style: when a trig fires, that step's discrete params apply
        // for the entire duration. mode / fb_blend_mode / fb_mirror / fb_enabled
        // must equal `b` from t=0, so the rendered mode matches the highlighted
        // step in the sequencer dock without a half-step lag.
        let mut a = FieldParams::default();
        a.mode = 1;
        a.fb_blend_mode = 0;
        a.fb_mirror = 0;
        a.fb_enabled = false;
        let mut b = FieldParams::default();
        b.mode = 7;
        b.fb_blend_mode = 3;
        b.fb_mirror = 5;
        b.fb_enabled = true;
        for t in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let r = lerp_fp(&a, &b, t);
            assert_eq!(r.mode, 7, "mode must snap to b at t={t}");
            assert_eq!(r.fb_blend_mode, 3, "fb_blend_mode must snap to b at t={t}");
            assert_eq!(r.fb_mirror, 5, "fb_mirror must snap to b at t={t}");
            assert!(r.fb_enabled, "fb_enabled must snap to b at t={t}");
        }
    }

    // fb_auto_params must override every fb_* field of base, and only those.
    #[test]
    fn fb_auto_overrides_only_fb_fields() {
        let base = sp(5, 2.5, 0.7, 0.6, 0.4, 0.3, 1.5, 1.7, 0.9, 0.6)
            .params
            .clone();
        let auto = fb_auto_params(3.7, &base);
        // Non-feedback fields preserved exactly:
        assert_eq!(auto.mode, base.mode);
        assert!(approx_eq(
            auto.mp[crate::MP_KSCALE],
            base.mp[crate::MP_KSCALE],
            1e-5
        ));
        assert!(approx_eq(
            auto.mp[crate::MP_SPEED],
            base.mp[crate::MP_SPEED],
            1e-5
        ));
        assert!(approx_eq(
            auto.mp[crate::MP_FIELD_MIX],
            base.mp[crate::MP_FIELD_MIX],
            1e-5
        ));
        assert!(approx_eq(
            auto.mp[crate::MP_ISO_LEVEL],
            base.mp[crate::MP_ISO_LEVEL],
            1e-5
        ));
        assert!(approx_eq(
            auto.mp[crate::MP_COLOR_SHIFT],
            base.mp[crate::MP_COLOR_SHIFT],
            1e-5
        ));
        assert!(approx_eq(
            auto.mp[crate::MP_ZOOM],
            base.mp[crate::MP_ZOOM],
            1e-5
        ));
        assert!(approx_eq(
            auto.mp[crate::MP_W_LATTICE],
            base.mp[crate::MP_W_LATTICE],
            1e-5
        ));
        assert!(approx_eq(
            auto.mp[crate::MP_W_MOTIF],
            base.mp[crate::MP_W_MOTIF],
            1e-5
        ));
        assert!(approx_eq(
            auto.mp[crate::MP_W_BAND],
            base.mp[crate::MP_W_BAND],
            1e-5
        ));
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
        assert!(
            differs,
            "fb_auto must evolve across scenes (a={:?}, b={:?})",
            (a.fb_mirror, a.fb_zoom, a.fb_decay),
            (b.fb_mirror, b.fb_zoom, b.fb_decay)
        );
    }

    #[test]
    fn fb_auto_is_finite() {
        let base = FieldParams::default();
        for k in 0..500 {
            let t = (k as f32) * 0.13;
            let a = fb_auto_params(t, &base);
            for v in [
                a.fb_zoom,
                a.fb_decay,
                a.fb_offset_x,
                a.fb_offset_y,
                a.fb_rotation,
                a.fb_color_shift,
                a.fb_saturation,
                a.fb_brightness,
                a.fb_inject,
                a.fb_fold_angle,
                a.fb_motion_blur,
            ] {
                assert!(v.is_finite(), "fb_auto produced non-finite value at t={t}");
            }
        }
    }

    // ── LfoSrc enum behaviour ─────────────────────────────────────────────
    #[test]
    fn lfo_src_cycles_off_a_b() {
        let mut s = LfoSrc::Off;
        s = s.next();
        assert_eq!(s, LfoSrc::A);
        s = s.next();
        assert_eq!(s, LfoSrc::B);
        s = s.next();
        assert_eq!(s, LfoSrc::Off);
    }

    // ── LfoEngine sampling ────────────────────────────────────────────────
    #[test]
    fn lfo_engine_phase_offset_shifts_sample() {
        let e0 = LfoEngine {
            rate: 1.0,
            depth: 1.0,
            wave: LfoWave::Sine,
            phase: 0.0,
        };
        let e1 = LfoEngine {
            rate: 1.0,
            depth: 1.0,
            wave: LfoWave::Sine,
            phase: 0.25,
        };
        // sin(2π·0) = 0; sin(2π·0.25) = 1.
        assert!(approx_eq(e0.sample(0.0), 0.0, 1e-4));
        assert!(approx_eq(e1.sample(0.0), 1.0, 1e-4));
    }

    // ── Two independent LFOs combine in apply_modulation ──────────────────
    #[test]
    fn apply_modulation_uses_two_lfos_independently() {
        let fp = FieldParams::default();
        let mic = MicParams::default();
        let bands = audio::AudioBands::default();
        let mut lfo = LfoParams::unrouted();
        lfo.flux = 1.0; // generous budget: this test is about routing, not the cap
                        // LFO A: square at +1 → bumps kscale upward; LFO B: square at -1 → pulls speed down.
        lfo.a = LfoEngine {
            rate: 0.5,
            depth: 1.0,
            wave: LfoWave::Square,
            phase: 0.0,
        };
        lfo.b = LfoEngine {
            rate: 0.5,
            depth: 1.0,
            wave: LfoWave::Square,
            phase: 0.5,
        };
        lfo.mp[crate::MP_KSCALE] = LfoSrc::A;
        lfo.mp[crate::MP_SPEED] = LfoSrc::B;
        let eff = apply_modulation(&fp, &lfo, &mic, &bands, 0.0);
        // A at t=0 → +1. B at t=0 with phase 0.5 → -1.
        // kscale base 1.4 + 1.0*range(4.9) clamped to 5.0
        assert!(
            eff.mp[crate::MP_KSCALE] > 4.5,
            "LFO A (square +1) should pull kscale to max; got {}",
            eff.mp[crate::MP_KSCALE]
        );
        // speed base 0.3 − range(2.0) → clamped to 0.0
        assert!(
            eff.mp[crate::MP_SPEED] < 0.05,
            "LFO B (square -1) should pull speed to min; got {}",
            eff.mp[crate::MP_SPEED]
        );
    }

    #[test]
    fn apply_modulation_off_src_does_not_modulate() {
        let mut fp = FieldParams::default();
        fp.mp[crate::MP_KSCALE] = 2.0;
        let mic = MicParams::default();
        let bands = audio::AudioBands::default();
        let mut lfo = LfoParams::default();
        lfo.a.depth = 1.0;
        lfo.a.wave = LfoWave::Square;
        lfo.a.phase = 0.0;
        lfo.mp[crate::MP_KSCALE] = LfoSrc::Off; // explicitly off
        let eff = apply_modulation(&fp, &lfo, &mic, &bands, 0.0);
        assert!(
            (eff.mp[crate::MP_KSCALE] - 2.0).abs() < 1e-5,
            "LfoSrc::Off must not change the param"
        );
    }

    // ── Flux budget ──────────────────────────────────────────────────────
    // With many maxed routes, total requested modulation must be scaled down
    // to the budget — no single frame can move the image more than the cap.
    #[test]
    fn flux_budget_scales_down_over_asking_routes() {
        let fp = FieldParams::default();
        let mic = MicParams::default();
        let bands = audio::AudioBands::default();
        let mut lfo = LfoParams::unrouted();
        lfo.a = LfoEngine {
            rate: 0.5,
            depth: 1.0,
            wave: LfoWave::Square,
            phase: 0.0,
        };
        lfo.mp = [LfoSrc::A; MP_SLOTS]; // 16 routes × depth 1.0 = 16 units requested
        lfo.flux = 0.5; // budget = 1.5 units
        let (eff, load) = apply_modulation_ex(&fp, &lfo, &mic, &bands, 0.0);
        assert!(
            load > 1.0,
            "16 maxed routes must overload a 0.5 flux budget, got {load}"
        );
        // kscale asked for a full-range swing (+4.9) but the budget admits
        // only 1.5/16 of it per route ≈ 0.46 — far from the 5.0 rail.
        let delta = eff.mp[crate::MP_KSCALE] - fp.mp[crate::MP_KSCALE];
        assert!(delta > 0.0, "scaled route must still move in its direction");
        assert!(
            delta < 1.0,
            "budget must prevent the full-range jump, got +{delta}"
        );
    }

    #[test]
    fn flux_budget_leaves_underbudget_routes_untouched() {
        let fp = FieldParams::default();
        let mic = MicParams::default();
        let bands = audio::AudioBands::default();
        let mut lfo = LfoParams::unrouted();
        lfo.a = LfoEngine {
            rate: 0.5,
            depth: 0.2,
            wave: LfoWave::Square,
            phase: 0.0,
        };
        lfo.mp[crate::MP_KSCALE] = LfoSrc::A; // one small route: 0.2 units
        lfo.flux = 0.5; // budget 1.5 — plenty
        let (eff, load) = apply_modulation_ex(&fp, &lfo, &mic, &bands, 0.0);
        assert!(
            load < 1.0,
            "a single 0.2-unit route must not saturate, got {load}"
        );
        let (lo, hi) = slot_range(fp.mode, crate::MP_KSCALE);
        let want = fp.mp[crate::MP_KSCALE] + 0.2 * (hi - lo);
        assert!(
            (eff.mp[crate::MP_KSCALE] - want.min(hi)).abs() < 1e-4,
            "under budget the delta must be applied unscaled"
        );
    }

    #[test]
    fn flux_zero_freezes_all_modulation() {
        let fp = FieldParams::default();
        let mic = MicParams::default();
        let bands = audio::AudioBands::default();
        let mut lfo = LfoParams::unrouted();
        lfo.a = LfoEngine {
            rate: 0.5,
            depth: 1.0,
            wave: LfoWave::Square,
            phase: 0.0,
        };
        lfo.mp = [LfoSrc::A; MP_SLOTS];
        lfo.flux = 0.0;
        let eff = apply_modulation(&fp, &lfo, &mic, &bands, 0.0);
        for i in 0..MP_SLOTS {
            assert!(
                (eff.mp[i] - fp.mp[i]).abs() < 1e-5,
                "flux 0 must freeze slot {i}"
            );
        }
    }

    // ── tour_lfo_preset wiring ────────────────────────────────────────────
    #[test]
    fn tour_lfo_preset_has_both_engines_configured() {
        let p = tour_lfo_preset();
        assert!(p.a.depth > 0.0 && p.a.rate > 0.0);
        assert!(p.b.depth > 0.0 && p.b.rate > 0.0);
        // At least one target routed to A and one to B
        let any_a = [
            p.mp[crate::MP_KSCALE],
            p.fb_saturation,
            p.mp[crate::MP_COLOR_SHIFT],
        ]
        .into_iter()
        .any(|s| s == LfoSrc::A);
        let any_b = [p.fb_zoom, p.fb_color_shift, p.fb_fold_angle]
            .into_iter()
            .any(|s| s == LfoSrc::B);
        assert!(
            any_a && any_b,
            "tour_lfo_preset should route some targets to A and some to B"
        );
    }

    #[test]
    fn tour_lfo_preset_fb_heavy_drives_feedback() {
        let p = tour_lfo_preset_fb_heavy();
        let fb_targets = [
            p.fb_zoom,
            p.fb_decay,
            p.fb_color_shift,
            p.fb_saturation,
            p.fb_brightness,
            p.fb_rotation,
            p.fb_offset_x,
            p.fb_offset_y,
            p.fb_fold_angle,
            p.fb_motion_blur,
        ];
        let active = fb_targets.iter().filter(|&&s| s != LfoSrc::Off).count();
        assert!(
            active >= 6,
            "fb_heavy preset should route at least 6 feedback params to an LFO, got {active}"
        );
    }

    // Number of LFO targets routed in a preset (anything not Off).
    fn routed_count(p: &LfoParams) -> usize {
        let targets = [
            p.mp[crate::MP_KSCALE],
            p.mp[crate::MP_SPEED],
            p.mp[crate::MP_FIELD_MIX],
            p.mp[crate::MP_ISO_LEVEL],
            p.mp[crate::MP_COLOR_SHIFT],
            p.mp[crate::MP_ZOOM],
            p.mp[crate::MP_W_LATTICE],
            p.mp[crate::MP_W_MOTIF],
            p.mp[crate::MP_W_BAND],
            p.fb_zoom,
            p.fb_decay,
            p.fb_offset_x,
            p.fb_offset_y,
            p.fb_rotation,
            p.fb_color_shift,
            p.fb_saturation,
            p.fb_brightness,
            p.fb_inject,
            p.fb_fold_angle,
            p.fb_motion_blur,
        ];
        targets.iter().filter(|&&s| s != LfoSrc::Off).count()
    }

    // ── Trip level ────────────────────────────────────────────────────────
    #[test]
    fn trip_level_clamps_to_valid_range() {
        assert_eq!(TripLevel::new(0).get(), 0);
        assert_eq!(TripLevel::new(9).get(), 9);
        assert_eq!(TripLevel::new(99).get(), 9);
        assert_eq!(TripLevel::new(0).dec().get(), 0);
        assert_eq!(TripLevel::new(9).inc().get(), 9);
    }

    #[test]
    fn trip_level_zero_is_silent() {
        let p = tour_lfo_preset_for_level(TripLevel::new(0));
        assert_eq!(
            routed_count(&p),
            0,
            "level 0 must not route any LFO target — got {} routed",
            routed_count(&p)
        );
        assert_eq!(p.a.depth, 0.0, "level 0 LFO A depth should be 0");
        assert_eq!(p.b.depth, 0.0, "level 0 LFO B depth should be 0");
        let lvl0 = TripLevel::new(0);
        assert!(!lvl0.fb_enabled(), "level 0 disables the feedback layer");
    }

    #[test]
    fn trip_level_nine_is_madness() {
        let p = tour_lfo_preset_for_level(TripLevel::new(9));
        let r = routed_count(&p);
        assert!(r >= 15, "level 9 should route >=15 LFO targets, got {r}");
        assert!(
            p.a.depth > 0.5 && p.b.depth > 0.4,
            "level 9 depths should be near max"
        );
        let lvl9 = TripLevel::new(9);
        assert!(lvl9.fb_enabled(), "level 9 enables feedback layer");
        // Madness wave choices: steps on A, square on B.
        assert_eq!(p.a.wave, LfoWave::Steps);
        assert_eq!(p.b.wave, LfoWave::Square);
    }

    #[test]
    fn trip_level_scales_routed_targets_monotonically() {
        // Each level should route at least as many targets as the previous one.
        let mut prev = 0;
        for l in 0..=9u8 {
            let r = routed_count(&tour_lfo_preset_for_level(TripLevel::new(l)));
            assert!(
                r >= prev,
                "routed-target count must be monotonic — level {l} got {r} after {prev}"
            );
            prev = r;
        }
    }

    #[test]
    fn trip_level_scales_scene_and_walk_durations_downward() {
        // Higher level → shorter scene / walk.
        for l in 0..9u8 {
            let lo = TripLevel::new(l);
            let hi = TripLevel::new(l + 1);
            assert!(
                lo.scene_len() > hi.scene_len(),
                "scene_len must shrink with level — level {l}: {} vs level {}: {}",
                lo.scene_len(),
                l + 1,
                hi.scene_len()
            );
            assert!(
                lo.walk_dur() > hi.walk_dur(),
                "walk_dur must shrink with level"
            );
        }
    }

    #[test]
    fn trip_level_feedback_gated_by_level() {
        for l in 0..=2u8 {
            assert!(!TripLevel::new(l).fb_enabled());
        }
        for l in 3..=9u8 {
            assert!(TripLevel::new(l).fb_enabled());
        }
    }

    #[test]
    fn tour_field_params_level_zero_holds_mode_steady() {
        let lvl0 = TripLevel::new(0);
        // Two times far apart should yield the SAME mode at level 0 (no scene rotation).
        let a = tour_field_params(0.5, 3, TourStyle::Curated, lvl0);
        let b = tour_field_params(120.0, 3, TourStyle::Curated, lvl0);
        assert_eq!(
            a.mode, b.mode,
            "level 0 should keep a single mode per crystal across time"
        );
        assert!(!a.fb_enabled, "level 0 should disable the feedback layer");
    }

    #[test]
    fn tour_field_params_level_nine_swaps_modes() {
        let lvl9 = TripLevel::new(9);
        // Sample several points; at level 9 we expect the mode index to change at least once.
        let modes: Vec<u32> = (0..20)
            .map(|i| tour_field_params(i as f32 * 0.4, 0, TourStyle::Curated, lvl9).mode)
            .collect();
        let distinct: std::collections::HashSet<_> = modes.iter().copied().collect();
        assert!(
            distinct.len() >= 3,
            "level 9 should swap modes rapidly — got distinct={:?}",
            distinct
        );
    }
    #[test]
    fn tour_mode_switch_keeps_previous_image_until_fade_completes() {
        let level = TripLevel::new(6);
        let mut tour = Tour::new();
        let mut from = FieldParams::default();
        from.mode = 4;
        let mut to = FieldParams::default();
        to.mode = 22;

        assert!(tour.transition_params(from.clone(), 0.0, level).is_none());

        let (previous, at_start) = tour
            .transition_params(to.clone(), 0.0, level)
            .expect("a changed mode must begin a transition");
        assert_eq!(previous.mode, from.mode);
        assert_eq!(
            at_start, 0.0,
            "outgoing image must be fully visible at the cut"
        );

        let duration = (level.scene_len() * 0.18).clamp(0.18, 0.75);
        let (_, midpoint) = tour
            .transition_params(to.clone(), duration * 0.5, level)
            .expect("transition must remain active at its midpoint");
        assert!(
            midpoint > 0.0 && midpoint < 1.0,
            "blend must progress continuously"
        );

        assert!(
            tour.transition_params(to, duration, level).is_none(),
            "completed fade must release the outgoing image"
        );
    }
    #[test]
    fn default_scene_transition_crossfades_every_mode_change() {
        let color = [0.2, 0.4, 0.6, 0.0];
        let mut from_params = FieldParams::default();
        from_params.mode = 3;
        let mut to_params = FieldParams::default();
        to_params.mode = 27;
        let from = field_params_to_uniform(&from_params, 1.0, 1.0, &color);
        let to = field_params_to_uniform(&to_params, 2.0, 1.0, &color);
        let mut transition = SceneTransition::new();

        assert!(transition.track(from, 0.0).is_none());
        let (outgoing, at_start) = transition
            .track(to, 0.0)
            .expect("mode changes must start the default crossfade");
        assert_eq!(outgoing.mode, from.mode);
        assert_eq!(outgoing.time, to.time, "outgoing animation stays live");
        assert_eq!(at_start, 0.0);

        let (_, midpoint) = transition
            .track(to, DEFAULT_SCENE_FADE_DUR * 0.5)
            .expect("crossfade must remain live at its midpoint");
        assert!(midpoint > 0.0 && midpoint < 1.0);
        assert!(transition.track(to, DEFAULT_SCENE_FADE_DUR).is_none());

        // Randomized fields and k-point snaps retain the same mode but still
        // request an explicit image bridge before their G-vectors change.
        transition.begin(None);
        assert!(
            transition.track(to, 0.0).is_some(),
            "explicit same-mode regeneration must crossfade"
        );
    }

    #[test]
    fn crossfade_styles_preserve_endpoints_and_vary_the_melt() {
        let mids: Vec<f32> = CrossfadeStyle::ALL
            .into_iter()
            .map(|style| {
                assert_eq!(
                    style.apply(0.0),
                    0.0,
                    "{style:?} must retain outgoing image at start"
                );
                assert_eq!(
                    style.apply(1.0),
                    1.0,
                    "{style:?} must fully reveal incoming image at end"
                );
                style.apply(0.5)
            })
            .collect();
        assert!(
            mids.windows(2).any(|pair| (pair[0] - pair[1]).abs() > 0.05),
            "crossfade styles need visibly distinct midpoint timing: {mids:?}"
        );
    }
}
