//! MIDI input — Control Change → live param control, Note On → triggers.
//!
//! Native builds use `midir` (ALSA / CoreMIDI / WinMM). The browser build
//! exposes the same `MidiCapture::start() -> Option<Self>` shape but always
//! returns `None`; Web MIDI support can be added later via `web-sys`.
//!
//! ## CC mapping (channel-agnostic; takes any channel)
//!
//! Field params (broad-range, smooth dial sources work best):
//!   CC  1  kscale        0.1 .. 5.0
//!   CC  2  speed         0.0 .. 2.0
//!   CC  3  field_mix     0.0 .. 1.0
//!   CC  4  iso_level     0.0 .. 1.0
//!   CC  5  color_shift   0.0 .. 1.0
//!   CC  6  zoom          0.2 .. 5.0
//!   CC  7  w_lattice     0.0 .. 2.0
//!   CC  8  w_motif       0.0 .. 2.0
//!   CC  9  w_band        0.0 .. 2.0
//!
//! Feedback params:
//!   CC 20  fb_decay        0.30 .. 0.99
//!   CC 21  fb_zoom         0.90 .. 1.10
//!   CC 22  fb_color_shift -1.00 .. 1.00
//!   CC 23  fb_saturation   0.00 .. 2.00
//!   CC 24  fb_brightness   0.00 .. 2.00
//!   CC 25  fb_rotation    -0.30 .. 0.30
//!   CC 26  fb_inject       0.00 .. 1.00
//!   CC 27  fb_motion_blur  0.00 .. 0.95
//!
//! Notes (any channel, Note On with velocity > 0):
//!   C2  (36)  toggle FB AUTO
//!   D2  (38)  toggle TOUR
//!   E2  (40)  toggle SEQ
//!   F2  (41)  SEQ manual prev step
//!   G2  (43)  SEQ manual next step
//!   A2  (45)  📸 CAPTURE current state into a new step

use std::sync::{Arc, Mutex};

/// Shared, lock-protected state populated by the MIDI input thread.
pub struct MidiState {
    /// CC values in [0, 1], indexed by CC number 0..128. Latched until changed.
    pub cc:        [f32; 128],
    /// Whether a given CC has *ever* been received this session — used to avoid
    /// stomping on UI sliders for CCs the controller never touched.
    pub cc_set:    [bool; 128],
    /// Note On events pulled by the app once per frame (FIFO, capped).
    pub notes:     Vec<u8>,
    /// Last received CC `(number, raw value 0..127)`, for the status UI.
    pub last_cc:   Option<(u8, u8)>,
    /// Connected port name, if any.
    pub port_name: Option<String>,
}

impl Default for MidiState {
    fn default() -> Self {
        Self {
            cc: [0.0; 128], cc_set: [false; 128],
            notes: Vec::new(), last_cc: None, port_name: None,
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub struct MidiCapture {
    pub state: Arc<Mutex<MidiState>>,
}

#[cfg(target_arch = "wasm32")]
impl MidiCapture {
    pub fn start() -> Option<Self> { None }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::MidiCapture;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{Arc, MidiState, Mutex};
    use midir::{MidiInput, MidiInputConnection};

    pub struct MidiCapture {
        pub state: Arc<Mutex<MidiState>>,
        _conn:     MidiInputConnection<()>,
    }

    impl MidiCapture {
        /// Try to open the first non-default MIDI input port. Returns None if
        /// no port is available (e.g. user has no MIDI device or virtual cable).
        pub fn start() -> Option<Self> {
            let midi_in = MidiInput::new("crystaldive").ok()?;
            let ports = midi_in.ports();
            if ports.is_empty() {
                log::info!("midi: no input ports available");
                return None;
            }
            // Prefer a port that isn't the Windows "Microsoft GS Wavetable Synth"
            // output side (we only see inputs, but tolerate any naming).
            let port = ports.iter().find(|p| {
                midi_in.port_name(p)
                    .map(|n| !n.contains("Microsoft GS"))
                    .unwrap_or(true)
            }).unwrap_or(&ports[0]).clone();
            let name = midi_in.port_name(&port).ok();
            let state = Arc::new(Mutex::new(MidiState {
                port_name: name.clone(),
                ..MidiState::default()
            }));
            let s2 = state.clone();
            let conn = midi_in.connect(&port, "crystaldive-midi-in", move |_t, msg, _| {
                if msg.is_empty() { return; }
                let status = msg[0] & 0xF0;
                match status {
                    0xB0 => {
                        // Control Change: status, cc, value
                        if msg.len() < 3 { return; }
                        let cc = msg[1] as usize;
                        let val = msg[2];
                        if cc < 128 {
                            if let Ok(mut s) = s2.lock() {
                                s.cc[cc] = (val as f32) / 127.0;
                                s.cc_set[cc] = true;
                                s.last_cc = Some((cc as u8, val));
                            }
                        }
                    }
                    0x90 => {
                        // Note On (vel>0); Note Off when vel==0 we just ignore.
                        if msg.len() < 3 { return; }
                        let note = msg[1];
                        let vel  = msg[2];
                        if vel > 0 {
                            if let Ok(mut s) = s2.lock() {
                                // Cap the queue so a stuck device can't run us OOM.
                                if s.notes.len() < 64 { s.notes.push(note); }
                            }
                        }
                    }
                    _ => { /* ignore everything else for now */ }
                }
            }, ()).ok()?;
            log::info!("midi: open  ({})", name.as_deref().unwrap_or("?"));
            Some(MidiCapture { state, _conn: conn })
        }
    }
}

// ── Apply MIDI to FieldParams ────────────────────────────────────────────
//
// Pulled out as a free function (CPU-only) so it can be unit-tested without
// any GPU/MIDI dependencies. Each FieldParams field maps to a CC; if that CC
// hasn't been touched, the corresponding param is left alone (so manual UI
// edits and tour values keep working in parallel).

/// Snapshot of just the CC state needed by `apply_midi` — keeps the apply
/// function GPU-/midir-free and trivially testable.
#[derive(Clone)]
pub struct CcSnapshot {
    pub cc:     [f32; 128],
    pub cc_set: [bool; 128],
}

impl Default for CcSnapshot {
    fn default() -> Self {
        Self { cc: [0.0; 128], cc_set: [false; 128] }
    }
}

impl CcSnapshot {
    pub fn from_state(s: &MidiState) -> Self {
        Self { cc: s.cc, cc_set: s.cc_set }
    }
}

/// MIDI Note IDs for triggers (see module docs).
pub mod note {
    pub const FB_AUTO_TOGGLE: u8 = 36; // C2
    pub const TOUR_TOGGLE:    u8 = 38; // D2
    pub const SEQ_TOGGLE:     u8 = 40; // E2
    pub const SEQ_PREV:       u8 = 41; // F2
    pub const SEQ_NEXT:       u8 = 43; // G2
    pub const SEQ_CAPTURE:    u8 = 45; // A2
}

/// CC numbers and their per-param ranges (kept in sync with src/midi.rs docs
/// and tools/midi_sim.py). Free function so tests can sweep deterministically.
pub fn apply_midi_cc(fp: &mut crate::FieldParams, snap: &CcSnapshot) {
    let lerp = |c: usize, lo: f32, hi: f32| -> Option<f32> {
        if snap.cc_set[c] { Some(lo + (hi - lo) * snap.cc[c]) } else { None }
    };
    if let Some(v) = lerp(1,  0.1,  5.0)  { fp.kscale         = v; }
    if let Some(v) = lerp(2,  0.0,  2.0)  { fp.speed          = v; }
    if let Some(v) = lerp(3,  0.0,  1.0)  { fp.field_mix      = v; }
    if let Some(v) = lerp(4,  0.0,  1.0)  { fp.iso_level      = v; }
    if let Some(v) = lerp(5,  0.0,  1.0)  { fp.color_shift    = v; }
    if let Some(v) = lerp(6,  0.2,  5.0)  { fp.zoom           = v; }
    if let Some(v) = lerp(7,  0.0,  2.0)  { fp.w_lattice      = v; }
    if let Some(v) = lerp(8,  0.0,  2.0)  { fp.w_motif        = v; }
    if let Some(v) = lerp(9,  0.0,  2.0)  { fp.w_band         = v; }
    if let Some(v) = lerp(20, 0.30, 0.99) { fp.fb_decay       = v; }
    if let Some(v) = lerp(21, 0.90, 1.10) { fp.fb_zoom        = v; }
    if let Some(v) = lerp(22,-1.0,  1.0)  { fp.fb_color_shift = v; }
    if let Some(v) = lerp(23, 0.0,  2.0)  { fp.fb_saturation  = v; }
    if let Some(v) = lerp(24, 0.0,  2.0)  { fp.fb_brightness  = v; }
    if let Some(v) = lerp(25,-0.30, 0.30) { fp.fb_rotation    = v; }
    if let Some(v) = lerp(26, 0.0,  1.0)  { fp.fb_inject      = v; }
    if let Some(v) = lerp(27, 0.0,  0.95) { fp.fb_motion_blur = v; }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FieldParams;

    fn approx_eq(a: f32, b: f32) -> bool { (a - b).abs() < 1e-4 }

    #[test]
    fn unset_ccs_do_not_modify_fp() {
        let original = FieldParams { kscale: 2.5, speed: 0.7, ..FieldParams::default() };
        let mut fp = original.clone();
        let snap = CcSnapshot::default(); // no CCs set
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.kscale, original.kscale));
        assert!(approx_eq(fp.speed, original.speed));
    }

    #[test]
    fn cc1_drives_kscale_range() {
        let mut fp = FieldParams::default();
        let mut snap = CcSnapshot::default();
        snap.cc_set[1] = true;

        snap.cc[1] = 0.0;
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.kscale, 0.1), "cc=0 → kscale=0.1");

        snap.cc[1] = 1.0;
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.kscale, 5.0), "cc=1 → kscale=5.0");

        snap.cc[1] = 0.5;
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.kscale, 2.55), "cc=0.5 → kscale=mid");
    }

    #[test]
    fn cc22_supports_bipolar_range() {
        let mut fp = FieldParams::default();
        let mut snap = CcSnapshot::default();
        snap.cc_set[22] = true;
        snap.cc[22] = 0.0;
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.fb_color_shift, -1.0));
        snap.cc[22] = 0.5;
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.fb_color_shift, 0.0));
        snap.cc[22] = 1.0;
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.fb_color_shift, 1.0));
    }

    #[test]
    fn cc_set_only_partial_does_not_touch_others() {
        let original = FieldParams { speed: 1.2, w_motif: 1.7, ..FieldParams::default() };
        let mut fp = original.clone();
        let mut snap = CcSnapshot::default();
        snap.cc_set[1] = true; snap.cc[1] = 0.5;  // kscale only
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.kscale,  2.55));
        assert!(approx_eq(fp.speed,   original.speed));
        assert!(approx_eq(fp.w_motif, original.w_motif));
    }

    #[test]
    fn fb_decay_clamps_to_documented_range() {
        // Ranges per module docs: 0.30 .. 0.99
        let mut fp = FieldParams::default();
        let mut snap = CcSnapshot::default();
        snap.cc_set[20] = true;
        snap.cc[20] = 0.0;
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.fb_decay, 0.30));
        snap.cc[20] = 1.0;
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.fb_decay, 0.99));
    }
}
