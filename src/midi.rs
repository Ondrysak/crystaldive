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
                if let Ok(mut s) = s2.lock() {
                    super::parse_message(msg, &mut s);
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

// ── Message parsing ──────────────────────────────────────────────────────
//
// Pulled out of the midir callback so it can be unit-tested without spinning
// up a real MIDI device. Takes a raw MIDI message (1+ bytes) and a state to
// mutate. Channel-agnostic: we mask the low nibble off the status byte.

/// Status nibble: Control Change.
const STATUS_CC:  u8 = 0xB0;
/// Status nibble: Note On.
const STATUS_NOTE_ON: u8 = 0x90;

/// Maximum queued note events. Stops a stuck device from growing the Vec
/// without bound between frames; pulled into a const so tests can pin it.
pub const NOTES_QUEUE_CAP: usize = 64;

/// Parse one MIDI message into `state`. Robust to short/empty/unknown messages.
/// Returns `true` if the message produced a state mutation.
pub fn parse_message(msg: &[u8], state: &mut MidiState) -> bool {
    if msg.is_empty() { return false; }
    let status = msg[0] & 0xF0;
    match status {
        STATUS_CC => {
            if msg.len() < 3 { return false; }
            let cc  = msg[1] as usize;
            let val = msg[2];
            if cc < 128 && val < 128 {
                state.cc[cc]     = (val as f32) / 127.0;
                state.cc_set[cc] = true;
                state.last_cc    = Some((cc as u8, val));
                return true;
            }
            false
        }
        STATUS_NOTE_ON => {
            if msg.len() < 3 { return false; }
            let note = msg[1];
            let vel  = msg[2];
            // Note On with velocity 0 is conventionally Note Off — ignore.
            if vel > 0 && state.notes.len() < NOTES_QUEUE_CAP {
                state.notes.push(note);
                return true;
            }
            false
        }
        _ => false, // pitch bend, aftertouch, sysex, etc — not wired up
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
    use crate::{MP_KSCALE, MP_SPEED, MP_FIELD_MIX, MP_ISO_LEVEL, MP_COLOR_SHIFT,
                MP_ZOOM, MP_W_LATTICE, MP_W_MOTIF, MP_W_BAND};
    if let Some(v) = lerp(1,  0.1,  5.0)  { fp.mp[MP_KSCALE]      = v; }
    if let Some(v) = lerp(2,  0.0,  2.0)  { fp.mp[MP_SPEED]       = v; }
    if let Some(v) = lerp(3,  0.0,  1.0)  { fp.mp[MP_FIELD_MIX]   = v; }
    if let Some(v) = lerp(4,  0.0,  1.0)  { fp.mp[MP_ISO_LEVEL]   = v; }
    if let Some(v) = lerp(5,  0.0,  1.0)  { fp.mp[MP_COLOR_SHIFT] = v; }
    if let Some(v) = lerp(6,  0.2,  5.0)  { fp.mp[MP_ZOOM]        = v; }
    if let Some(v) = lerp(7,  0.0,  2.0)  { fp.mp[MP_W_LATTICE]   = v; }
    if let Some(v) = lerp(8,  0.0,  2.0)  { fp.mp[MP_W_MOTIF]     = v; }
    if let Some(v) = lerp(9,  0.0,  2.0)  { fp.mp[MP_W_BAND]      = v; }
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

    // ── apply_midi_cc: range mapping ──────────────────────────────────────

    #[test]
    fn unset_ccs_do_not_modify_fp() {
        let mut original = FieldParams::default();
        original.mp[crate::MP_KSCALE] = 2.5;
        original.mp[crate::MP_SPEED]  = 0.7;
        let mut fp = original.clone();
        let snap = CcSnapshot::default(); // no CCs set
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.mp[crate::MP_KSCALE], original.mp[crate::MP_KSCALE]));
        assert!(approx_eq(fp.mp[crate::MP_SPEED], original.mp[crate::MP_SPEED]));
    }

    #[test]
    fn cc1_drives_kscale_range() {
        let mut fp = FieldParams::default();
        let mut snap = CcSnapshot::default();
        snap.cc_set[1] = true;

        snap.cc[1] = 0.0;
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.mp[crate::MP_KSCALE], 0.1), "cc=0 -> kscale=0.1");

        snap.cc[1] = 1.0;
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.mp[crate::MP_KSCALE], 5.0), "cc=1 -> kscale=5.0");

        snap.cc[1] = 0.5;
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.mp[crate::MP_KSCALE], 2.55), "cc=0.5 -> kscale=mid");
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
        let mut original = FieldParams::default();
        original.mp[crate::MP_SPEED]   = 1.2;
        original.mp[crate::MP_W_MOTIF] = 1.7;
        let mut fp = original.clone();
        let mut snap = CcSnapshot::default();
        snap.cc_set[1] = true; snap.cc[1] = 0.5;  // kscale only
        apply_midi_cc(&mut fp, &snap);
        assert!(approx_eq(fp.mp[crate::MP_KSCALE],  2.55));
        assert!(approx_eq(fp.mp[crate::MP_SPEED],   original.mp[crate::MP_SPEED]));
        assert!(approx_eq(fp.mp[crate::MP_W_MOTIF], original.mp[crate::MP_W_MOTIF]));
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

    // ── Full CC mapping table — guards against copy-paste typos in apply ──

    /// (cc, range_lo, range_hi, getter)
    type CcExpect = (usize, f32, f32, fn(&FieldParams) -> f32);
    const CC_EXPECT: &[CcExpect] = &[
        (1,   0.1,  5.0,  |p| p.mp[crate::MP_KSCALE]),
        (2,   0.0,  2.0,  |p| p.mp[crate::MP_SPEED]),
        (3,   0.0,  1.0,  |p| p.mp[crate::MP_FIELD_MIX]),
        (4,   0.0,  1.0,  |p| p.mp[crate::MP_ISO_LEVEL]),
        (5,   0.0,  1.0,  |p| p.mp[crate::MP_COLOR_SHIFT]),
        (6,   0.2,  5.0,  |p| p.mp[crate::MP_ZOOM]),
        (7,   0.0,  2.0,  |p| p.mp[crate::MP_W_LATTICE]),
        (8,   0.0,  2.0,  |p| p.mp[crate::MP_W_MOTIF]),
        (9,   0.0,  2.0,  |p| p.mp[crate::MP_W_BAND]),
        (20,  0.30, 0.99, |p| p.fb_decay),
        (21,  0.90, 1.10, |p| p.fb_zoom),
        (22, -1.0,  1.0,  |p| p.fb_color_shift),
        (23,  0.0,  2.0,  |p| p.fb_saturation),
        (24,  0.0,  2.0,  |p| p.fb_brightness),
        (25, -0.30, 0.30, |p| p.fb_rotation),
        (26,  0.0,  1.0,  |p| p.fb_inject),
        (27,  0.0,  0.95, |p| p.fb_motion_blur),
    ];

    #[test]
    fn every_documented_cc_maps_min_mid_max() {
        for &(cc, lo, hi, get) in CC_EXPECT {
            for (v, expected) in [(0.0_f32, lo),
                                  (0.5,     lo + 0.5 * (hi - lo)),
                                  (1.0,     hi)] {
                let mut fp = FieldParams::default();
                let mut snap = CcSnapshot::default();
                snap.cc_set[cc] = true;
                snap.cc[cc] = v;
                apply_midi_cc(&mut fp, &snap);
                let got = get(&fp);
                assert!((got - expected).abs() < 1e-4,
                    "CC {cc} at v={v} expected {expected} got {got}");
            }
        }
    }

    #[test]
    fn unmapped_ccs_do_not_affect_fp() {
        // CCs 10..19 and 28..127 have no mapping. Setting them all to 1.0
        // must leave a default FieldParams completely untouched.
        let baseline = FieldParams::default();
        let mut fp = baseline.clone();
        let mut snap = CcSnapshot::default();
        for cc in 0..128usize {
            // Skip mapped CCs.
            let mapped = CC_EXPECT.iter().any(|&(c, ..)| c == cc);
            if mapped { continue; }
            snap.cc_set[cc] = true;
            snap.cc[cc] = 1.0;
        }
        apply_midi_cc(&mut fp, &snap);
        // Sample every mapped getter and confirm it equals the default.
        for &(_, _, _, get) in CC_EXPECT {
            assert!(approx_eq(get(&fp), get(&baseline)),
                "an unmapped CC bled into a mapped param");
        }
    }

    #[test]
    fn apply_midi_cc_is_idempotent_for_same_snapshot() {
        let mut snap = CcSnapshot::default();
        snap.cc_set[1] = true; snap.cc[1] = 0.42;
        snap.cc_set[24] = true; snap.cc[24] = 0.7;
        let mut a = FieldParams::default();
        let mut b = FieldParams::default();
        apply_midi_cc(&mut a, &snap);
        apply_midi_cc(&mut b, &snap);
        apply_midi_cc(&mut b, &snap); // second time should be a no-op effect
        assert!(approx_eq(a.mp[crate::MP_KSCALE],        b.mp[crate::MP_KSCALE]));
        assert!(approx_eq(a.fb_brightness, b.fb_brightness));
    }

    // ── parse_message: per-byte protocol behaviour ────────────────────────

    #[test]
    fn parse_empty_message_is_noop() {
        let mut s = MidiState::default();
        assert!(!parse_message(&[], &mut s));
        assert!(s.last_cc.is_none());
        assert!(s.notes.is_empty());
    }

    #[test]
    fn parse_truncated_cc_is_noop() {
        let mut s = MidiState::default();
        // CC status byte only — missing cc number and value
        assert!(!parse_message(&[0xB0], &mut s));
        // status + cc but no value
        assert!(!parse_message(&[0xB0, 7], &mut s));
        assert!(s.last_cc.is_none());
        for set in s.cc_set { assert!(!set); }
    }

    #[test]
    fn parse_truncated_note_is_noop() {
        let mut s = MidiState::default();
        assert!(!parse_message(&[0x90], &mut s));
        assert!(!parse_message(&[0x90, 60], &mut s));
        assert!(s.notes.is_empty());
    }

    #[test]
    fn parse_cc_updates_cc_cc_set_and_last_cc() {
        let mut s = MidiState::default();
        // Channel 5 (0xB5) → still parsed as CC: status nibble is 0xB0.
        assert!(parse_message(&[0xB5, 22, 64], &mut s));
        assert!((s.cc[22] - 64.0 / 127.0).abs() < 1e-4);
        assert!(s.cc_set[22]);
        assert_eq!(s.last_cc, Some((22, 64)));
        // CC at the endpoints:
        assert!(parse_message(&[0xB0, 1, 0], &mut s));
        assert!(s.cc[1].abs() < 1e-6);
        assert!(parse_message(&[0xB0, 1, 127], &mut s));
        assert!((s.cc[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn parse_note_on_velocity_zero_is_ignored() {
        let mut s = MidiState::default();
        assert!(!parse_message(&[0x90, 60, 0], &mut s));
        assert!(s.notes.is_empty());
    }

    #[test]
    fn parse_note_on_velocity_positive_queues() {
        let mut s = MidiState::default();
        assert!(parse_message(&[0x90, 36, 100], &mut s));
        assert!(parse_message(&[0x95, 38, 64], &mut s)); // any channel
        assert_eq!(s.notes, vec![36, 38]);
    }

    #[test]
    fn parse_note_off_status_is_ignored() {
        // True Note Off (0x80) — our minimal parser ignores it; only Note On
        // with velocity > 0 should queue. Belt-and-braces: the (Note On, vel=0)
        // convention is already covered above.
        let mut s = MidiState::default();
        assert!(!parse_message(&[0x80, 60, 64], &mut s));
        assert!(s.notes.is_empty());
    }

    #[test]
    fn parse_unknown_status_is_ignored() {
        let mut s = MidiState::default();
        // Pitch bend (0xE0), Channel Pressure (0xD0), Program Change (0xC0)
        for status in [0xE0u8, 0xD0, 0xC0] {
            assert!(!parse_message(&[status, 0x10, 0x20], &mut s));
        }
        assert!(s.last_cc.is_none());
        assert!(s.notes.is_empty());
    }

    #[test]
    fn parse_notes_queue_caps_at_constant() {
        let mut s = MidiState::default();
        for i in 0..(NOTES_QUEUE_CAP + 10) as u8 {
            parse_message(&[0x90, i % 100, 100], &mut s);
        }
        assert_eq!(s.notes.len(), NOTES_QUEUE_CAP,
            "queue must cap exactly at NOTES_QUEUE_CAP");
    }

    #[test]
    fn parse_out_of_range_cc_byte_is_ignored() {
        // CC index byte cannot be >= 128 in real MIDI (high bit reserved for status)
        // but defend the [cc as usize] indexing anyway.
        let mut s = MidiState::default();
        // 0x80 has high bit set — would be a new status byte mid-stream. We reject.
        assert!(!parse_message(&[0xB0, 0x80, 64], &mut s));
        for set in s.cc_set { assert!(!set); }
    }

    // ── State + constants sanity ──────────────────────────────────────────

    #[test]
    fn midi_state_default_is_empty() {
        let s = MidiState::default();
        for v in s.cc { assert_eq!(v, 0.0); }
        for v in s.cc_set { assert!(!v); }
        assert!(s.notes.is_empty());
        assert!(s.last_cc.is_none());
        assert!(s.port_name.is_none());
    }

    #[test]
    fn cc_snapshot_from_state_copies_cc_arrays() {
        let mut s = MidiState::default();
        s.cc[7] = 0.5;       s.cc_set[7]  = true;
        s.cc[42] = 0.9;      s.cc_set[42] = true;
        // notes/last_cc/port_name are NOT part of the snapshot — only CC state.
        s.notes.push(60);
        s.last_cc = Some((1, 1));
        let snap = CcSnapshot::from_state(&s);
        assert_eq!(snap.cc[7],  0.5);
        assert!(snap.cc_set[7]);
        assert_eq!(snap.cc[42], 0.9);
        assert!(snap.cc_set[42]);
        assert_eq!(snap.cc[0],  0.0);
        assert!(!snap.cc_set[0]);
    }

    #[test]
    fn note_constants_match_documented_ids() {
        // These are the user-visible bindings, so they must not drift silently.
        assert_eq!(note::FB_AUTO_TOGGLE, 36, "C2");
        assert_eq!(note::TOUR_TOGGLE,    38, "D2");
        assert_eq!(note::SEQ_TOGGLE,     40, "E2");
        assert_eq!(note::SEQ_PREV,       41, "F2");
        assert_eq!(note::SEQ_NEXT,       43, "G2");
        assert_eq!(note::SEQ_CAPTURE,    45, "A2");
    }

    #[test]
    fn notes_queue_cap_is_pinned() {
        // App reads `state.notes` once per frame; the cap exists to guarantee
        // a single frame doesn't allocate without bound. Pin it so a future
        // refactor doesn't quietly drop it.
        assert_eq!(NOTES_QUEUE_CAP, 64);
    }

    // ── End-to-end: parse a small message sequence and check fp result ────

    #[test]
    fn end_to_end_parse_then_apply() {
        // Simulate: controller sends CC1=64, CC24=127, Note 36 on.
        let mut s = MidiState::default();
        parse_message(&[0xB0,  1,  64], &mut s);
        parse_message(&[0xB0, 24, 127], &mut s);
        parse_message(&[0x90, 36, 100], &mut s);
        // App drains notes for triggers...
        let drained = std::mem::take(&mut s.notes);
        assert_eq!(drained, vec![36]);
        // ...and applies CC state to a copy of FieldParams.
        let mut fp = FieldParams::default();
        let snap = CcSnapshot::from_state(&s);
        apply_midi_cc(&mut fp, &snap);
        // CC1: 64/127 ≈ 0.5039 → kscale lerps near mid of [0.1, 5.0] ≈ 2.57
        assert!((fp.mp[crate::MP_KSCALE] - 2.57).abs() < 0.05,
            "kscale should land near mid-range, got {}", fp.mp[crate::MP_KSCALE]);
        // CC24: 1.0 → fb_brightness at max of [0.0, 2.0]
        assert!((fp.fb_brightness - 2.0).abs() < 1e-3,
            "fb_brightness should hit 2.0, got {}", fp.fb_brightness);
    }
}
