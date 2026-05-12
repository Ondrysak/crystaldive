#!/usr/bin/env python3
"""
MIDI simulator for crystaldive -- sends CCs / Notes to a MIDI output port so
you can demo MIDI control without a hardware controller.

Mapping (kept in sync with src/midi.rs):

  CC  1..9    field params (kscale, speed, field_mix, iso_level, color_shift,
                            zoom, w_lattice, w_motif, w_band)
  CC 20..27   feedback (decay, zoom, color_shift, sat, brightness, rotation,
                        inject, motion_blur)
  Notes (any channel):
    36 (C2)  toggle FB AUTO
    38 (D2)  toggle TOUR
    40 (E2)  toggle SEQ
    41 (F2)  SEQ prev step (manual)
    43 (G2)  SEQ next step (manual)
    45 (A2)  CAPTURE step (snapshot current state into sequencer)

Requirements:
  pip install mido python-rtmidi

Windows note:
  Standard Windows MIDI does not expose virtual loopback ports. Install
  loopMIDI (https://www.tobias-erichsen.de/software/loopmidi.html), open it,
  click [+] to create a port (e.g. "crystaldive"). Then both this script and
  crystaldive will see that port -- pick it in both.

macOS / Linux:
  No setup needed. macOS has IAC Driver in Audio MIDI Setup; Linux has
  ALSA/snd-virmidi.

Usage:
  python tools/midi_sim.py --list                      # list output ports
  python tools/midi_sim.py PORT                        # sweep CCs (default 30s)
  python tools/midi_sim.py PORT --mode sweep --duration 60
  python tools/midi_sim.py PORT --mode notes           # fire one of each trigger
  python tools/midi_sim.py PORT --mode interactive     # arrow keys = step CCs
"""
from __future__ import annotations

import argparse
import math
import sys
import time

try:
    import mido
except ImportError:
    print("`mido` not installed. Run:  pip install mido python-rtmidi", file=sys.stderr)
    sys.exit(1)

# (cc_number, name, lo, hi) -- same ranges as src/midi.rs::apply_midi_cc.
CC_MAP: list[tuple[int, str, float, float]] = [
    (1,  "kscale",         0.1,  5.0),
    (2,  "speed",          0.0,  2.0),
    (3,  "field_mix",      0.0,  1.0),
    (4,  "iso_level",      0.0,  1.0),
    (5,  "color_shift",    0.0,  1.0),
    (6,  "zoom",           0.2,  5.0),
    (7,  "w_lattice",      0.0,  2.0),
    (8,  "w_motif",        0.0,  2.0),
    (9,  "w_band",         0.0,  2.0),
    (20, "fb_decay",       0.30, 0.99),
    (21, "fb_zoom",        0.90, 1.10),
    (22, "fb_color_shift",-1.00, 1.00),
    (23, "fb_saturation",  0.00, 2.00),
    (24, "fb_brightness",  0.00, 2.00),
    (25, "fb_rotation",   -0.30, 0.30),
    (26, "fb_inject",      0.00, 1.00),
    (27, "fb_motion_blur", 0.00, 0.95),
]

NOTE_MAP = {
    36: "FB_AUTO_TOGGLE",
    38: "TOUR_TOGGLE",
    40: "SEQ_TOGGLE",
    41: "SEQ_PREV",
    43: "SEQ_NEXT",
    45: "SEQ_CAPTURE",
}


def list_ports() -> None:
    print("Available MIDI output ports:")
    names = mido.get_output_names()
    if not names:
        print("  (none -- install loopMIDI on Windows, or enable IAC on macOS)")
        return
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")


def match_port(query: str) -> str | None:
    """Return the first output port whose name contains `query` (case-insensitive)."""
    q = query.lower()
    for n in mido.get_output_names():
        if q in n.lower():
            return n
    return None


def sweep(port: mido.ports.BaseOutput, duration: float) -> None:
    """Send each CC as a low-frequency sine with a distinct phase offset."""
    print(f"Sweeping {len(CC_MAP)} CCs for {duration:.1f}s "
          f"(at 30 Hz update rate)…  Ctrl-C to stop.")
    t0 = time.time()
    rate = 30.0  # Hz
    period = 1.0 / rate
    try:
        while True:
            t = time.time() - t0
            if t >= duration:
                break
            for i, (cc, _name, _lo, _hi) in enumerate(CC_MAP):
                # Each CC drifts at slightly different rate so they don't sync.
                f = 0.07 + 0.013 * i
                phase = i * 0.4
                v = 0.5 + 0.5 * math.sin(2 * math.pi * f * t + phase)
                port.send(mido.Message("control_change", control=cc,
                                       value=int(round(v * 127))))
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopped.")


def fire_notes(port: mido.ports.BaseOutput, gap: float = 0.6) -> None:
    """Tap each trigger note in turn so you can see them all wire through."""
    for n, name in NOTE_MAP.items():
        print(f"  Note On {n:>3}  ({name})")
        port.send(mido.Message("note_on",  note=n, velocity=100))
        time.sleep(0.05)
        port.send(mido.Message("note_off", note=n, velocity=0))
        time.sleep(gap)


def interactive(port: mido.ports.BaseOutput) -> None:
    """Simple REPL: type 'cc N V' or 'note N' or 'help'."""
    print("Interactive mode. Type:")
    print("  cc <N> <0..127>     send Control Change")
    print("  cc <N> <0..1>       float, normalized 0..1 → 0..127")
    print("  note <N>            send Note On then Off")
    print("  list                show CC + note mapping")
    print("  quit                exit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        try:
            if cmd in ("q", "quit", "exit"):
                return
            elif cmd in ("h", "help", "list"):
                print("CC map:")
                for cc, name, lo, hi in CC_MAP:
                    print(f"  CC {cc:>2}  {name:<14}  {lo:>6.2f} .. {hi:>6.2f}")
                print("Notes:")
                for n, name in NOTE_MAP.items():
                    print(f"  Note {n}  {name}")
            elif cmd == "cc" and len(parts) == 3:
                cc = int(parts[1])
                raw = parts[2]
                if "." in raw:
                    val = max(0, min(127, int(round(float(raw) * 127))))
                else:
                    val = max(0, min(127, int(raw)))
                port.send(mido.Message("control_change", control=cc, value=val))
                print(f"  → CC {cc} = {val}")
            elif cmd == "note" and len(parts) == 2:
                n = int(parts[1])
                port.send(mido.Message("note_on",  note=n, velocity=100))
                time.sleep(0.04)
                port.send(mido.Message("note_off", note=n, velocity=0))
                print(f"  → Note {n}")
            else:
                print("  ?  type 'help'")
        except ValueError:
            print("  bad number")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("port", nargs="?", help="output port name (substring OK)")
    p.add_argument("--list", action="store_true", help="list output ports and exit")
    p.add_argument("--mode", choices=["sweep", "notes", "interactive"],
                   default="sweep",
                   help="sweep CCs (default), tap each trigger note, "
                        "or open an interactive REPL")
    p.add_argument("--duration", type=float, default=30.0,
                   help="sweep mode duration in seconds (default 30)")
    args = p.parse_args()

    if args.list or not args.port:
        list_ports()
        return 0

    name = match_port(args.port)
    if not name:
        print(f"No output port matches '{args.port}'.", file=sys.stderr)
        list_ports()
        return 1

    print(f"Opening {name!r}…")
    with mido.open_output(name) as port:
        if args.mode == "sweep":
            sweep(port, args.duration)
        elif args.mode == "notes":
            fire_notes(port)
        elif args.mode == "interactive":
            interactive(port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
