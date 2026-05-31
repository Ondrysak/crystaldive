---
name: testing
description: How to test the crystaldive visualizer — run the right test layer for a change, validate WGSL shaders without a GPU, run the GPU perf gate, and regenerate bundled presets. Use when adding/editing a render mode, touching shaders, changing FieldParams/presets/sequencer, or before committing.
---

# Testing crystaldive

This repo has three test layers, run with different commands. Pick the layer that matches what you changed — don't reach for the GPU bench when a unit test covers it, and don't trust `cargo test` alone after a shader edit.

## Layer 1 — fast suite (no GPU), run this constantly

```sh
cargo test                    # 37 lib + 92 bin tests, ~1s after build
cargo test --bin crystal-viz  # just the bin (sequencer, LFO, MIDI, presets, routing)
cargo test --lib              # just the lib (modes, poscar, reciprocal, renderer, symmetry)
```

This is the default loop. It compiles **and naga-validates the assembled WGSL shader** (see Layer 2) but needs no GPU, so it runs anywhere. Run it after every change before moving on.

Run one test by name:
```sh
cargo test --lib field_shader_parses_and_validates
cargo test --bin crystal-viz seq_tests::play_mode_pingpong_reverses_at_ends
```

## Layer 2 — shader validation (no GPU), the safety net for mode authors

`src/modes/mod.rs::shader_tests::field_shader_parses_and_validates` parses the
concatenated `FIELD_SHADER` with **naga** (the same WGSL frontend wgpu uses at
runtime) and validates it. It runs inside Layer 1 — no separate command.

**Why it matters:** `cargo test` historically passed while the app crashed at
launch, because nothing compiled the shader. This test closes that gap. It
catches, before the app ever launches:

- **Reserved keywords** used as identifiers — `target`, `patch`, `sample`,
  `filter`, `select` (as a var), etc. These are the #1 cause of
  `create_shader_module` panics. WGSL's reserved list is large; if naga rejects
  a name, rename the local (e.g. `target` → `look_at`, `patch` → `half`).
- Type errors, undefined identifiers, bad function signatures.

After **any** `mode_*.wgsl` edit, run:
```sh
cargo test --lib field_shader_parses_and_validates
```
A green here means the shader will compile on a real device. A red here is the
exact error you'd otherwise only see as a runtime `wgpu::Validation Error`.

## Layer 3 — GPU perf gate (needs a real GPU)

```sh
cargo test --release --test mode_perf -- --ignored --nocapture
```

`tests/mode_perf.rs` actually creates the pipeline and renders **every** mode at
512×512, then fails if any mode exceeds **10× the median** frame time. This is
the only layer that proves a mode renders on hardware and is performant. It's
`#[ignore]`d so the fast suite stays GPU-free.

- Median is set by cheap 2D modes (~0.2ms). A fullscreen raymarcher is
  inherently ~30× that and **cannot** pass — add its display name to the
  `EXEMPT` list at the top of `tests/mode_perf.rs` (see `LATTICE WALK`).
- A new mode that lands at e.g. 16× median is a real regression — optimize it
  (e.g. bounding-sphere ray clipping, fewer march steps, cheaper normals)
  before exempting. Only exempt modes that are *intentionally* heavy.

## Layer 4 — bundled presets (gated generator)

Presets in `presets/*.preset.json` are generated, not hand-written. Regenerate
the whole set:
```sh
cargo test --bin crystal-viz -- --ignored gen_bundled_presets
```
The guard `bundled_presets_have_no_silent_skips` (runs in Layer 1) asserts every
step has `prob == 1.0` and `muted == false`. Don't relax it — the tame
`prob`/`muted` distributions are for live randomization only, not the bundle.

## After adding or renaming a mode — full checklist

A new mode touches several synced lists. Run, in order:

```sh
cargo test --lib field_shader_parses_and_validates   # shader compiles
cargo test                                            # MODE_NAMES sync, indexing
cargo test --release --test mode_perf -- --ignored    # renders + perf (needs GPU)
```

The fast suite enforces the cross-file invariants for you:
- `modes::info::tests::modes_are_indexed_in_order` / `names_match_modes` — the
  `MODES` table in `src/modes/info.rs` is the single source of truth.
- `bench::tests::mode_names_count_matches_dispatch` — bump the
  `assert_eq!(MODE_NAMES.len(), N)` in `src/bench.rs` when N changes.

If any of these are red, you forgot a step in the "Adding a new mode" list in
the top-level `CLAUDE.md`.

## Pre-commit hook (optional, opt-in)

`.githooks/pre-commit` runs `cargo test -- --include-ignored` (fast suite + GPU
perf gate) on every commit. It is **not** enabled by default
(`core.hooksPath` points at `.git/hooks`). Turn it on with:
```sh
git config core.hooksPath .githooks
```
Bypass a single commit with `git commit --no-verify`. Because it's opt-in, do
not assume a clean `git commit` means the perf gate passed — run Layer 3
yourself before pushing a mode change.

## Quick reference

| Change you made                     | Run this                                                        |
|-------------------------------------|-----------------------------------------------------------------|
| Edited a `mode_*.wgsl`              | `cargo test --lib field_shader_parses_and_validates` then Layer 3 |
| Added/renamed a mode                | full checklist above                                            |
| Sequencer / LFO / MIDI / routing    | `cargo test --bin crystal-viz`                                  |
| Preset format or generator          | `cargo test` + regenerate (Layer 4) if output changed           |
| Renderer / FieldUniform layout      | `cargo test --lib` + Layer 3 (uniform mismatch only shows on GPU)|
| Anything, before pushing            | `cargo test` + Layer 3                                          |
