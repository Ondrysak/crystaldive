# crystaldive — agent orientation

Real-time crystal + physics-field visualizer. Rust + WGSL + WebGPU, native (winit+egui+wgpu) and wasm32 targets. Deployed as a single HTML to a Cloudflare Worker (`plain-band-13be`).

The mode catalog started as condensed-matter visualizations (Bloch, Berry, Hofstadter, plasmon, magnon, …) and has expanded to cover broader physics — quantum wave mechanics (`wavepacket`), classical EM radiation (`dipole_rad`), fluid dynamics (`karman`), nonlinear dynamics (`lorenz`), and general relativity (`lensing`). All modes follow the same per-file contract; see [`src/modes/CONTRACT.md`](src/modes/CONTRACT.md). New physics domains are welcome — bias toward one self-contained `mode_<slug>.wgsl` file per concept rather than wider refactors.

## Layout

- `src/main.rs` — app, egui UI, App event loop, sequencer, LFO engine, MIDI/mic input, headless `--render` CLI. Keep types `pub` if `preset.rs` or `bench.rs` need them.
- `src/modes/` — one `mode_<name>.wgsl` per visual mode + `prelude.wgsl`, `core_modes.wgsl`, `dispatch.wgsl`. Glued by `mod.rs` via `include_str!` at compile time.
- `src/preset.rs` — `Preset` struct, bundled presets (`include_str!("../presets/*.preset.json")`), random generator. Random presets must use tame `prob`/`muted` distributions (see history of "skipping steps" bug).
- `src/bench.rs` — wgpu headless renderer used by `cargo bench` and the `--render` CLI. Includes `MODE_NAMES.len()` assertion.
- `src/renderer.rs` — `FieldUniform` (must include `_pad: [f32; 2]`), crystal upload, pipelines.
- `presets/*.preset.json` — bundled clips. Every step must have `prob: 1.0, muted: false`.
- `web/`, `scripts/build-web.{ps1,sh}`, `scripts/make-single-html.mjs` — wasm build + single-file bundle.
- `fetch_mp.py` — Materials Project fetcher; reads `MP_API_KEY` from `.env`.

## Adding a new mode

1. Create `src/modes/mode_<name>.wgsl` defining `fn render_<name>(uv: vec2<f32>) -> vec3<f32>`.
2. Add `include_str!("mode_<name>.wgsl"),` in `src/modes/mod.rs` (before `dispatch.wgsl`).
3. Add a `case Nu { col = render_<name>(uv); }` in `src/modes/dispatch.wgsl`.
4. Append the display name to **both** `MODE_NAMES` lists — `src/main.rs` (typed `[&str; N]`) and `src/bench.rs` (`&[&str]`). They must stay in sync.
5. Bump the count in the `assert_eq!(MODE_NAMES.len(), N)` in `src/bench.rs` (`mode_names_count_matches_dispatch` test).
6. Prefix mode-local WGSL helpers with the mode name to avoid collisions.

WGSL gotcha: function-local `let` arrays cannot be runtime-indexed. Unroll or factor out a helper.

## Data model

- `Crystal` (`src/poscar.rs`) — lattice (3×3 Å basis) + `Vec<Atom>` (species + cartesian pos). Built by parsing POSCAR-format strings; one is bundled per crystal in `src/crystals/`.
- `FieldParams` (`src/main.rs:31`) — all per-frame mode/feedback parameters, edited by egui and serialized into presets.
- `FieldUniform` (`src/renderer.rs:50`) — `#[repr(C)] Pod/Zeroable` block uploaded to the WGSL `FU` uniform. Layout must match `prelude.wgsl`'s `FU` struct exactly, including `_pad: [f32; 2]` and `num_g` (set per-frame from the loaded crystal).
- Field-shader input texture (`crystal_field`) is sampled by every mode via helpers in `prelude.wgsl` (`crystal_field`, `cfield_col`, etc.).

## Runtime pipeline (frame composition)

Each frame the App ticks:

1. **Sequencer** (`Sequencer`, `SeqStep`) advances by `step_dur * step.dur_mul`, picks the next step per `SeqPlayMode` (Forward/Reverse/PingPong/Random), and interpolates `FieldParams` from prev→cur using `TranCurve`. Discrete fields (`mode`, feedback enable, mirror, blend) snap at `t=0` so the bottom-row highlight matches what's on screen (Digitakt sync).
2. **LFO engine** (`LfoParams`, two banks A/B) samples its waveforms at `t` and applies depth-scaled offsets to every routed param (`LfoSrc::{Off, A, B}`).
3. **Mic / MIDI** modulation layered on top (`MicParams`, MIDI CC bindings).
4. `field_params_to_uniform` packs the result into `FieldUniform`, then `renderer` draws via the fragment-shader switch in `dispatch.wgsl`.

`TripLevel` (0–9) is a meta-knob that scales LFO rates/depths, scene length, fade duration, and feedback strength — `0 still` → `9 madness`. Random presets and the "tour" both pull their distributions from `TripLevel` so the ladder stays coherent.

## Bundled presets

- Files live in `presets/*.preset.json`; each is `include_str!`'d into `BUNDLED_PRESETS` in `src/preset.rs`.
- Regenerate the whole set with the gated test:
  ```sh
  cargo test --bin crystal-viz -- --ignored gen_bundled_presets
  ```
- The `bundled_presets_have_no_silent_skips` test guards `prob == 1.0` and `muted == false` on every step; do not relax it without reason — the random-preset generator's tame `prob`/`muted` distributions only apply to live randomization, not the bundle.
- Adding a new bundled preset: extend the `(slug, name, seed, trip, dur, mode_bias)` table inside `gen_bundled_presets`, rerun the test, then add the new file to `BUNDLED_PRESETS`.

## Secrets

- `.env`, `.env.local`, `.env.*.local` are gitignored. Never commit them.
- `MP_API_KEY` lives only in `.env`. Never hardcode it in source files. The retired key in git history was rotated.
- `fetch_mp.py` reads via `_load_dotenv()` + `os.environ.get("MP_API_KEY")`. Don't refactor that to inline the key.

## Build / test / render

```sh
cargo test                                            # fast suite (lib+bin), naga-validates the shader, no GPU
cargo test --bin crystal-viz                          # bin only (~92 tests)
cargo test --release --test mode_perf -- --ignored    # GPU perf gate: renders every mode, fails >10× median
cargo run --release                                   # native app
pwsh scripts/build-web.ps1                            # wasm + single-html bundle
target/release/crystal-viz --render PRESET.json \
  --out FRAMES_DIR --res 1280x720 --fps 30 --duration 10
```

**For anything beyond a trivial change, follow the [`testing` skill](.claude/skills/testing/SKILL.md)** — it explains the three test layers and which to run for what. Key rule: after editing any `mode_*.wgsl`, run `cargo test --lib field_shader_parses_and_validates`. It parses the assembled `FIELD_SHADER` with naga (no GPU) and catches WGSL **reserved-keyword** identifiers (`target`, `patch`, `sample`, …) and type errors that otherwise only surface as a `create_shader_module` crash at app launch — `cargo test` passing is **not** sufficient proof a shader compiles without it.

Deploy (human-approved, never pre-approve):
```sh
npx wrangler deploy --name plain-band-13be
```

## Conventions

- No `Read`/`Edit` on `web/pkg/*` — those are wasm-bindgen build artifacts, regenerated by `build-web.ps1`.
- Don't run `git push`, `git reset --hard`, or `wrangler deploy` without explicit human approval. These are excluded from pre-approved permissions on purpose.
- Headless render uses `Rgba8UnormSrgb` so PNGs land in sRGB; do not switch back to `Rgba8Unorm` (was the "washed out" bug).
- New presets: keep `prob: 1.0` and `muted: false` on every step unless deliberately demoing randomness.
