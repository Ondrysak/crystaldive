# Mode-author contract

This directory holds one WGSL file per render mode of the crystal-field
fragment shader. The Rust glue (`mod.rs`) concatenates `prelude.wgsl` +
`core_modes.wgsl` + every `mode_*.wgsl` + `dispatch.wgsl` into a single WGSL
module at compile time.

## Your job as a mode author

Edit ONE file: `mode_<your_name>.wgsl`. Define exactly one entry function
with this signature:

```wgsl
fn render_<your_name>(uv: vec2<f32>) -> vec3<f32> { ... }
```

`uv` is in normalized device coords already aspect-corrected:
`uv.x ∈ [-aspect, +aspect]`, `uv.y ∈ [-1, +1]`. Origin is screen centre.

Return a linear-RGB color (no need to gamma-correct — the dispatcher does
tone-map + gamma + vignette afterwards).

## Available uniforms (from prelude.wgsl)

```wgsl
struct FU {
    mp:            array<vec4<f32>, 4>,  // 16-slot per-mode param bank (see below)
    time:          f32,    // seconds since start
    mode:          u32,    // current mode index
    num_g:         u32,    // number of valid entries in g_block (≤ 128)
    aspect:        f32,        // viewport.x / viewport.y
    crystal_color: vec4<f32>,  // .xyz accent color of current crystal system
    mouse:         vec2<f32>,  // .x,.y in [0,1] when mouse_down >= 0.5
    mouse_down:    f32,        // 0 or 1
    // … feedback (fb_*) fields follow; post-process, not per-mode.
}
@group(0) @binding(0) var<uniform> u: FU;
```

## The per-mode parameter bank (`mp`)

There are **no fixed global sliders** any more. Each mode declares its own
panel. Read slot `i` (0..15) with the accessor `mp(i)`:

```wgsl
fn mp(i: u32) -> f32 { return u.mp[i >> 2u][i & 3u]; }
```

Slots **0..8 are the canonical crystal-field generator params** — `crystal_field`,
`cf2`, `sdf`, `calc_normal`, and `cfield_col` read them internally:

```wgsl
const MP_KSCALE=0u; MP_SPEED=1u; MP_FIELD_MIX=2u; MP_ISO_LEVEL=3u;
const MP_COLOR_SHIFT=4u; MP_ZOOM=5u; MP_W_LATTICE=6u; MP_W_MOTIF=7u; MP_W_BAND=8u;
```

- If your mode **calls** `crystal_field`/`sdf`/`cfield_col`, leave slots 0..8
  meaning the generator's params (you may still expose/relabel them).
- If your mode does **not** use those helpers, you may repurpose all 16 slots.
- Slots **9..15 are always free** for bespoke per-mode knobs (e.g. LORENZ's
  `sigma`/`rho`/`beta`/`dt`).

Declare the panel in `info.rs` (`mode_params` → a `&[ParamDesc]` list of
`{slot, name, min, max, default}`). The egui panel renders one named slider per
declared slot for the active mode; LFO/mic ranges come from the same metadata.

The preset/random/tour generators only fill slots 0..8. A bespoke **free** slot
will arrive as `0` from those sources, so guard it:
`let v = mp(9u); let eff = select(DEFAULT, v, v > 1e-4);`.

```wgsl
struct GBlock {
    gamp:   array<vec4<f32>, 128>,  // xyz = G (unscaled Å⁻¹), w = amplitude
    phases: array<vec4<f32>, 128>,  // x = initial phase (yzw unused)
}
@group(0) @binding(1) var<uniform> g_block: GBlock;
```

## Reading G vectors directly

```wgsl
let ga = g_block.gamp[i];   // ga.xyz = G (unscaled), ga.w = amp
let ph = g_block.phases[i].x;  // initial phase
```

There are up to `MAX_G = 128` entries; iterate while `i < i32(u.num_g)`.
Use a hoisted bound for efficiency:
```wgsl
let ng = i32(u.num_g);
for (var i = 0i; i < ng; i++) { ... }
```

## Available helpers (from prelude.wgsl)

```wgsl
const TAU: f32 = 6.28318530718;

fn crystal_field(x: vec3<f32>) -> f32;          // Σ amp·cos(G·x + φ + ωt) (3 bands)
fn cf2(x: vec3<f32>) -> f32;                    // shifted/scaled twin field
fn sdf(p: vec3<f32>) -> f32;                    // |mix(cf, cf2, mp(2))| - mp(3)*0.3
fn calc_normal(p: vec3<f32>) -> vec3<f32>;      // gradient of sdf
fn cfield_col(f: f32, f2: f32, n: vec3<f32>) -> vec3<f32>;
```

You may write additional helpers in your own file. **Prefix them with your
mode name** (e.g. `phonon_rotate`, `moire_lattice`) to avoid colliding with
other modes — all files concatenate into one WGSL module.

## What the dispatcher does after you return

1. Vignette (`× (1 - 0.35·r²)`) unless your mode is in the skip-list in
   `dispatch.wgsl` (currently XRD, RECIP3D, KIKUCHI).
2. Reinhard tone-map (`col / (col + 0.6)`).
3. Gamma 0.85.

So return values can be HDR (≫ 1.0) — bright features will roll off.

## Mouse / input convention

Most existing modes use:
```wgsl
var az = u.time * u.speed * 0.15;
var el = 0.4;
if u.mouse_down >= 0.5 {
    az = u.mouse.x * TAU;
    el = (u.mouse.y - 0.5) * 2.5;
}
```
for orbit cameras. Follow this if you ship a 3D mode.

## Performance budget

- Loops with `if (i >= i32(u.num_g)) { break; }` — bound real iteration
  count, but the loop literal must be ≤ 128.
- Ray-march loops in core modes use 55–90 iterations; aim similar.
- No textures other than `g_tex`. No storage buffers. No compute.

## Testing

Once the agent has written the file, the orchestrator runs `cargo build`
from `visualizer/`. The mode is selectable from the FIELD-mode dropdown in
the UI (slot index assigned in `dispatch.wgsl`).
