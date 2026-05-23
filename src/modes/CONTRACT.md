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
    time:          f32,    // seconds since start
    kscale:        f32,    // global scale on every G vector
    speed:         f32,    // animation rate multiplier
    field_mix:     f32,    // user slider, 0..1, semantic up to you
    iso_level:     f32,    // user slider, 0..1, semantic up to you
    color_shift:   f32,    // user slider, hue rotation
    zoom:          f32,    // user slider, 0.2..3 typical
    w_lattice:     f32,    // band weight (already used by crystal_field)
    w_motif:       f32,    // band weight
    w_band:        f32,    // band weight
    mode:          u32,    // current mode index
    num_g:         u32,    // number of valid entries in g_block (≤ 128)
    crystal_color: vec4<f32>,  // .xyz accent color of current crystal system
    mouse:         vec2<f32>,  // .x,.y in [0,1] when mouse_down >= 0.5
    mouse_down:    f32,        // 0 or 1
    aspect:        f32,        // viewport.x / viewport.y
}
@group(0) @binding(0) var<uniform> u: FU;

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
fn sdf(p: vec3<f32>) -> f32;                    // |mix(cf, cf2, field_mix)| - iso_level*0.3
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
