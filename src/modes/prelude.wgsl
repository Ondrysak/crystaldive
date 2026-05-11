// ── Crystal-field shader prelude ─────────────────────────────────────────
// Shared uniforms, bindings, vertex shader, and helpers used by every mode.
// Mode files (mode_*.wgsl) define `fn render_<name>(uv: vec2<f32>) -> vec3<f32>`
// and may use anything declared here. WGSL function order within a single
// module is flexible — modes can call each other or these helpers freely.

struct FU {
    time:          f32,
    kscale:        f32,
    speed:         f32,
    field_mix:     f32,
    iso_level:     f32,
    color_shift:   f32,
    zoom:          f32,
    w_lattice:     f32,
    w_motif:       f32,
    w_band:        f32,
    mode:          u32,
    num_g:         u32,
    crystal_color: vec4<f32>,
    mouse:         vec2<f32>,
    mouse_down:    f32,
    aspect:        f32,
}
@group(0) @binding(0) var<uniform> u: FU;
@group(0) @binding(1) var g_tex: texture_2d<f32>;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@vertex fn vs_screen(@builtin(vertex_index) vi: u32) -> VOut {
    var uvs = array<vec2<f32>,3>(vec2(0.0,0.0), vec2(2.0,0.0), vec2(0.0,2.0));
    let uv = uvs[vi];
    var o: VOut;
    o.pos = vec4<f32>(uv.x*2.0-1.0, 1.0-uv.y*2.0, 0.0, 1.0);
    o.uv  = uv;
    return o;
}

const TAU: f32 = 6.28318530718;

fn crystal_field(x: vec3<f32>) -> f32 {
    var v = 0.0; var ns = 0.0;
    let t = u.time * u.speed;
    for (var i = 0i; i < 64i; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga  = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let ph  = textureLoad(g_tex, vec2<i32>(i, 1), 0).r;
        let G   = ga.xyz * u.kscale;
        let amp = ga.w;
        let gx  = dot(G, x);
        let lat   = cos(gx + ph + t*(1.0 + f32(i)*0.01));
        let motif = cos(gx*1.13 + ph*1.7 + t*0.7);
        let band  = cos(gx + dot(G,G)*0.12 + t*0.4);
        v  += amp * (u.w_lattice*lat + u.w_motif*motif + u.w_band*band);
        ns += amp;
    }
    return v / max(ns, 0.001);
}

fn cf2(x: vec3<f32>) -> f32 {
    return crystal_field(x*1.37 + vec3<f32>(1.618, 2.718, 3.141));
}

fn sdf(p: vec3<f32>) -> f32 {
    return abs(mix(crystal_field(p), cf2(p), u.field_mix)) - u.iso_level*0.3;
}


// Crystal_field + cf2 in a single G-vector loop: saves loop overhead and shared
// per-G computations (dot(G,G), t*(1+i*0.01)) vs calling both functions separately.
fn sdf_combined(p: vec3<f32>) -> f32 {
    let p2 = p * 1.37 + vec3<f32>(1.618, 2.718, 3.141);
    var v1 = 0.0; var v2 = 0.0; var ns = 0.0;
    let t  = u.time * u.speed;
    for (var i = 0i; i < 64i; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga  = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let ph  = textureLoad(g_tex, vec2<i32>(i, 1), 0).r;
        let G   = ga.xyz * u.kscale;
        let amp = ga.w;
        let gg      = dot(G, G);
        let lat_t   = t * (1.0 + f32(i) * 0.01);
        let gx1 = dot(G, p);
        let gx2 = dot(G, p2);
        v1 += amp * (u.w_lattice * cos(gx1 + ph + lat_t)
                   + u.w_motif  * cos(gx1 * 1.13 + ph * 1.7 + t * 0.7)
                   + u.w_band   * cos(gx1 + gg * 0.12 + t * 0.4));
        v2 += amp * (u.w_lattice * cos(gx2 + ph + lat_t)
                   + u.w_motif  * cos(gx2 * 1.13 + ph * 1.7 + t * 0.7)
                   + u.w_band   * cos(gx2 + gg * 0.12 + t * 0.4));
        ns += amp;
    }
    let inv_ns = 1.0 / max(ns, 0.001);
    return abs(mix(v1 * inv_ns, v2 * inv_ns, u.field_mix)) - u.iso_level * 0.3;
}


// Tetrahedron normal (Inigo Quilez) — 4 sdf samples instead of 6.
fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e  = 0.004;
    let k0 = vec3<f32>( 1.0, -1.0, -1.0);
    let k1 = vec3<f32>(-1.0, -1.0,  1.0);
    let k2 = vec3<f32>(-1.0,  1.0, -1.0);
    let k3 = vec3<f32>( 1.0,  1.0,  1.0);
    return normalize(
        k0 * sdf(p + k0 * e) +
        k1 * sdf(p + k1 * e) +
        k2 * sdf(p + k2 * e) +
        k3 * sdf(p + k3 * e)
    );
}

// Returns vec3(value, dvalue/dx, dvalue/dy) for crystal_field in a single G-vector pass.
// Use instead of 4 finite-difference calls when the 2-D Jacobian is needed.
fn crystal_field_val_grad_xy(x: vec3<f32>) -> vec3<f32> {
    var v = 0.0; var ns = 0.0;
    var gx_acc = 0.0; var gy_acc = 0.0;
    let t = u.time * u.speed;
    for (var i = 0i; i < 64i; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga  = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let ph  = textureLoad(g_tex, vec2<i32>(i, 1), 0).r;
        let G   = ga.xyz * u.kscale;
        let amp = ga.w;
        let gx  = dot(G, x);
        let lat_arg   = gx + ph + t * (1.0 + f32(i) * 0.01);
        let motif_arg = gx * 1.13 + ph * 1.7 + t * 0.7;
        let band_arg  = gx + dot(G, G) * 0.12 + t * 0.4;
        v  += amp * (u.w_lattice * cos(lat_arg) + u.w_motif * cos(motif_arg) + u.w_band * cos(band_arg));
        ns += amp;
        let ds = u.w_lattice * sin(lat_arg) + u.w_motif * sin(motif_arg) * 1.13 + u.w_band * sin(band_arg);
        gx_acc -= amp * G.x * ds;
        gy_acc -= amp * G.y * ds;
    }
    let inv_ns = 1.0 / max(ns, 0.001);
    return vec3<f32>(v * inv_ns, gx_acc * inv_ns, gy_acc * inv_ns);
}

fn cfield_col(f: f32, f2: f32, n: vec3<f32>) -> vec3<f32> {
    let hue = fract(f*2.0 + f2*0.7 + u.color_shift + u.time*u.speed*0.1);
    var col  = 0.5 + 0.5*cos(TAU*(hue + vec3<f32>(0.0, 0.333, 0.667)));
    col = mix(col, u.crystal_color.xyz, 0.3);
    col += pow(1.0 - abs(n.z), 3.0) * 0.4 * vec3<f32>(0.6, 0.4, 1.0);
    col = mix(col, col*0.4 + vec3<f32>(0.6,0.8,1.0)*(0.5+0.5*sin(f*20.0+f2*15.0))*0.6, 0.25);
    return col;
}
