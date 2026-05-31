// ── Modes 0..12 — original visualizer modes ──────────────────────────────
// Extracted verbatim from the original FIELD_SHADER. Each `render_<name>`
// is dispatched from dispatch.wgsl via the `u.mode` index.

// ── Mode 0: ray-march isosurface ──────────────────────────────────────────
fn render_rm(uv: vec2<f32>) -> vec3<f32> {
    var az = u.time*mp(MP_SPEED)*0.15; var el = 0.4;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.513;
    }
    let cp  = vec3<f32>(sin(az)*cos(el), sin(el), cos(az)*cos(el)) * (3.5/mp(MP_ZOOM));
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0,1,0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x*rgt + uv.y*up);
    var t   = 0.0;
    var col = vec3<f32>(0.01, 0.005, 0.02);
    // sdf_combined merges crystal_field + cf2 in one G-vector loop per SDF call.
    for (var i = 0; i < 80; i++) {
        let p = cp + rd*t;
        let d = sdf_combined(p);
        if d < 0.003 {
            // Analytical 3-D gradient of sdf, inlined to avoid adding VGPR pressure
            // to all 36 modes via a shared prelude function.
            let cf_v  = crystal_field(p);
            let cf2_v = cf2(p);
            let mixed = mix(cf_v, cf2_v, mp(MP_FIELD_MIX));
            let sgn   = sign(mixed);
            let t_n   = u.time * mp(MP_SPEED);
            let p2    = p * 1.37 + vec3<f32>(1.618, 2.718, 3.141);
            var grad1 = vec3<f32>(0.0); var grad2 = vec3<f32>(0.0); var ns_n = 0.0;
            let ng_n  = i32(u.num_g);
            for (var j = 0i; j < ng_n; j++) {
                let ga_n = g_block.gamp[j];
                let ph_n = g_block.phases[j].x;
                let G_n  = ga_n.xyz * mp(MP_KSCALE);
                let gg_n = dot(G_n, G_n);
                let lt_n = t_n * (1.0 + f32(j) * 0.01);
                let gx1  = dot(G_n, p);   let gx2 = dot(G_n, p2);
                let ds1  = mp(MP_W_LATTICE) * sin(gx1 + ph_n + lt_n)
                         + mp(MP_W_MOTIF)  * sin(gx1*1.13 + ph_n*1.7 + t_n*0.7) * 1.13
                         + mp(MP_W_BAND)   * sin(gx1 + gg_n*0.12 + t_n*0.4);
                let ds2  = mp(MP_W_LATTICE) * sin(gx2 + ph_n + lt_n)
                         + mp(MP_W_MOTIF)  * sin(gx2*1.13 + ph_n*1.7 + t_n*0.7) * 1.13
                         + mp(MP_W_BAND)   * sin(gx2 + gg_n*0.12 + t_n*0.4);
                grad1 -= ga_n.w * G_n * ds1;
                grad2 -= ga_n.w * G_n * ds2;
                ns_n  += ga_n.w;
            }
            let inv_ns = 1.0 / max(ns_n, 0.001);
            let g1  = grad1 * inv_ns;
            let g2  = grad2 * (inv_ns * 1.37);
            let nm  = normalize(sgn * ((1.0 - mp(MP_FIELD_MIX)) * g1 + mp(MP_FIELD_MIX) * g2)
                              + vec3<f32>(1e-12));
            col = cfield_col(cf_v, cf2_v, nm) * (0.5 + 0.5*(1.0-f32(i)/80.0));
            break;
        }
        if t > 8.0 { break; }
        t += max(d, 0.005);
    }
    col += u.crystal_color.xyz * max(crystal_field(cp + rd*4.0)*0.3, 0.0) * 0.08;
    return col;
}

// ── Mode 1: BZ slice ──────────────────────────────────────────────────────
fn render_bz(uv: vec2<f32>) -> vec3<f32> {
    let p  = vec3<f32>(uv*3.0, u.time*mp(MP_SPEED)*0.3);
    let f  = crystal_field(p);
    let f2 = cf2(p);
    var c  = mix(vec3<f32>(0.02,0.01,0.04), u.crystal_color.xyz*0.6, smoothstep(-0.8,0.8,f));
    c += smoothstep(0.02, 0.0, abs(f))  * vec3<f32>(1.0,0.9,0.5) * 1.2;
    c += smoothstep(0.03, 0.0, abs(f - mp(MP_ISO_LEVEL)*0.5)) * u.crystal_color.xyz * 1.5;
    c += (0.5+0.5*sin(f*TAU*3.0)*cos(f2*TAU*2.0)) * 0.08 * u.crystal_color.xyz;
    return c;
}

// ── Mode 2: Fermi surface (scanning slice) ────────────────────────────────
fn render_fermi(uv: vec2<f32>) -> vec3<f32> {
    let z  = sin(u.time*mp(MP_SPEED)*0.5)*0.8;
    let p  = vec3<f32>(uv*2.5/mp(MP_ZOOM), z);
    let f  = crystal_field(p); let f2 = cf2(p);
    var c  = vec3<f32>(0.01,0.005,0.025);
    c += smoothstep(0.025,0.0, abs(f  - mp(MP_ISO_LEVEL)*0.6)) * u.crystal_color.xyz * 2.0;
    c += smoothstep(0.02, 0.0, abs(f2 + mp(MP_ISO_LEVEL)*0.4)) * vec3<f32>(1.0,0.8,0.3) * 1.5;
    c += (0.5+0.5*sin(f*f2*30.0 + u.time*mp(MP_SPEED))) * u.crystal_color.xyz * 0.06;
    c += smoothstep(-1.0,1.0,f) * u.crystal_color.xyz * 0.3;
    return c;
}

// ── Mode 3: charge density ρ ∝ |ψ|² ─────────────────────────────────────
fn render_density(uv: vec2<f32>) -> vec3<f32> {
    var acc = 0.0;
    for (var di = 0; di < 6; di++) {
        let z  = (f32(di)/6.0 - 0.5)*2.0;
        let f  = crystal_field(vec3<f32>(uv*2.0/mp(MP_ZOOM), z));
        acc += f*f;
    }
    acc /= 6.0;
    let hue = fract(acc*1.5 + mp(MP_COLOR_SHIFT) + u.time*mp(MP_SPEED)*0.05);
    var col  = 0.45 + 0.45*cos(TAU*(hue + vec3<f32>(0.0,0.33,0.67)));
    col  = mix(col, u.crystal_color.xyz, 0.25) * pow(acc, 0.4) * 2.0;
    return col;
}

// ── Mode 4: nodal lines ───────────────────────────────────────────────────
fn render_nodal(uv: vec2<f32>) -> vec3<f32> {
    let p  = vec3<f32>(uv*2.5/mp(MP_ZOOM), u.time*mp(MP_SPEED)*0.2);
    let f  = crystal_field(p); let f2 = cf2(p);
    let qc = select(vec3<f32>(1.0,0.5,0.2), u.crystal_color.xyz, sign(f)*sign(f2) > 0.0);
    var c  = vec3<f32>(0.01,0.005,0.02);
    c += qc * smoothstep(-1.0,1.0,f*f2) * 0.15;
    c += smoothstep(0.04,0.0, abs(f))   * vec3<f32>(0.9,0.95,1.0) * 1.8;
    c += smoothstep(0.03,0.0, abs(f2))  * u.crystal_color.xyz * 1.4;
    c += smoothstep(0.05,0.0, abs(f)+abs(f2)-0.06) * vec3<f32>(1.0,1.0,0.6) * 2.5;
    return c;
}

// ── Mode 6: quantum phase portrait ───────────────────────────────────────
fn render_phase(uv: vec2<f32>) -> vec3<f32> {
    let z   = u.time * mp(MP_SPEED) * 0.12;
    let p   = vec3<f32>(uv * 2.2 / mp(MP_ZOOM), z);
    let f   = crystal_field(p);
    let f2  = cf2(p);
    let amp = sqrt(f*f + f2*f2);
    let ang = atan2(f2, f);
    let hue = fract(ang / TAU + mp(MP_COLOR_SHIFT) + u.time*mp(MP_SPEED)*0.02);
    var col = 0.5 + 0.5*cos(TAU*(hue + vec3<f32>(0.0, 0.333, 0.667)));
    col *= smoothstep(0.0, 0.3, amp) * 1.6;
    let core = 1.0 - smoothstep(0.0, 0.06, amp);
    col += core * vec3<f32>(1.2, 1.3, 2.0);
    col += smoothstep(0.02, 0.0, abs(f))  * vec3<f32>(1.0, 0.9, 0.5) * 0.9;
    col += smoothstep(0.02, 0.0, abs(f2)) * vec3<f32>(0.5, 0.9, 1.0) * 0.9;
    col = mix(vec3<f32>(0.01, 0.005, 0.025), col, smoothstep(0.0, 0.15, amp));
    return col;
}

// ── Mode 7: quantised Fermi stripes (DOS) ────────────────────────────────
fn render_stripes(uv: vec2<f32>) -> vec3<f32> {
    let p  = vec3<f32>(uv * 2.5 / mp(MP_ZOOM), u.time*mp(MP_SPEED)*0.12);
    let f  = crystal_field(p);
    let f2 = cf2(p);
    let fm = mix(f, f2, mp(MP_FIELD_MIX));

    let n_lines  = 6.0 + mp(MP_ISO_LEVEL) * 14.0;
    let stripe   = fract(fm * n_lines);
    let line_w   = smoothstep(0.06, 0.0, min(stripe, 1.0 - stripe));

    let interf = 0.5 + 0.5*sin(f*TAU*2.5 + f2*TAU*1.7 + u.time*mp(MP_SPEED)*0.4);

    let hue = fract(fm * 0.4 + mp(MP_COLOR_SHIFT) + u.time*mp(MP_SPEED)*0.04);
    var col = 0.5 + 0.5*cos(TAU*(hue + vec3<f32>(0.0, 0.333, 0.667)));
    col = mix(vec3<f32>(0.01,0.005,0.02),
              col * mix(vec3<f32>(1.0), u.crystal_color.xyz, 0.55),
              smoothstep(-0.7, 0.7, fm));
    col += line_w * vec3<f32>(1.0, 0.95, 0.7) * 2.2;
    col += interf * u.crystal_color.xyz * 0.07;
    return col;
}

// ── Mode 8: gradient-warp lensing ────────────────────────────────────────
fn render_warp(uv: vec2<f32>) -> vec3<f32> {
    let z  = u.time * mp(MP_SPEED) * 0.1;
    let p0 = vec3<f32>(uv * 2.0 / mp(MP_ZOOM), z);
    // Analytical gradient replaces 4 finite-difference crystal_field calls.
    // Scale by 2*e to match the central-difference magnitude that the warp factor expects.
    let e   = 0.035;
    let vg  = crystal_field_val_grad_xy(p0);
    let gx  = vg.y * (2.0 * e);
    let gy  = vg.z * (2.0 * e);

    let warp   = (mp(MP_FIELD_MIX) * 0.6 + 0.1);
    let pw     = vec3<f32>((uv + vec2<f32>(gx, gy) * warp) * 2.0 / mp(MP_ZOOM), z);
    let f      = crystal_field(pw);
    let f2     = cf2(pw);

    let hue = fract((f + f2)*0.5 + mp(MP_COLOR_SHIFT) + u.time*mp(MP_SPEED)*0.04);
    var col = 0.5 + 0.5*cos(TAU*(hue + vec3<f32>(0.0, 0.333, 0.667)));
    col = mix(vec3<f32>(0.01, 0.005, 0.02),
              col * mix(vec3<f32>(1.0), u.crystal_color.xyz, 0.5),
              smoothstep(-1.0, 1.0, f));
    col += smoothstep(0.018, 0.0, abs(f))  * vec3<f32>(1.0, 0.9, 0.5) * 1.4;
    col += smoothstep(0.022, 0.0, abs(f2)) * u.crystal_color.xyz * 1.2;
    let grad_mag = sqrt(gx*gx + gy*gy);
    col += grad_mag * u.crystal_color.xyz * 0.3;
    return col;
}

// ── Mode 12: non-Euclidean kaleidoscope (Kleinian circle inversions) ─────
fn circle_inv(z: vec2<f32>, c: vec2<f32>, r: f32) -> vec2<f32> {
    let d = z - c;
    return c + r * r * d / max(dot(d, d), 1e-15);
}

fn render_noneuclidean(uv: vec2<f32>) -> vec3<f32> {
    let t  = u.time * mp(MP_SPEED) * 0.07;
    var az = t * 0.18;
    var el = 0.28;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.5;
    }
    let ca = cos(az); let sa = sin(az);
    let ce = cos(el); let se = sin(el);

    let sc = mp(MP_KSCALE) * 0.40;
    var ctrs: array<vec2<f32>, 4>;
    var rads: array<f32, 4>;
    for (var i = 0u; i < 4u; i++) {
        let ga  = g_block.gamp[i];
        let G   = ga.xyz * sc;
        let gx  = G.x*ca - G.z*sa;
        let gyr = G.x*sa + G.z*ca;
        let gy  = G.y*ce - gyr*se;
        ctrs[i] = vec2<f32>(gx, gy);
        rads[i] = max(length(ctrs[i]) * (0.45 + mp(MP_ISO_LEVEL) * 0.35), 0.04);
    }

    var z    = uv;
    var col  = vec3<f32>(0.0);
    var last = -1i;

    for (var iter = 0i; iter < 52i; iter++) {
        var inv_c = -1i;
        for (var ci = 0i; ci < 4i; ci++) {
            if ci != last && inv_c < 0i {
                let d2 = dot(z - ctrs[u32(ci)], z - ctrs[u32(ci)]);
                if d2 < rads[u32(ci)] * rads[u32(ci)] { inv_c = ci; }
            }
        }
        if inv_c < 0i { break; }

        let ic = u32(inv_c);
        z    = circle_inv(z, ctrs[ic], rads[ic]);
        last = inv_c;

        let f   = crystal_field(vec3<f32>(z,        t + f32(iter) * 0.08));
        let f2v = cf2(          vec3<f32>(z * 1.17, t + f32(iter) * 0.13));
        let dcay = exp(-f32(iter) * 0.050);
        let hue  = fract(f32(ic) * 0.25 + f * 0.9 + f2v * 0.4 + mp(MP_COLOR_SHIFT) + t * 0.04);
        var  tc  = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
        tc   = mix(tc, u.crystal_color.xyz, 0.28);
        col += tc * dcay * (0.08 + 0.45 * abs(f));
    }

    let f_fin  = crystal_field(vec3<f32>(z * 1.4, t * 0.18));
    let f2_fin = cf2(          vec3<f32>(z * 1.6, t * 0.22));
    let hue_f  = fract(f_fin * 1.1 + f2_fin * 0.5 + mp(MP_COLOR_SHIFT) + t * 0.025);
    var  base  = 0.5 + 0.5 * cos(TAU * (hue_f + vec3<f32>(0.0, 0.333, 0.667)));
    base  = mix(base, u.crystal_color.xyz, 0.28);
    col   = col * 0.72 + base * (0.10 + 0.32 * abs(f_fin));

    return col;
}
