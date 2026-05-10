// ── Modes 0..12 — original visualizer modes ──────────────────────────────
// Extracted verbatim from the original FIELD_SHADER. Each `render_<name>`
// is dispatched from dispatch.wgsl via the `u.mode` index.

// ── Mode 0: ray-march isosurface ──────────────────────────────────────────
fn render_rm(uv: vec2<f32>) -> vec3<f32> {
    var az = u.time*u.speed*0.15; var el = 0.4;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.513;
    }
    let cp  = vec3<f32>(sin(az)*cos(el), sin(el), cos(az)*cos(el)) * (3.5/u.zoom);
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0,1,0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x*rgt + uv.y*up);
    var t   = 0.0;
    var col = vec3<f32>(0.01, 0.005, 0.02);
    for (var i = 0; i < 80; i++) {
        let p = cp + rd*t;
        let d = sdf(p);
        if d < 0.003 {
            let nm = calc_normal(p);
            col = cfield_col(crystal_field(p), cf2(p), nm) * (0.5 + 0.5*(1.0-f32(i)/80.0));
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
    let p  = vec3<f32>(uv*3.0, u.time*u.speed*0.3);
    let f  = crystal_field(p);
    let f2 = cf2(p);
    var c  = mix(vec3<f32>(0.02,0.01,0.04), u.crystal_color.xyz*0.6, smoothstep(-0.8,0.8,f));
    c += smoothstep(0.02, 0.0, abs(f))  * vec3<f32>(1.0,0.9,0.5) * 1.2;
    c += smoothstep(0.03, 0.0, abs(f - u.iso_level*0.5)) * u.crystal_color.xyz * 1.5;
    c += (0.5+0.5*sin(f*TAU*3.0)*cos(f2*TAU*2.0)) * 0.08 * u.crystal_color.xyz;
    return c;
}

// ── Mode 2: Fermi surface (scanning slice) ────────────────────────────────
fn render_fermi(uv: vec2<f32>) -> vec3<f32> {
    let z  = sin(u.time*u.speed*0.5)*0.8;
    let p  = vec3<f32>(uv*2.5/u.zoom, z);
    let f  = crystal_field(p); let f2 = cf2(p);
    var c  = vec3<f32>(0.01,0.005,0.025);
    c += smoothstep(0.025,0.0, abs(f  - u.iso_level*0.6)) * u.crystal_color.xyz * 2.0;
    c += smoothstep(0.02, 0.0, abs(f2 + u.iso_level*0.4)) * vec3<f32>(1.0,0.8,0.3) * 1.5;
    c += (0.5+0.5*sin(f*f2*30.0 + u.time*u.speed)) * u.crystal_color.xyz * 0.06;
    c += smoothstep(-1.0,1.0,f) * u.crystal_color.xyz * 0.3;
    return c;
}

// ── Mode 3: charge density ρ ∝ |ψ|² ─────────────────────────────────────
fn render_density(uv: vec2<f32>) -> vec3<f32> {
    var acc = 0.0;
    for (var di = 0; di < 6; di++) {
        let z  = (f32(di)/6.0 - 0.5)*2.0;
        let f  = crystal_field(vec3<f32>(uv*2.0/u.zoom, z));
        acc += f*f;
    }
    acc /= 6.0;
    let hue = fract(acc*1.5 + u.color_shift + u.time*u.speed*0.05);
    var col  = 0.45 + 0.45*cos(TAU*(hue + vec3<f32>(0.0,0.33,0.67)));
    col  = mix(col, u.crystal_color.xyz, 0.25) * pow(acc, 0.4) * 2.0;
    return col;
}

// ── Mode 4: nodal lines ───────────────────────────────────────────────────
fn render_nodal(uv: vec2<f32>) -> vec3<f32> {
    let p  = vec3<f32>(uv*2.5/u.zoom, u.time*u.speed*0.2);
    let f  = crystal_field(p); let f2 = cf2(p);
    let qc = select(vec3<f32>(1.0,0.5,0.2), u.crystal_color.xyz, sign(f)*sign(f2) > 0.0);
    var c  = vec3<f32>(0.01,0.005,0.02);
    c += qc * smoothstep(-1.0,1.0,f*f2) * 0.15;
    c += smoothstep(0.04,0.0, abs(f))   * vec3<f32>(0.9,0.95,1.0) * 1.8;
    c += smoothstep(0.03,0.0, abs(f2))  * u.crystal_color.xyz * 1.4;
    c += smoothstep(0.05,0.0, abs(f)+abs(f2)-0.06) * vec3<f32>(1.0,1.0,0.6) * 2.5;
    return c;
}

// ── Mode 5: volumetric emission cloud ────────────────────────────────────
fn render_cloud(uv: vec2<f32>) -> vec3<f32> {
    var az = u.time*u.speed*0.1;
    if u.mouse_down >= 0.5 { az = u.mouse.x * TAU; }
    let el  = 0.45;
    let cp  = vec3<f32>(sin(az)*cos(el), sin(el), cos(az)*cos(el)) * (3.0/u.zoom);
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0.0,1.0,0.0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x*rgt + uv.y*up);

    var col      = vec3<f32>(0.0);
    var transmit = 1.0;
    var t        = 0.0;
    let dt       = 0.09;
    for (var i = 0; i < 55; i++) {
        if t > 6.0 || transmit < 0.01 { break; }
        let p   = cp + rd * t;
        let f   = crystal_field(p);
        let f2  = cf2(p);
        let rho = max(mix(f*f, f2*f2, u.field_mix) - u.iso_level*0.05, 0.0);
        let hue = fract(f*1.5 + u.color_shift + u.time*u.speed*0.05);
        let ec  = 0.5 + 0.5*cos(TAU*(hue + vec3<f32>(0.0, 0.333, 0.667)));
        let op  = rho * dt * 3.5;
        col      += ec * op * transmit * mix(vec3<f32>(1.0), u.crystal_color.xyz, 0.6);
        transmit *= exp(-op * 2.5);
        t += dt;
    }
    return col;
}

// ── Mode 6: quantum phase portrait ───────────────────────────────────────
fn render_phase(uv: vec2<f32>) -> vec3<f32> {
    let z   = u.time * u.speed * 0.12;
    let p   = vec3<f32>(uv * 2.2 / u.zoom, z);
    let f   = crystal_field(p);
    let f2  = cf2(p);
    let amp = sqrt(f*f + f2*f2);
    let ang = atan2(f2, f);
    let hue = fract(ang / TAU + u.color_shift + u.time*u.speed*0.02);
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
    let p  = vec3<f32>(uv * 2.5 / u.zoom, u.time*u.speed*0.12);
    let f  = crystal_field(p);
    let f2 = cf2(p);
    let fm = mix(f, f2, u.field_mix);

    let n_lines  = 6.0 + u.iso_level * 14.0;
    let stripe   = fract(fm * n_lines);
    let line_w   = smoothstep(0.06, 0.0, min(stripe, 1.0 - stripe));

    let interf = 0.5 + 0.5*sin(f*TAU*2.5 + f2*TAU*1.7 + u.time*u.speed*0.4);

    let hue = fract(fm * 0.4 + u.color_shift + u.time*u.speed*0.04);
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
    let z  = u.time * u.speed * 0.1;
    let p0 = vec3<f32>(uv * 2.0 / u.zoom, z);
    let e  = 0.035;
    let gx = crystal_field(p0 + vec3<f32>(e, 0.0, 0.0))
           - crystal_field(p0 - vec3<f32>(e, 0.0, 0.0));
    let gy = crystal_field(p0 + vec3<f32>(0.0, e, 0.0))
           - crystal_field(p0 - vec3<f32>(0.0, e, 0.0));

    let warp   = (u.field_mix * 0.6 + 0.1);
    let pw     = vec3<f32>((uv + vec2<f32>(gx, gy) * warp) * 2.0 / u.zoom, z);
    let f      = crystal_field(pw);
    let f2     = cf2(pw);

    let hue = fract((f + f2)*0.5 + u.color_shift + u.time*u.speed*0.04);
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

// ── Mode 9: GLKITTY-style hollow sphere + crystal chain links ────────────
fn smin_links(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

fn sd_link(p: vec3<f32>, le: f32, r1: f32, r2: f32) -> f32 {
    let q = vec3<f32>(p.x, max(abs(p.y) - le, 0.0), p.z);
    return length(vec2<f32>(length(q.xy) - r1, q.z)) - r2;
}

fn links_scene(p: vec3<f32>) -> f32 {
    let noise  = crystal_field(p * 0.85) * 0.10 * max(1.0 - length(p) * 0.9, 0.0);
    let q      = p + noise;
    let sphere = abs(length(q) - 0.65) - 0.04;

    let cell_sz = 1.1;
    let cell_y  = floor(q.y / cell_sz + 0.5);
    let lp      = vec3<f32>(q.x, q.y - cell_y * cell_sz, q.z);
    var lq: vec3<f32>;
    if (i32(cell_y) & 1) != 0 {
        lq = vec3<f32>(lp.z, lp.y, lp.x);
    } else {
        lq = lp;
    }
    let link = sd_link(lq, 0.28, 0.22, 0.035);
    return smin_links(sphere, link, 0.18);
}

fn links_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.003;
    return normalize(vec3<f32>(
        links_scene(p + vec3<f32>(e, 0.0, 0.0)) - links_scene(p - vec3<f32>(e, 0.0, 0.0)),
        links_scene(p + vec3<f32>(0.0, e, 0.0)) - links_scene(p - vec3<f32>(0.0, e, 0.0)),
        links_scene(p + vec3<f32>(0.0, 0.0, e)) - links_scene(p - vec3<f32>(0.0, 0.0, e)),
    ));
}

fn render_links(uv: vec2<f32>) -> vec3<f32> {
    let t  = u.time * u.speed;
    var az = t * 0.22;
    var el = 0.38;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.2;
    }
    let cp  = vec3<f32>(sin(az)*cos(el), sin(el), cos(az)*cos(el)) * (3.2 / u.zoom);
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x*rgt + uv.y*up);

    let bg_f = crystal_field(rd * 1.8);
    var col  = mix(vec3<f32>(0.02, 0.01, 0.05),
                   vec3<f32>(0.04, 0.12, 0.18),
                   smoothstep(-0.6, 0.6, bg_f)) * 0.55;

    var ray_t = 0.0;
    for (var i = 0; i < 90; i++) {
        let p = cp + rd * ray_t;
        let d = links_scene(p);
        if d < 0.003 {
            let nm     = links_normal(p);
            let cf_val = crystal_field(p);
            let cf2v   = cf2(p);

            let hue  = fract(cf_val * 1.4 + cf2v * 0.5 + u.color_shift + t * 0.04);
            var surf = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
            surf = mix(surf, vec3<f32>(0.10, 0.72, 0.82), 0.38);
            surf = mix(surf, u.crystal_color.xyz, 0.22);

            let light = normalize(vec3<f32>(1.5, 2.0, 1.0));
            let diff  = max(dot(nm, light), 0.0);
            let rim   = pow(1.0 - abs(dot(nm, -rd)), 2.5);
            let spec  = pow(max(dot(reflect(-light, nm), -rd), 0.0), 32.0);

            col = surf * (0.3 + 0.7 * diff)
                + rim  * vec3<f32>(0.3, 0.8, 1.0) * 0.65
                + surf * spec * 0.50;
            col += cf_val * cf_val * vec3<f32>(0.2, 0.8, 1.0) * 0.40;
            col *= 0.35 + 0.65 * (1.0 - f32(i) / 90.0);
            break;
        }
        if ray_t > 6.5 { break; }
        ray_t += max(d, 0.003);
    }
    return col;
}

// ── Mode 10: simulated X-ray diffraction pattern ─────────────────────────
fn render_xrd(uv: vec2<f32>) -> vec3<f32> {
    let t  = u.time * u.speed * 0.06;
    var az = t;
    var el = 0.15;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.0;
    }
    let ca = cos(az); let sa = sin(az);
    let ce = cos(el); let se = sin(el);

    let sc    = 0.75 / u.zoom;
    let sigma = max(0.014 - u.iso_level * 0.011, 0.003);
    let ewald_k = 20.0 + u.iso_level * 60.0;

    var spots  = 0.0;
    var powder = 0.0;

    for (var i = 0; i < 128; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga  = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let amp = ga.w;
        let G   = ga.xyz * u.kscale;

        let gx  = G.x*ca - G.z*sa;
        let gyr = G.x*sa + G.z*ca;
        let gy  = G.y*ce - gyr*se;
        let gz  = G.y*se + gyr*ce;

        let ew = exp(-gz*gz * ewald_k);
        let sx = gx * sc;  let sy = gy * sc;
        let d2 = (uv.x - sx)*(uv.x - sx) + (uv.y - sy)*(uv.y - sy);
        spots += amp*amp * ew * exp(-d2 / (sigma*sigma));

        let G_r = length(G) * sc;
        let dr  = length(uv) - G_r;
        powder += amp*amp * exp(-dr*dr * 28000.0);
    }

    let hue_shift = u.color_shift + t * 0.03;
    let ring_col  = mix(vec3<f32>(0.55, 0.28, 0.04), u.crystal_color.xyz, 0.4);
    let spot_col  = mix(vec3<f32>(0.82, 0.93, 1.00), u.crystal_color.xyz, 0.25);

    var col = vec3<f32>(0.005, 0.003, 0.012);

    col += powder * ring_col * mix(0.5, 0.0, u.field_mix);
    col += spots  * spot_col * mix(0.4, 1.2, u.field_mix);

    let cf_bg = crystal_field(vec3<f32>(uv * 2.0 / u.zoom, t * 0.2)) * 0.015;
    col += cf_bg * u.crystal_color.xyz;

    let r = length(uv);
    col *= smoothstep(0.020, 0.036, r);
    col += exp(-r*r * 500.0) * vec3<f32>(1.0, 0.92, 0.70) * 0.25;
    col *= 1.0 - smoothstep(0.80, 0.98, r / (1.0 / u.zoom));

    return col;
}

// ── Mode 11: 3-D reciprocal-lattice point cloud ───────────────────────────
fn render_recip3d(uv: vec2<f32>) -> vec3<f32> {
    let t  = u.time * u.speed * 0.12;
    var az = t;
    var el = 0.30;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.5;
    }
    let cp  = vec3<f32>(sin(az)*cos(el), sin(el), cos(az)*cos(el)) * (4.5 / u.zoom);
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x*rgt + uv.y*up);

    let G_scale  = 0.42;
    let base_r   = mix(0.025, 0.10, u.iso_level);

    var best_t   = 1e9;
    var best_col = vec3<f32>(0.0);
    var got_hit  = false;

    {
        let r   = base_r * 1.4;
        let oc  = -cp;
        let b   = dot(oc, rd);
        let h   = b*b - dot(oc, oc) + r*r;
        if h >= 0.0 {
            let th = b - sqrt(h);
            if th > 0.0 {
                best_t   = th;
                got_hit  = true;
                let nm   = normalize(cp + rd*th);
                let diff = max(dot(nm, normalize(vec3<f32>(1.0, 2.0, 1.5))), 0.0);
                best_col = vec3<f32>(1.0, 0.97, 0.85) * (0.35 + 0.65*diff);
            }
        }
    }

    for (var i = 0; i < 128; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga  = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let amp = ga.w;
        let G   = ga.xyz * u.kscale * G_scale;

        let r   = base_r * (0.5 + amp);
        let oc  = G - cp;
        let b   = dot(oc, rd);
        let h   = b*b - dot(oc, oc) + r*r;
        if h < 0.0 { continue; }
        let th  = b - sqrt(h);
        if th <= 0.0 || th >= best_t { continue; }

        best_t  = th;
        got_hit = true;

        let p   = cp + rd * th;
        let nm  = normalize(p - G);

        let hue  = fract(amp*1.8 + length(G)*0.9 + u.color_shift + t*0.025);
        var surf = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
        surf = mix(surf, u.crystal_color.xyz, 0.30);

        let light = normalize(vec3<f32>(2.0, 3.0, 1.5));
        let diff  = max(dot(nm, light), 0.0);
        let spec  = pow(max(dot(reflect(-light, nm), -rd), 0.0), 28.0);
        let rim   = pow(1.0 - abs(dot(nm, -rd)), 2.2);

        best_col = surf * (0.25 + 0.75*diff)
                 + rim  * vec3<f32>(0.4, 0.75, 1.0) * 0.35
                 + spec * 0.55;
        best_col *= (0.4 + 0.6*amp);
    }

    var col = vec3<f32>(0.005, 0.003, 0.012);
    if got_hit { col = best_col; }

    for (var i = 0; i < 128; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga    = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let amp   = ga.w;
        let G     = ga.xyz * u.kscale * G_scale;
        let oc    = G - cp;
        let proj  = dot(oc, rd);
        if proj < 0.1 { continue; }
        let cl    = cp + rd*proj;
        let dist2 = dot(G - cl, G - cl);
        let gr    = base_r * (0.8 + amp) * 2.5;
        col += amp*amp * exp(-dist2 / (gr*gr)) * u.crystal_color.xyz * 0.12;
    }

    for (var i = 0; i < 128; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga   = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let amp  = ga.w;
        let G    = ga.xyz * u.kscale * G_scale;
        let oc   = -cp;
        let ab   = G;
        let ab2  = dot(ab, ab);
        let s    = clamp(dot(rd, ab) / max(ab2, 0.0001), 0.0, 1.0);
        let seg_pt = s * ab;
        let ray_t2 = dot(seg_pt - cp, rd);
        let ray_pt = cp + rd * max(ray_t2, 0.0);
        let d2    = dot(seg_pt - ray_pt, seg_pt - ray_pt);
        let spoke_w = 0.007 * amp;
        col += amp * exp(-d2 / (spoke_w*spoke_w)) * 0.06
             * mix(vec3<f32>(0.3,0.4,0.5), u.crystal_color.xyz, 0.5);
    }

    return col;
}

// ── Mode 12: non-Euclidean kaleidoscope (Kleinian circle inversions) ─────
fn circle_inv(z: vec2<f32>, c: vec2<f32>, r: f32) -> vec2<f32> {
    let d = z - c;
    return c + r * r * d / max(dot(d, d), 1e-15);
}

fn render_noneuclidean(uv: vec2<f32>) -> vec3<f32> {
    let t  = u.time * u.speed * 0.07;
    var az = t * 0.18;
    var el = 0.28;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.5;
    }
    let ca = cos(az); let sa = sin(az);
    let ce = cos(el); let se = sin(el);

    let sc = u.kscale * 0.40;
    var ctrs: array<vec2<f32>, 4>;
    var rads: array<f32, 4>;
    for (var i = 0u; i < 4u; i++) {
        let ga  = textureLoad(g_tex, vec2<i32>(i32(i), 0), 0);
        let G   = ga.xyz * sc;
        let gx  = G.x*ca - G.z*sa;
        let gyr = G.x*sa + G.z*ca;
        let gy  = G.y*ce - gyr*se;
        ctrs[i] = vec2<f32>(gx, gy);
        rads[i] = max(length(ctrs[i]) * (0.45 + u.iso_level * 0.35), 0.04);
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
        let hue  = fract(f32(ic) * 0.25 + f * 0.9 + f2v * 0.4 + u.color_shift + t * 0.04);
        var  tc  = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
        tc   = mix(tc, u.crystal_color.xyz, 0.28);
        col += tc * dcay * (0.08 + 0.45 * abs(f));
    }

    let f_fin  = crystal_field(vec3<f32>(z * 1.4, t * 0.18));
    let f2_fin = cf2(          vec3<f32>(z * 1.6, t * 0.22));
    let hue_f  = fract(f_fin * 1.1 + f2_fin * 0.5 + u.color_shift + t * 0.025);
    var  base  = 0.5 + 0.5 * cos(TAU * (hue_f + vec3<f32>(0.0, 0.333, 0.667)));
    base  = mix(base, u.crystal_color.xyz, 0.28);
    col   = col * 0.72 + base * (0.10 + 0.32 * abs(f_fin));

    return col;
}
