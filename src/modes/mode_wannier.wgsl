// ── Mode 16: WANNIER — localized inverse-FT orbital |ψ|² isosurface ──

fn wannier_psi(r: vec3<f32>) -> vec2<f32> {
    var re = 0.0;
    var im = 0.0;
    for (var i = 0i; i < 64i; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let ph = textureLoad(g_tex, vec2<i32>(i, 1), 0).r;
        let G  = ga.xyz * u.kscale;
        let amp = ga.w;
        let arg = dot(G, r) + ph;
        re += amp * cos(arg);
        im += amp * sin(arg);
    }
    return vec2<f32>(re, im);
}

fn wannier_center() -> vec3<f32> {
    return vec3<f32>((u.field_mix - 0.5) * 1.5, 0.0, 0.0);
}

fn wannier_loc(r: vec3<f32>) -> vec2<f32> {
    let r_local = r - wannier_center();
    let envelope = exp(-dot(r_local, r_local) * 0.6);
    return wannier_psi(r_local) * envelope;
}

fn wannier_sdf(r: vec3<f32>) -> f32 {
    let psi = wannier_loc(r);
    let rho = psi.x*psi.x + psi.y*psi.y;
    return u.iso_level * 0.5 - sqrt(rho + 1e-6);
}

fn wannier_normal(r: vec3<f32>) -> vec3<f32> {
    let e = 0.0035;
    return normalize(vec3<f32>(
        wannier_sdf(r + vec3<f32>(e, 0.0, 0.0)) - wannier_sdf(r - vec3<f32>(e, 0.0, 0.0)),
        wannier_sdf(r + vec3<f32>(0.0, e, 0.0)) - wannier_sdf(r - vec3<f32>(0.0, e, 0.0)),
        wannier_sdf(r + vec3<f32>(0.0, 0.0, e)) - wannier_sdf(r - vec3<f32>(0.0, 0.0, e)),
    ));
}

fn render_wannier(uv: vec2<f32>) -> vec3<f32> {
    var az = u.time * u.speed * 0.15;
    var el = 0.4;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.5;
    }
    let cp  = vec3<f32>(sin(az)*cos(el), sin(el), cos(az)*cos(el)) * (3.0 / u.zoom);
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x*rgt + uv.y*up);

    var t     = 0.0;
    var hit   = false;
    var hit_p = vec3<f32>(0.0);
    var hit_i = 0;
    for (var i = 0; i < 80; i++) {
        let p = cp + rd * t;
        let d = wannier_sdf(p);
        if d < 0.0025 {
            hit   = true;
            hit_p = p;
            hit_i = i;
            break;
        }
        if t > 8.0 { break; }
        t += max(d, 0.005);
    }

    var col = vec3<f32>(0.005, 0.008, 0.025);

    if hit {
        let nm    = wannier_normal(hit_p);
        let psi   = wannier_loc(hit_p);
        let s     = psi.x;
        let mix_t = smoothstep(-0.06, 0.06, s);
        let neg_c = vec3<f32>(0.20, 0.45, 1.20);
        let pos_c = vec3<f32>(1.30, 0.30, 0.25);
        var surf  = mix(neg_c, pos_c, mix_t);
        surf = mix(surf, u.crystal_color.xyz, 0.18);

        let light = normalize(vec3<f32>(1.5, 2.0, 1.0));
        let diff  = max(dot(nm, light), 0.0);
        let rim   = pow(1.0 - abs(dot(nm, -rd)), 2.5);
        let rim_c = mix(vec3<f32>(0.30, 0.85, 0.95), vec3<f32>(1.0), 0.35);

        col = surf * (0.3 + 0.7 * diff)
            + rim_c * rim * 0.55;

        // nodal-surface highlight where sign s ≈ 0
        col += smoothstep(0.04, 0.0, abs(s)) * vec3<f32>(0.95, 0.98, 1.10) * 0.45;
        col *= 0.5 + 0.5 * (1.0 - f32(hit_i) / 80.0);
    } else {
        // very dark navy with glow proportional to rho along the ray
        var glow = 0.0;
        for (var k = 0; k < 4; k++) {
            let tg  = 0.6 + f32(k) * 0.9;
            let psi = wannier_loc(cp + rd * tg);
            glow += psi.x*psi.x + psi.y*psi.y;
        }
        glow *= 0.25;
        col += u.crystal_color.xyz * glow * 0.35;
        col += vec3<f32>(0.20, 0.55, 1.00) * glow * 0.18;
    }

    return col;
}
