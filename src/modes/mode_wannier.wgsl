// ── Mode 16: WANNIER — localized inverse-FT orbital |ψ|² isosurface ──

fn wannier_psi(r: vec3<f32>) -> vec2<f32> {
    var re = 0.0;
    var im = 0.0;
    let ng_wa = i32(u.num_g);
    for (var i = 0i; i < ng_wa; i++) {
        let ga  = g_block.gamp[i];
        let ph  = g_block.phases[i].x;
        let G   = ga.xyz * mp(MP_KSCALE);
        let amp = ga.w;
        let arg = dot(G, r) + ph;
        re += amp * cos(arg);
        im += amp * sin(arg);
    }
    return vec2<f32>(re, im);
}

fn wannier_center() -> vec3<f32> {
    return vec3<f32>((mp(MP_FIELD_MIX) - 0.5) * 1.5, 0.0, 0.0);
}

fn wannier_loc(r: vec3<f32>) -> vec2<f32> {
    let r_local  = r - wannier_center();
    let envelope = exp(-dot(r_local, r_local) * 0.6);
    return wannier_psi(r_local) * envelope;
}

fn wannier_sdf(r: vec3<f32>) -> f32 {
    let psi = wannier_loc(r);
    let rho = psi.x * psi.x + psi.y * psi.y;
    return mp(MP_ISO_LEVEL) * 0.5 - sqrt(rho + 1e-6);
}

fn render_wannier(uv: vec2<f32>) -> vec3<f32> {
    var az = u.time * mp(MP_SPEED) * 0.15;
    var el = 0.4;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.5;
    }
    let cp  = vec3<f32>(sin(az)*cos(el), sin(el), cos(az)*cos(el)) * (3.0 / mp(MP_ZOOM));
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x * rgt + uv.y * up);

    let ctr      = wannier_center();
    let cp_local = cp - ctr;

    var col = vec3<f32>(0.005, 0.008, 0.025);

    // Sphere pre-test: the Gaussian envelope exp(-0.6|rl|²) ensures the orbital
    // density is negligibly small outside radius 2.5 for any reasonable iso_level.
    // Rays that miss this sphere skip all 80 march steps entirely.
    let mid_t    = -dot(cp_local, rd);
    let closest  = cp_local + rd * mid_t;
    let close_sq = dot(closest, closest);
    if close_sq > 6.25 {   // 2.5² — never enters the orbital support sphere
        return col;
    }
    // Begin the march at the sphere entry — avoids empty space from camera to sphere.
    let enter_t = max(mid_t - sqrt(max(6.25 - close_sq, 0.0)), 0.0);

    var t     = enter_t;
    var hit   = false;
    var hit_p = vec3<f32>(0.0);
    var hit_i = 0;
    for (var i = 0; i < 80; i++) {
        let p = cp + rd * t;
        let d = wannier_sdf(p);
        if d < 0.0025 { hit = true; hit_p = p; hit_i = i; break; }
        if t > 8.0 { break; }
        t += max(d, 0.005);
    }

    if hit {
        // Combined normal + coloring in one G-vector pass.
        // Eliminates separate wannier_normal + wannier_loc calls (saves one full pass).
        let rl     = hit_p - ctr;
        let env    = exp(-dot(rl, rl) * 0.6);
        let env_sq = env * env;
        var re  = 0.0;  var im  = 0.0;
        var gre = vec3<f32>(0.0);
        var gim = vec3<f32>(0.0);
        let ng_wn = i32(u.num_g);
        for (var k = 0i; k < ng_wn; k++) {
            let ga  = g_block.gamp[k];
            let ph  = g_block.phases[k].x;
            let G   = ga.xyz * mp(MP_KSCALE);
            let amp = ga.w;
            let arg = dot(G, rl) + ph;
            let c   = cos(arg);
            let s   = sin(arg);
            re  += amp * c;
            im  += amp * s;
            gre -= amp * s * G;   // ∂re/∂r_local
            gim += amp * c * G;   // ∂im/∂r_local
        }

        // Analytical normal from density gradient.
        let psi_sq      = re * re + im * im;
        let grad_psi_sq = 2.0 * (re * gre + im * gim);
        let grad_rho    = env_sq * (grad_psi_sq - 2.4 * psi_sq * rl);
        let nm = normalize(-grad_rho + vec3<f32>(1e-12));

        // Coloring from the real part of wannier_loc.
        let s_val = re * env;
        let mix_t = smoothstep(-0.06, 0.06, s_val);
        let neg_c = vec3<f32>(0.20, 0.45, 1.20);
        let pos_c = vec3<f32>(1.30, 0.30, 0.25);
        var surf  = mix(neg_c, pos_c, mix_t);
        surf = mix(surf, u.crystal_color.xyz, 0.18);

        let light = normalize(vec3<f32>(1.5, 2.0, 1.0));
        let diff  = max(dot(nm, light), 0.0);
        let rim   = pow(1.0 - abs(dot(nm, -rd)), 2.5);
        let rim_c = mix(vec3<f32>(0.30, 0.85, 0.95), vec3<f32>(1.0), 0.35);
        col = surf * (0.3 + 0.7 * diff) + rim_c * rim * 0.55;
        col += smoothstep(0.04, 0.0, abs(s_val)) * vec3<f32>(0.95, 0.98, 1.10) * 0.45;
        col *= 0.5 + 0.5 * (1.0 - f32(hit_i) / 80.0);
    } else {
        // Glow: sample 4 points inside the sphere along the ray.
        var glow = 0.0;
        for (var k = 0; k < 4; k++) {
            let tg  = enter_t + 0.4 + f32(k) * 0.9;
            let psi = wannier_loc(cp + rd * tg);
            glow += psi.x * psi.x + psi.y * psi.y;
        }
        glow *= 0.25;
        col += u.crystal_color.xyz * glow * 0.35;
        col += vec3<f32>(0.20, 0.55, 1.00) * glow * 0.18;
    }

    return col;
}
