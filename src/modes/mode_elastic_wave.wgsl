// Mode 48: ELASTIC WAVE — anisotropic acoustic wavefronts from a repeating
// impulse at the origin.  Phase velocity v(θ) derives from G-vector
// force-constant projections giving qL and qT branches.  Inner diagram shows
// the slowness surface (1/v vs θ).

// Force-constant longitudinal projection: C_L(n̂) = Σ aᵢ²(n̂·Ĝᵢ)²
fn ewave_c_long(n: vec2<f32>) -> f32 {
    let ng = i32(u.num_g);
    var s = 0.0;
    var w_total = 0.0;
    for (var i = 0; i < ng; i++) {
        let ga = g_block.gamp[i];
        let glen = length(ga.xy);
        if glen < 1e-6 { continue; }
        let proj = dot(n, ga.xy / glen);
        let w = ga.w * ga.w;
        s += w * proj * proj;
        w_total += w;
    }
    if w_total < 1e-9 { return 0.5; }
    return s / w_total;
}

fn ewave_phase_vel_L(theta: f32) -> f32 {
    let n = vec2<f32>(cos(theta), sin(theta));
    let cl = ewave_c_long(n);
    let v_hi = 0.75 + 0.35 * clamp(u.kscale * 0.5, 0.0, 1.0);
    let v_lo = 0.30 + 0.12 * clamp(u.kscale * 0.5, 0.0, 1.0);
    return mix(v_lo, v_hi, cl);
}

fn ewave_phase_vel_T(theta: f32) -> f32 {
    let n = vec2<f32>(cos(theta), sin(theta));
    let ct = 1.0 - ewave_c_long(n);
    let v_hi = 0.45 + 0.20 * clamp(u.kscale * 0.5, 0.0, 1.0);
    let v_lo = 0.15 + 0.06 * clamp(u.kscale * 0.5, 0.0, 1.0);
    return mix(v_lo, v_hi, ct);
}

fn render_elastic_wave(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed * 0.55;
    let zoom = max(u.zoom, 0.1);
    let p = uv / zoom;
    let r = length(p);
    let theta = atan2(p.y, p.x);

    var col = vec3<f32>(0.012, 0.016, 0.032);

    // 4 evenly-spaced pulses cycling on period T
    let period = 2.8;
    let sharp = 0.010 + 0.022 * (1.0 - u.field_mix);
    let show_T = u.field_mix;

    for (var pulse = 0; pulse < 4; pulse++) {
        // fract gives age ∈ [0, 1) for each pulse offset by 0.25
        let age_frac = fract(t / period + f32(pulse) * 0.25);
        let t_age = age_frac * period;
        let amp_decay = exp(-t_age * 0.25);

        // qL wavefront
        let v_L = ewave_phase_vel_L(theta);
        let r_L = v_L * t_age;
        let ring_L = amp_decay * exp(-((r - r_L) * (r - r_L)) / (sharp * sharp));

        // Phonon focusing: |dv/dθ| → caustic brightening
        let dth = 0.04;
        let dv_L = ewave_phase_vel_L(theta + dth) - ewave_phase_vel_L(theta - dth);
        let focus_L = 1.0 + 2.0 * (dv_L / dth) * (dv_L / dth);

        col += vec3<f32>(1.0, 0.70, 0.22) * ring_L * focus_L;

        // qT wavefront
        let v_T = ewave_phase_vel_T(theta);
        let r_T = v_T * t_age;
        let ring_T = amp_decay * 0.65 * exp(-((r - r_T) * (r - r_T)) / ((sharp * 1.3) * (sharp * 1.3)));
        let dv_T = ewave_phase_vel_T(theta + dth) - ewave_phase_vel_T(theta - dth);
        let focus_T = 1.0 + 2.0 * (dv_T / dth) * (dv_T / dth);

        col += mix(u.crystal_color.xyz, vec3<f32>(0.25, 0.55, 1.0), 0.6) * ring_T * focus_T * show_T;
    }

    // Slowness surface inset (small, centre)
    let s_scale = 0.18 / zoom;
    if r < s_scale * 2.5 {
        let v_L_s = ewave_phase_vel_L(theta);
        let v_T_s = ewave_phase_vel_T(theta);
        let r_slow_L = s_scale / max(v_L_s, 0.01);
        let r_slow_T = s_scale / max(v_T_s, 0.01);
        col += vec3<f32>(1.0, 0.8, 0.3) * exp(-300.0 * (r - r_slow_L) * (r - r_slow_L));
        col += vec3<f32>(0.35, 0.65, 1.0) * exp(-300.0 * (r - r_slow_T) * (r - r_slow_T)) * show_T;
    }

    // Source glow
    col += vec3<f32>(0.85, 0.92, 1.0) * exp(-90.0 * r * r / (zoom * zoom)) * 0.7;

    // iso_level: sector tint from crystal colour
    let sector = 0.5 + 0.5 * sin(theta * 4.0 + u.color_shift * TAU);
    col += u.crystal_color.xyz * sector * 0.05 * u.iso_level;

    return col;
}
