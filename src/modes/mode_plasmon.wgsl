// ── Mode 37: PLASMON — Lindhard particle-hole continuum + plasmon dispersion in the (q,ω) plane ──

// Hue rotation around the (1,1,1) axis — used for color_shift on the plasmon line.
fn plasmon_hue_shift(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k  = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

// Upper edge of the 2D Lindhard particle-hole continuum: ω = q + q².
fn plasmon_omega_plus(q: f32) -> f32 {
    return q + q * q;
}

// Lower edge of the 2D Lindhard particle-hole continuum: ω = max(q² - q, 0).
fn plasmon_omega_minus(q: f32) -> f32 {
    return max(q * q - q, 0.0);
}

// Plasmon dispersion ω_p(q) = sqrt(ω_p0² + α q²).
fn plasmon_dispersion(q: f32, omega_p0: f32, alpha: f32) -> f32 {
    return sqrt(omega_p0 * omega_p0 + alpha * q * q);
}

fn render_plasmon(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed;

    // Screen mapping: x → q ∈ [0, 1], y → ω ∈ [0, ~3] (zero at bottom).
    let q     = (uv.x / max(u.aspect, 0.0001)) * 0.5 + 0.5;
    let omega = (1.0 - uv.y) * 0.5 * 3.0 / max(u.zoom, 0.05);

    // Lindhard continuum boundaries.
    let om_plus  = plasmon_omega_plus(q);
    let om_minus = plasmon_omega_minus(q);
    let inside_continuum = step(om_minus, omega) * step(omega, om_plus);

    // Plasmon dispersion parameters — controlled by user sliders.
    let omega_p0 = 0.7 + 0.4 * u.field_mix;
    let alpha    = 0.6 + 0.5 * u.iso_level;
    let om_p     = plasmon_dispersion(q, omega_p0, alpha);

    // Landau damping: broaden plasmon line inside the continuum.
    var gamma = 0.005;
    if (om_p < om_plus) {
        gamma += 0.07 * smoothstep(0.0, 0.3, omega - om_minus);
    }

    // Lorentzian plasmon intensity (per π).
    let dw   = omega - om_p;
    let plas = (1.0 / 3.14159265) * gamma / (dw * dw + gamma * gamma);

    // ── Background — very dark blue with a faint crystal_color tint ──
    var col = vec3<f32>(0.008, 0.010, 0.028);
    col += u.crystal_color.xyz * 0.05;

    // ── Continuum cloud — soft greyish-blue stippled glow ──
    let noise = 0.5 + 0.5 * sin(q * 40.0 + omega * 30.0);
    col += inside_continuum
         * vec3<f32>(0.10, 0.25, 0.40)
         * (0.4 + 0.6 * noise) * 0.7;

    // ── Plasmon line: bright yellow-gold dispersion ──
    var plas_col = vec3<f32>(1.0, 0.85, 0.30);
    // Tint with crystal_color (mix 0.30).
    plas_col = mix(plas_col, plas_col * u.crystal_color.xyz, 0.30);
    // Hue-rotate by color_shift.
    plas_col = plasmon_hue_shift(plas_col, u.color_shift);

    // Subtle scrolling shimmer along the plasmon line — animated by u.time.
    let shimmer = 0.85 + 0.30 * sin(q * 18.0 - t * 2.5);

    col += plas * plas_col * 1.5 * shimmer;

    // ── Damping warm-orange "blur" where plasmon enters continuum ──
    let damp_region = step(om_p, om_plus); // 1 inside continuum
    let blur_w      = 0.08;
    let blur_kernel = exp(-(dw * dw) / (blur_w * blur_w));
    col += damp_region * blur_kernel
         * smoothstep(0.0, 0.3, omega - om_minus)
         * vec3<f32>(1.00, 0.55, 0.18) * 0.55;

    // ── Faint white guide lines for the continuum boundaries ──
    let guide_w = 0.005;
    let guide_plus  = smoothstep(guide_w, 0.0, abs(omega - om_plus));
    let guide_minus = smoothstep(guide_w, 0.0, abs(omega - om_minus));
    col += (guide_plus + guide_minus) * vec3<f32>(0.95, 0.97, 1.00) * 0.15;

    // ── Faint axis ticks ──
    // Horizontal ω ticks at 0.5, 1.0, 1.5, 2.0 — drawn near the q=0 edge.
    let tick_w = 0.005;
    var omega_tick = 0.0;
    omega_tick += smoothstep(tick_w, 0.0, abs(omega - 0.5));
    omega_tick += smoothstep(tick_w, 0.0, abs(omega - 1.0));
    omega_tick += smoothstep(tick_w, 0.0, abs(omega - 1.5));
    omega_tick += smoothstep(tick_w, 0.0, abs(omega - 2.0));
    let edge_q = smoothstep(0.04, 0.0, q);
    col += omega_tick * edge_q * vec3<f32>(0.25, 0.28, 0.35);

    // Vertical q ticks at 0.25, 0.5, 0.75 — drawn near the ω=0 edge.
    let qtick_w = 0.004;
    var q_tick  = 0.0;
    q_tick += smoothstep(qtick_w, 0.0, abs(q - 0.25));
    q_tick += smoothstep(qtick_w, 0.0, abs(q - 0.50));
    q_tick += smoothstep(qtick_w, 0.0, abs(q - 0.75));
    let edge_w = smoothstep(0.04, 0.0, omega);
    col += q_tick * edge_w * vec3<f32>(0.25, 0.28, 0.35);

    return col;
}
