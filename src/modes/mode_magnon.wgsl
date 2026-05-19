// ── Mode 39: MAGNON — spin-wave dispersion + Stoner particle–hole continuum in the (q,ω) plane ──

// Hue rotation around the (1,1,1) axis — used for color_shift on the magnon line.
fn magnon_hue_shift(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k  = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

// Ferromagnetic single-ion-anisotropy magnon: ω_FM(q) = √(Δ² + (D q²)²).
fn magnon_omega_fm(q: f32, gap: f32, stiff: f32) -> f32 {
    let dq2 = stiff * q * q;
    return sqrt(gap * gap + dq2 * dq2);
}

// Antiferromagnetic magnon: ω_AFM(q) = √(Δ² + (c q)²) — linear small-q.
fn magnon_omega_afm(q: f32, gap: f32, vel: f32) -> f32 {
    let cq = vel * q;
    return sqrt(gap * gap + cq * cq);
}

// Stoner particle–hole upper edge: ω₊(q) = D_S q² + Δ_ex + v_F q.
fn magnon_stoner_plus(q: f32, ds: f32, dex: f32, vf: f32) -> f32 {
    return ds * q * q + dex + vf * q;
}

// Stoner particle–hole lower edge: max(D_S q² + Δ_ex - v_F q, Δ_ex).
fn magnon_stoner_minus(q: f32, ds: f32, dex: f32, vf: f32) -> f32 {
    return max(ds * q * q + dex - vf * q, dex);
}

// Lorentzian intensity (per π).
fn magnon_lorentz(dw: f32, gamma: f32) -> f32 {
    let g = max(gamma, 0.0005);
    return (1.0 / 3.14159265) * g / (dw * dw + g * g);
}

fn render_magnon(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed;

    // Screen mapping: x → q ∈ [0, 1], y → ω ∈ [0, ~3] (zero at bottom).
    let q     = (uv.x / max(u.aspect, 0.0001)) * 0.5 + 0.5;
    let omega = (1.0 - uv.y) * 0.5 * 3.0 / max(u.zoom, 0.05);

    // ── Magnon dispersion parameters ──
    let gap   = 0.05 + 0.7 * u.iso_level;     // anisotropy Δ
    let stiff = 1.2  + 2.0 * u.kscale * 0.5;  // FM stiffness D
    let vel   = 1.0  + 1.6 * u.kscale * 0.5;  // AFM velocity c

    let om_fm  = magnon_omega_fm(q, gap, stiff);
    let om_afm = magnon_omega_afm(q, gap, vel);
    // field_mix interpolates FM (q²) ↔ AFM (linear-q).
    let om_m   = mix(om_fm, om_afm, clamp(u.field_mix, 0.0, 1.0));

    // ── Stoner continuum parameters ──
    let d_ex = 0.35 + 0.35 * u.field_mix;     // exchange splitting onset
    let d_s  = 0.6;                            // single-particle stiffness
    let v_f  = 1.6  + 1.2 * u.kscale * 0.5;   // Fermi velocity

    let om_sp = magnon_stoner_plus (q, d_s, d_ex, v_f);
    let om_sm = magnon_stoner_minus(q, d_s, d_ex, v_f);
    let inside_cont = step(om_sm, omega) * step(omega, om_sp);

    // ── Landau damping: broaden line where magnon enters Stoner continuum ──
    var gamma = 0.006;
    let inside_line = step(om_sm, om_m) * step(om_m, om_sp);
    gamma += inside_line * u.w_band * (0.06 + 0.10 * smoothstep(0.0, 0.4, om_m - om_sm));
    // Magnon decay above 2Δ — extra (softer) broadening once kinematically allowed.
    gamma += smoothstep(2.0 * gap, 2.0 * gap + 0.2, om_m) * u.w_band * 0.03;

    // Lorentzian magnon intensity.
    let dw   = omega - om_m;
    let magI = magnon_lorentz(dw, gamma);

    // ── Background — deep indigo with a faint accent tint ──
    var col = vec3<f32>(0.010, 0.008, 0.026);
    col += u.crystal_color.xyz * 0.05;

    // ── Stoner continuum cloud — soft violet stippled glow ──
    let noise = 0.5 + 0.5 * sin(q * 38.0 + omega * 27.0 + t * 0.4);
    let cont_strength = clamp(u.w_motif, 0.0, 1.5);
    col += inside_cont
         * vec3<f32>(0.28, 0.14, 0.42)
         * (0.4 + 0.6 * noise) * 0.7 * cont_strength;

    // ── Magnon line: bright magenta-cyan tint ──
    var mag_col = vec3<f32>(0.45, 0.95, 1.00);
    mag_col = mix(mag_col, mag_col * u.crystal_color.xyz, 0.30);
    mag_col = magnon_hue_shift(mag_col, u.color_shift);

    // Scrolling shimmer along the magnon line — precession feel.
    let shimmer = 0.85 + 0.30 * sin(q * 16.0 - t * 2.2);
    col += magI * mag_col * 1.5 * shimmer;

    // ── Damping ghost where magnon line is buried in continuum ──
    let blur_w      = 0.09;
    let blur_kernel = exp(-(dw * dw) / (blur_w * blur_w));
    col += inside_line * blur_kernel
         * smoothstep(0.0, 0.35, om_m - om_sm)
         * vec3<f32>(0.95, 0.35, 0.85) * 0.55;

    // ── Faint white guide lines for continuum boundaries ──
    let guide_w = 0.005;
    let guide_plus  = smoothstep(guide_w, 0.0, abs(omega - om_sp));
    let guide_minus = smoothstep(guide_w, 0.0, abs(omega - om_sm));
    col += (guide_plus + guide_minus) * vec3<f32>(0.85, 0.90, 1.00) * 0.13;

    // ── Anisotropy-gap guide: dashed horizontal line at ω = Δ ──
    let gap_dash = 0.5 + 0.5 * sin(q * 60.0);
    col += smoothstep(0.004, 0.0, abs(omega - gap)) * gap_dash
         * vec3<f32>(0.95, 0.80, 0.40) * 0.30;

    // ── Faint axis ticks ──
    // Horizontal ω ticks near q=0 edge.
    let tick_w = 0.005;
    var omega_tick = 0.0;
    omega_tick += smoothstep(tick_w, 0.0, abs(omega - 0.5));
    omega_tick += smoothstep(tick_w, 0.0, abs(omega - 1.0));
    omega_tick += smoothstep(tick_w, 0.0, abs(omega - 1.5));
    omega_tick += smoothstep(tick_w, 0.0, abs(omega - 2.0));
    let edge_q = smoothstep(0.04, 0.0, q);
    col += omega_tick * edge_q * vec3<f32>(0.25, 0.27, 0.35);

    // Vertical q ticks near ω=0 edge.
    let qtick_w = 0.004;
    var q_tick  = 0.0;
    q_tick += smoothstep(qtick_w, 0.0, abs(q - 0.25));
    q_tick += smoothstep(qtick_w, 0.0, abs(q - 0.50));
    q_tick += smoothstep(qtick_w, 0.0, abs(q - 0.75));
    let edge_w = smoothstep(0.04, 0.0, omega);
    col += q_tick * edge_w * vec3<f32>(0.25, 0.27, 0.35);

    return col;
}
