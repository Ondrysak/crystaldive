// ── Mode 18: DISPERSION — animated band-structure plot along Γ→X→M→Γ ──

// Hue rotation helper for color_shift on a base color.
fn dispersion_hue_shift(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k  = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

// Compute energy ε(K) = Σ amp_i · cos(G_i · K + phase_offset).
fn dispersion_eps(K: vec3<f32>, use_phase: f32) -> f32 {
    var e = 0.0;
    for (var i = 0; i < 64; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga  = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let G   = ga.xyz;
        let amp = ga.w;
        let ph  = textureLoad(g_tex, vec2<i32>(i, 1), 0).r;
        e += amp * cos(dot(G, K) + ph * use_phase);
    }
    return e;
}

// Map k_param ∈ [0,1] onto the Γ → X → M → Γ piecewise-linear path.
fn dispersion_path_K(k_param: f32) -> vec3<f32> {
    var K1: vec3<f32>;
    var K2: vec3<f32>;
    if (u.num_g >= 2u) {
        let g0 = textureLoad(g_tex, vec2<i32>(0, 0), 0).xyz;
        let g1 = textureLoad(g_tex, vec2<i32>(1, 0), 0).xyz;
        K1 = g0 * u.kscale * 0.5;
        K2 = (g0 + g1) * u.kscale * 0.5;
    } else {
        K1 = vec3<f32>(0.5, 0.0, 0.0) * u.kscale;
        K2 = vec3<f32>(0.5, 0.5, 0.0) * u.kscale;
    }
    let K0 = vec3<f32>(0.0);
    let K3 = vec3<f32>(0.0);

    if (k_param < 1.0 / 3.0) {
        return mix(K0, K1, k_param * 3.0);
    } else if (k_param < 2.0 / 3.0) {
        return mix(K1, K2, (k_param - 1.0 / 3.0) * 3.0);
    } else {
        return mix(K2, K3, (k_param - 2.0 / 3.0) * 3.0);
    }
}

fn render_dispersion(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed;

    // Map screen coords -> path parameter and energy axis.
    let k_param  = clamp((uv.x / max(u.aspect, 0.0001)) * 0.5 + 0.5, 0.0, 1.0);
    let eps_axis = uv.y * 2.0 / max(u.zoom, 0.05);

    // Sample path point and the two band candidates.
    let K         = dispersion_path_K(k_param);
    let eps_a     = dispersion_eps(K, 0.0);
    let eps_b     = dispersion_eps(K, 1.0);
    let eps1      = mix(eps_a, eps_b, u.field_mix);
    let eps2      = mix(eps_b, -eps_a, u.field_mix);

    // Slow time-evolution: drift bands in energy, gently breathing.
    let drift1 = sin(t * 0.20) * 0.12;
    let drift2 = cos(t * 0.17) * 0.12;
    let e1     = eps1 + drift1;
    let e2     = eps2 + drift2;

    // Background — very dark with a faint horizontal gradient.
    var col = vec3<f32>(0.012, 0.008, 0.030);
    col += vec3<f32>(0.020, 0.015, 0.040) * (0.5 - 0.5 * uv.y);

    // Faint vertical tick lines at high-symmetry k_param values (Γ, X, M, Γ).
    let tick_w = 0.004;
    var tick   = 0.0;
    tick += smoothstep(tick_w, 0.0, abs(k_param - 0.0));
    tick += smoothstep(tick_w, 0.0, abs(k_param - 1.0 / 3.0));
    tick += smoothstep(tick_w, 0.0, abs(k_param - 2.0 / 3.0));
    tick += smoothstep(tick_w, 0.0, abs(k_param - 1.0));
    col  += tick * vec3<f32>(0.18, 0.20, 0.28);

    // Faint horizontal zero-energy line.
    col += smoothstep(0.005, 0.0, abs(uv.y)) * vec3<f32>(0.20, 0.22, 0.28);

    // Band line width — iso_level controls sharpness (higher → thinner).
    let line_w = max(0.002, 0.012 * (1.0 - 0.7 * u.iso_level));

    // Scrolling glow that sweeps along the path over time.
    let highlight_k  = fract(t * 0.1);
    let dk           = abs(k_param - highlight_k);
    let dk_wrap      = min(dk, 1.0 - dk);
    let sweep_glow   = exp(-dk_wrap * dk_wrap * 80.0);

    // Band 1 — crystal_color hue.
    let band1_base = u.crystal_color.xyz;
    let band1_col  = dispersion_hue_shift(band1_base, u.color_shift);
    let band1_line = smoothstep(line_w, 0.0, abs(eps_axis - e1));
    col += band1_line * band1_col * (1.6 + 2.4 * sweep_glow);

    // Soft halo around band 1.
    let band1_halo = exp(-pow((eps_axis - e1) / (line_w * 6.0), 2.0));
    col += band1_halo * band1_col * 0.25;

    // Band 2 — complementary yellow-ish hue.
    let band2_base = vec3<f32>(1.0, 0.85, 0.35);
    let band2_col  = dispersion_hue_shift(band2_base, u.color_shift);
    let band2_line = smoothstep(line_w, 0.0, abs(eps_axis - e2));
    col += band2_line * band2_col * (1.4 + 2.2 * sweep_glow);

    let band2_halo = exp(-pow((eps_axis - e2) / (line_w * 6.0), 2.0));
    col += band2_halo * band2_col * 0.22;

    // Subtle vertical streak at the sweeping highlight position.
    col += sweep_glow * smoothstep(0.05, 0.0, abs(uv.y) - 1.0)
         * vec3<f32>(0.05, 0.06, 0.10);

    // Tiny crossing marker — when bands get close, sparkle the crossing.
    let gap     = abs(e1 - e2);
    let near_e  = exp(-pow((eps_axis - 0.5 * (e1 + e2)) / (line_w * 4.0), 2.0));
    col += near_e * exp(-gap * gap * 60.0) * vec3<f32>(1.0, 0.95, 0.7) * 0.6;

    return col;
}
