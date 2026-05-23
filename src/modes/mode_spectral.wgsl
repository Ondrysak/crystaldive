// ── Mode 34: SPECTRAL — ARPES-style A(ω,k) Lorentzian-broadened band map ──

// Hue rotation around the (1,1,1) axis — used so color_shift can rotate the
// warm/cool axis of the ARPES tint without changing brightness.
fn spectral_hue_shift(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k  = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

// Σ amp_i · cos(G_i · K + ph_i * use_phase) — local copy so we don't touch
// dispersion's helpers. use_phase=0 → first band, =1 → phase-shifted band.
fn spectral_eps(K: vec3<f32>, use_phase: f32) -> f32 {
    var e = 0.0;
    let ng_sp = i32(u.num_g);
    for (var i = 0; i < ng_sp; i++) {
        let ga  = g_block.gamp[i];
        let G   = ga.xyz;
        let amp = ga.w;
        let ph  = g_block.phases[i].x;
        e += amp * cos(dot(G, K) + ph * use_phase);
    }
    return e;
}

// Map k_param ∈ [0,1] onto the Γ → X → M → Γ piecewise-linear path.
fn spectral_path_K(k_param: f32) -> vec3<f32> {
    var K1: vec3<f32>;
    var K2: vec3<f32>;
    if (u.num_g >= 2u) {
        let g0 = g_block.gamp[0].xyz;
        let g1 = g_block.gamp[1].xyz;
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

// A(ω, k) = (1/π) · Σ / ((ω - ε)² + Σ²), single-pole Lorentzian per band,
// scaled by an external spectral weight.
fn spectral_lorentz(omega: f32, eps: f32, sig: f32, weight: f32) -> f32 {
    let s = max(sig, 0.005);
    return weight * s / (TAU * 0.5 * ((omega - eps)*(omega - eps) + s*s));
}

// ARPES palette: dark navy → blue → white → warm white at brightest.
fn spectral_palette(t: f32) -> vec3<f32> {
    let s = clamp(t * 0.5, 0.0, 1.5);
    return mix(vec3<f32>(0.02, 0.02, 0.08),
               mix(vec3<f32>(0.10, 0.30, 0.85),
                   vec3<f32>(1.05, 1.00, 0.92), smoothstep(0.5, 1.5, s)),
               smoothstep(0.0, 0.5, s));
}

fn render_spectral(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed;

    // Screen mapping: x → k path parameter, y → binding energy ω.
    let k_param = clamp((uv.x / max(u.aspect, 0.0001)) * 0.5 + 0.5, 0.0, 1.0);
    let omega   = uv.y * 2.0 / max(u.zoom, 0.05);

    // k-space sample point on Γ→X→M→Γ.
    let K = spectral_path_K(k_param);

    // Three band candidates: bare, phase-shifted, and a backfolded ghost.
    let eps_a = spectral_eps(K, 0.0);
    let eps_b = spectral_eps(K, 1.0);
    let eps1  = eps_a;
    let eps2  = eps_b;
    let eps3  = -eps1 + 0.5 * eps2;

    // Slow drift, just enough to keep the image alive.
    let e1 = eps1 + sin(t * 0.18) * 0.10;
    let e2 = eps2 + cos(t * 0.15) * 0.10;
    let e3 = eps3 + sin(t * 0.11) * 0.08;

    // Self-energy: Fermi-liquid-like ω² scattering. iso_level → coherent qp.
    var sigma = 0.04 + 0.30 * omega * omega;
    sigma *= (1.0 - 0.7 * u.iso_level);

    // A(ω, k) = sum of broadened poles with descending spectral weight.
    let A = spectral_lorentz(omega, e1, sigma, 1.0)
          + spectral_lorentz(omega, e2, sigma, 0.7)
          + spectral_lorentz(omega, e3, sigma, 0.4);

    // Log-compressed brightness (perceptual rolloff).
    let intensity = pow(max(A, 0.0), 0.7);

    // ARPES greyscale-with-warm-bias palette.
    var col = spectral_palette(intensity);

    // Apply color_shift as a hue rotation of the warm/cool axis.
    col = spectral_hue_shift(col, u.color_shift);

    // Tint slightly toward the crystal accent colour.
    col = mix(col, col * u.crystal_color.xyz, 0.18);

    // Faint vertical tick lines at high-symmetry k_param (Γ, X, M, Γ).
    let tick_w = 0.0035;
    var tick   = 0.0;
    tick += smoothstep(tick_w, 0.0, abs(k_param - 0.0));
    tick += smoothstep(tick_w, 0.0, abs(k_param - 1.0 / 3.0));
    tick += smoothstep(tick_w, 0.0, abs(k_param - 2.0 / 3.0));
    tick += smoothstep(tick_w, 0.0, abs(k_param - 1.0));
    col  += tick * vec3<f32>(0.06, 0.07, 0.10);

    // Faint horizontal Fermi-energy line at ω = 0.
    col += smoothstep(0.0035, 0.0, abs(uv.y)) * vec3<f32>(0.08, 0.09, 0.12);

    // Subtle horizontal detector noise — only visible in dim regions.
    let noise = (0.5 + 0.5 * sin(uv.x * 200.0 + uv.y * 97.0)) * 0.02;
    col += vec3<f32>(noise) * (1.0 - smoothstep(0.0, 0.6, intensity));

    return col;
}
