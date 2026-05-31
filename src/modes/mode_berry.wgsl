// ── Mode 31: BERRY — Berry curvature Ω(k) heatmap with connection streamlines ──

// Hue rotation around the grey axis — used to spin the divergent palette.
fn berry_hue_shift(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k  = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

// Two-band Hamiltonian vector h(k) = (h1, h2, h3). h3 carries the mass term
// driven by field_mix; field_mix ≈ 0.5 sits at the topological transition.
fn berry_h(k: vec2<f32>) -> vec3<f32> {
    let h1 = crystal_field(vec3<f32>(k, 0.0));
    let h2 = cf2(vec3<f32>(k, 0.0));
    let h3 = mp(MP_FIELD_MIX) * 2.0 - 1.0
           + 0.3 * crystal_field(vec3<f32>(k * 1.5, 0.7));
    return vec3<f32>(h1, h2, h3);
}

// Normalized Bloch vector d̂(k) = h(k) / |h(k)|, with |h| clamped above 1e-3.
fn berry_d_hat(k: vec2<f32>) -> vec3<f32> {
    let h = berry_h(k);
    let l = max(length(h), 1e-3);
    return h / l;
}

// Berry curvature Ω(k) = ½ d̂ · (∂_x d̂ × ∂_y d̂) via forward differences.
// Reuses the caller-supplied d0 = berry_d_hat(k), cutting 3 of 5 berry_d_hat calls
// compared to the original central-difference version (12/18 fewer crystal_field calls).
fn berry_omega(k: vec2<f32>, d0: vec3<f32>) -> f32 {
    let e   = 0.04;
    let dxp = berry_d_hat(k + vec2<f32>(e, 0.0));
    let dyp = berry_d_hat(k + vec2<f32>(0.0, e));
    let dx  = (dxp - d0) * (1.0 / e);
    let dy  = (dyp - d0) * (1.0 / e);
    return 0.5 * dot(d0, cross(dx, dy));
}

fn render_berry(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * mp(MP_SPEED);

    // Map screen → 2D Brillouin-zone coordinate.
    let k = uv * 2.0 / max(mp(MP_ZOOM), 0.05);

    // Sample the Hamiltonian and its curvature; reuse d0 inside berry_omega.
    let h_vec = berry_h(k);
    let h_mag = length(h_vec);
    let d0    = h_vec / max(h_mag, 1e-3);
    let omega = berry_omega(k, d0);

    // Divergent palette around 0: blue for negative Ω, orange for positive.
    let neg_col = vec3<f32>(0.05, 0.55, 0.95);
    let pos_col = vec3<f32>(1.00, 0.55, 0.10);
    let pal_neg = berry_hue_shift(neg_col, mp(MP_COLOR_SHIFT));
    let pal_pos = berry_hue_shift(pos_col, mp(MP_COLOR_SHIFT));

    let sat   = tanh(abs(omega) * 3.0);
    let sgn   = step(0.0, omega);                  // 1 if Ω ≥ 0
    let chosen = mix(pal_neg, pal_pos, sgn);
    var heat   = mix(u.crystal_color.xyz, chosen, 0.80);  // tint toward 0.20 of crystal
    heat       = heat * sat;

    // Dark background, faintly tinted by the crystal accent.
    let bg = mix(vec3<f32>(0.010, 0.006, 0.022),
                 u.crystal_color.xyz * 0.05,
                 0.5);
    var col = bg + heat;

    // Streamlines of the Berry connection — proxy: in-plane (h1, h2) flow.
    let in_plane_len = sqrt(h_vec.x * h_vec.x + h_vec.y * h_vec.y);
    let theta = atan2(h_vec.y, h_vec.x);
    let s     = uv.x * cos(theta) + uv.y * sin(theta);
    let freq  = 30.0 + 90.0 * mp(MP_ISO_LEVEL);          // higher iso → thinner streaks
    let stream = 0.5 + 0.5 * sin(s * freq + t * 0.6);
    let sharp  = 0.50 + 0.30 * mp(MP_ISO_LEVEL);         // sharpness ramp
    let dash   = smoothstep(sharp, sharp + 0.18, stream);

    // Streak color flips with local heatmap brightness so streamlines stay readable.
    let lum = dot(col, vec3<f32>(0.30, 0.59, 0.11));
    let streak_color = mix(vec3<f32>(1.00, 0.97, 0.85),
                           vec3<f32>(0.02, 0.02, 0.04),
                           smoothstep(0.20, 0.85, lum));
    col = mix(col, streak_color, dash * smoothstep(0.0, 0.6, in_plane_len) * 0.45);

    // Dirac-point glow: |h| → 0 lights up as a white halo.
    let dirac = exp(-h_mag * h_mag * 50.0);
    col += dirac * vec3<f32>(1.05, 1.00, 0.95) * 1.4;

    // Faint outer envelope so the BZ patch reads as a window onto k-space.
    let r = length(uv);
    col *= 1.0 - 0.18 * smoothstep(0.85, 1.30, r);

    return col;
}
