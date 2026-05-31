// Mode 51: TAMM/SHOCKLEY — surface state visualisation.
// Three panels in one screen:
//   Left: real-space slab cross-section showing evanescent wavefunction decay.
//   Right: (k∥, E) spectral function A(k,ω) — Dirac cone of the surface state
//          inside the bulk band gap (projected bulk bands shown as shadow).
//   Centre divider animates with field_mix.

// Evanescent surface-state wavefunction (Tamm model):
//   ψ(k∥, z) = C · e^(-z/ξ) · cos(k_perp · z + φ)
// where ξ = penetration depth, k_perp = bulk wave vector at gap edge.
fn tsss_psi(kpar: f32, z: f32, t: f32) -> f32 {
    // Penetration depth ξ: shrinks as iso_level → gap deepens → more localized
    let xi = 0.4 + 0.6 * (1.0 - mp(MP_ISO_LEVEL));
    let k_perp = 1.8 + 0.8 * mp(MP_KSCALE);
    let phi = kpar * 0.4 + t * 0.3;
    return exp(-z / max(xi, 0.01)) * cos(k_perp * z + phi);
}

// Surface energy (Dirac cone): E(k∥) = ħv_F |k∥|
// shifted by Dirac point position E_D (controlled by field_mix)
fn tsss_e_dirac(kpar: f32) -> f32 {
    let v_F = 0.9 + 0.6 * mp(MP_KSCALE);
    let e_D = (mp(MP_FIELD_MIX) - 0.5) * 0.6;   // Dirac point energy
    return v_F * abs(kpar) + e_D;
}

// Bulk band edge energies (projected gap): two parabolic bands
fn tsss_e_bulk_lower(kpar: f32) -> f32 {
    return -0.5 - 0.4 * kpar * kpar;
}
fn tsss_e_bulk_upper(kpar: f32) -> f32 {
    return 0.5 + 0.4 * kpar * kpar;
}

// Spectral function: Dirac-cone peak + Lorentzian broadened
fn tsss_spectral(kpar: f32, omega: f32) -> f32 {
    let e_ss = tsss_e_dirac(kpar);
    let gamma_ss = 0.04 + 0.06 * (1.0 - mp(MP_ISO_LEVEL));  // coherent when iso→1
    let A_ss = gamma_ss / (TAU * ((omega - e_ss) * (omega - e_ss) + gamma_ss * gamma_ss));

    // Bulk continuum (broad Lorentzian smear)
    let e_lo = tsss_e_bulk_lower(kpar);
    let e_hi = tsss_e_bulk_upper(kpar);
    let in_bulk = 1.0 - smoothstep(e_lo - 0.1, e_lo + 0.05, omega)
                      * smoothstep(e_hi - 0.05, e_hi + 0.1, omega);  // outside gap = bulk
    let gamma_bulk = 0.25;
    let A_bulk = in_bulk * 0.5 / (0.4 * (1.0 + ((omega - 0.0) / gamma_bulk) * ((omega - 0.0) / gamma_bulk)));

    return A_ss + A_bulk * 0.3;
}

fn render_tamm_shockley(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * mp(MP_SPEED) * 0.4;
    let zoom = max(mp(MP_ZOOM), 0.05);

    // Split: left = real-space, right = k-space spectral
    let split = -0.05 + mp(MP_FIELD_MIX) * 0.1 - 0.4;  // slightly left of centre by default
    let div_width = 0.04;

    var col = vec3<f32>(0.008, 0.010, 0.020);

    if uv.x < split - div_width {
        // ── LEFT PANEL: real-space evanescent wavefunction ────────────────
        // y axis = z (depth into crystal, 0 = surface at top → positive down)
        // x axis = position along surface
        let x_surf = uv.x / zoom;
        let z_depth = max(-(uv.y - 0.85) / zoom, 0.0);  // surface at y=0.85

        // Sample a few k∥ values and sum their contributions (density of states)
        var psi2_total = 0.0;
        let n_k = 6;
        for (var ik = 0; ik < n_k; ik++) {
            let kpar = (-0.5 + f32(ik) / f32(n_k - 1)) * 1.4;
            let psi = tsss_psi(kpar, z_depth, t);
            psi2_total += psi * psi / f32(n_k);
        }

        // Colour by |ψ|²: surface-warm (yellow) decaying to cool bulk (dark blue)
        let decay_frac = clamp(psi2_total * 2.5, 0.0, 1.0);
        let surface_col = mix(vec3<f32>(0.05, 0.07, 0.18), vec3<f32>(1.0, 0.85, 0.15), decay_frac);
        col = surface_col;

        // Crystal field background (bulk beyond penetration depth)
        let p3 = vec3<f32>(x_surf * 1.4, z_depth, t * 0.07);
        let cf = crystal_field(p3);
        let bulk_depth = smoothstep(0.1, 0.9, z_depth / zoom);
        col += u.crystal_color.xyz * (0.12 + 0.08 * cf) * bulk_depth;

        // Surface plane highlight
        let surf_plane = exp(-60.0 * (uv.y - 0.85) * (uv.y - 0.85));
        col += vec3<f32>(0.9, 0.95, 1.0) * surf_plane * 0.6;

        // Oscillating fringes along x (standing wave character)
        let kpar_probe = (x_surf - t * 0.1) * 0.9;
        let fringe = 0.5 + 0.5 * cos(kpar_probe * TAU * 2.0);
        col += vec3<f32>(0.3, 0.6, 1.0) * fringe * decay_frac * 0.35;

    } else if uv.x > split + div_width {
        // ── RIGHT PANEL: A(k∥, ω) spectral function ─────────────────────
        // x axis = k∥,  y axis = ω (energy, bottom=negative, top=positive)
        let kpar = uv.x / zoom * 1.5;
        let omega = uv.y / zoom * 1.2;

        let A = tsss_spectral(kpar, omega);

        // Bulk shadow band (grey projection)
        let e_lo = tsss_e_bulk_lower(kpar);
        let e_hi = tsss_e_bulk_upper(kpar);
        let in_gap = smoothstep(e_lo - 0.05, e_lo + 0.1, omega) * (1.0 - smoothstep(e_hi - 0.1, e_hi + 0.05, omega));
        let bulk_shadow = 1.0 - in_gap;
        col += vec3<f32>(0.12, 0.13, 0.17) * bulk_shadow;

        // Dirac cone — bright yellow-white line
        let e_dirac = tsss_e_dirac(kpar);
        let cone_line = exp(-120.0 * (omega - e_dirac) * (omega - e_dirac) * zoom * zoom);
        col += mix(vec3<f32>(1.0, 0.90, 0.20), u.crystal_color.xyz, 0.3) * cone_line * 1.5 * in_gap;

        // Full spectral weight (ARPES-like heatmap)
        let arpes_col = mix(vec3<f32>(0.0, 0.05, 0.2), vec3<f32>(1.0, 0.6, 0.0),
                             clamp(A * 1.8, 0.0, 1.0));
        col += arpes_col * clamp(A * 1.2, 0.0, 0.8);

        // Fermi level marker (horizontal line at ω = 0)
        let fermi = exp(-300.0 * omega * omega * zoom * zoom);
        col += vec3<f32>(0.3, 1.0, 0.5) * fermi * 0.5;

        // k=0 vertical marker
        let kzero = exp(-200.0 * kpar * kpar * zoom * zoom);
        col += vec3<f32>(0.3, 0.5, 1.0) * kzero * 0.3;

        // color_shift: hue of the cone
        let k3 = vec3<f32>(0.57735);
        let cs = cos(mp(MP_COLOR_SHIFT) * TAU);
        let sn = sin(mp(MP_COLOR_SHIFT) * TAU);
        col = col * cs + cross(k3, col) * sn + k3 * dot(k3, col) * (1.0 - cs);
    } else {
        // ── DIVIDER ───────────────────────────────────────────────────────
        col = vec3<f32>(0.25, 0.30, 0.45) * exp(-30.0 * (uv.x - split) * (uv.x - split) / (div_width * div_width));
    }

    return col;
}
