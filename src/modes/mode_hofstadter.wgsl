// ── Mode 32: HOFSTADTER — fractal energy butterfly via Harper-like recursion ──

fn hofstadter_density(phi: f32, eps: f32) -> f32 {
    // Sum of cosine modulations whose density tracks the continued-fraction
    // expansion of phi — produces a self-similar pattern reminiscent of
    // Hofstadter's butterfly. Not a real Harper solver, but visually right.
    var d = 0.0;
    var p = phi;
    var amp = 1.0;
    let gate = mix(0.92, 0.78, clamp(mp(MP_ISO_LEVEL), 0.0, 1.0));
    for (var i = 0; i < 7; i++) {
        let ph_mod = p * f32(i + 1);
        let kshift = fract(ph_mod) * TAU;
        // Harper-like dispersion: ε ≈ 2 cos(k) + 2 cos(k + 2πp·n)
        let band1 = cos(eps * f32(i + 1) * 1.7 + kshift);
        let band2 = cos(eps * f32(i + 1) * 1.13 - kshift * 0.7);
        let band  = 0.5 * (band1 + band2);
        d += amp * smoothstep(gate, 1.0, band);
        // continued-fraction refinement of phi
        p = fract(1.0 / max(p, 0.005));
        amp *= 0.72;
    }
    return d;
}

fn hofstadter_axis_phi(phi: f32, x: f32) -> f32 {
    return smoothstep(0.002, 0.0, abs(phi - x));
}

fn render_hofstadter(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * mp(MP_SPEED);

    // Map screen → (phi, energy). phi ∈ [0,1], eps ∈ ~[-2,2].
    let phi = clamp((uv.x / u.aspect) * 0.5 + 0.5, 0.0, 1.0);
    let eps_raw = uv.y * 2.0;
    let eps_eff = eps_raw * (1.0 + (mp(MP_FIELD_MIX) - 0.5) * 0.4);

    let dens = hofstadter_density(phi, eps_eff);

    // Rainbow hue along φ axis, rotated by color_shift, gentle drift.
    let hue = fract(phi + mp(MP_COLOR_SHIFT) + t * 0.015);
    var rainbow = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
    rainbow = mix(rainbow, u.crystal_color.xyz, 0.35);

    // Background: near-black with faint crystal tint.
    var col = u.crystal_color.xyz * 0.04 + vec3<f32>(0.006, 0.004, 0.012);

    // Bright bands where density is high.
    let bright = pow(dens, 0.85) * 1.6;
    col += rainbow * bright;

    // Sharp highlights on the densest filaments.
    col += smoothstep(0.55, 1.2, dens) * vec3<f32>(1.0, 0.95, 0.75) * 0.9;

    // Faint axis labels: vertical lines at φ = 1/2, 1/3, 1/4, 1/5.
    var ax = 0.0;
    ax += hofstadter_axis_phi(phi, 0.5);
    ax += hofstadter_axis_phi(phi, 1.0 / 3.0);
    ax += hofstadter_axis_phi(phi, 0.25);
    ax += hofstadter_axis_phi(phi, 0.2);
    col += ax * vec3<f32>(0.55, 0.65, 0.75) * 0.35;

    // Horizontal line at ε = 0.
    let ax_e = smoothstep(0.004, 0.0, abs(eps_raw));
    col += ax_e * vec3<f32>(0.55, 0.65, 0.75) * 0.30;

    return col;
}
