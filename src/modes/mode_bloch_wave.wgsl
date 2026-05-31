// ── Mode 36: BLOCH_WAVE — space-time wavepacket performing Bloch oscillations under a DC field ──

// Cosine-band dispersion ε(k) = -2·cos(k).
fn bloch_eps(k: f32) -> f32 {
    return -2.0 * cos(k);
}

// Semiclassical wavepacket centre under a DC field F:
//   x_c(t) = x_0 + (1/F) * (sin(k_0 + F·t) - sin(k_0))
// with x_0 = 0, k_0 = 0 → x_c(t) = sin(F·t)/F.
fn bloch_xc(t: f32, F: f32) -> f32 {
    let k0 = 0.0;
    let x0 = 0.0;
    return x0 + (sin(k0 + F * t) - sin(k0)) / max(F, 1e-4);
}

// Mild width breathing — Gaussian σ(t).
fn bloch_sigma(t: f32, F: f32) -> f32 {
    return 0.45 + 0.18 * sin(F * t * 0.5);
}

fn render_bloch_wave(uv: vec2<f32>) -> vec3<f32> {
    // Map screen-y to a time slice (older history downward, newest at top),
    // and scroll the whole pattern upward over wall-clock time.
    let zoom     = max(mp(MP_ZOOM), 0.1);
    let tau      = (1.0 - uv.y) * 0.5 * (8.0 / zoom);
    let t_global = u.time * mp(MP_SPEED) * 0.5;
    let t_eff    = tau + t_global;

    // Real-space x.
    let x = uv.x * 4.0 / zoom;

    // DC field strength controls Bloch period T_B = 2π / F.
    let F = 0.4 + mp(MP_FIELD_MIX) * 1.2;

    // Wavepacket centre and width at this (x, t) sample.
    let xc    = bloch_xc(t_eff, F);
    let sigma = bloch_sigma(t_eff, F);
    let dx    = x - xc;

    // Gaussian density ρ(x, t) ∝ exp(-(x-xc)² / 2σ²).
    let rho = exp(-(dx * dx) / max(2.0 * sigma * sigma, 1e-4));

    // Wave-like phase that modulates with time so the fringes look "alive".
    let phase = dx * (1.0 + 4.0 * sin(F * t_eff * 0.5));

    // ── Background: dark navy, slightly faded for older history. ─────────
    let age_fade = mix(1.0, 0.55, clamp(tau / max(8.0 / zoom, 1e-3), 0.0, 1.0));
    var col = vec3<f32>(0.012, 0.018, 0.045) * age_fade;

    // ── Faint lattice potential heatmap (vertical stripes). ──────────────
    let lattice = 0.5 + 0.5 * cos(x * (3.0 + mp(MP_KSCALE) * 1.5));
    col += pow(lattice, 6.0) * u.crystal_color.xyz * 0.10 * age_fade;

    // ── Wavepacket glow, coloured by phase (rainbow palette). ────────────
    let hue = fract(phase / TAU * 0.5 + mp(MP_COLOR_SHIFT));
    let pal = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
    col += pal * pow(rho, 0.7) * 1.4;

    // ── Phase fringes overlaid on the density. ───────────────────────────
    let fringe_contrast = 0.5 + 0.5 * mp(MP_ISO_LEVEL);
    col += rho * (0.5 + 0.5 * cos(phase * 4.0 + tau * 2.0)) * 0.25
              * fringe_contrast * u.crystal_color.xyz;

    // ── Bright leading-edge head where the packet centre currently sits. ─
    let head = smoothstep(0.03, 0.0, abs(dx));
    col += head * vec3<f32>(1.0, 0.95, 0.75) * 1.2;

    // ── "Now" line: brighten the top edge so the user sees time flows down.
    let now_line = smoothstep(0.04, 0.0, abs(uv.y - 1.0));
    col += now_line * vec3<f32>(0.7, 0.85, 1.0) * 0.6;

    // Subtle vertical guide at x = 0 (origin of the lattice).
    col += smoothstep(0.004, 0.0, abs(uv.x)) * vec3<f32>(0.15, 0.18, 0.25);

    return col;
}
