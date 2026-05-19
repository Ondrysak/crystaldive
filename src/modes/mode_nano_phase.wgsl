// ── Mode 38: NANO_PHASE — size-dependent melting of a nanocrystal ──────────
//
// Physics:
//   Pawlow, Z. Phys. Chem. 65, 1 (1909)         — Tm(R) depression
//   Buffat & Borel, PRA 13, 2287 (1976)         — gold nanoparticle melting
//   Sakai, Surf. Sci. 351, 285 (1996)           — surface premelting
//   Wang et al., Science 312, 1199 (2006)       — coexisting twins in nano-Au
//
//   Gibbs–Thomson / liquid-skin model:
//       Tm(R) = Tm_bulk · (1 − 2σ_sl / (ρ L R))
//   A liquid skin of thickness δ(T) nucleates at the surface long before the
//   core melts. We render a single nanoparticle with:
//     • a lattice-ordered crystalline core (crystal_field),
//     • a quasi-liquid premelted shell whose thickness ∝ smoothstep(Tm − ΔT, Tm, T),
//     • an internal twin/habit plane (martensitic flavour) that sweeps with t,
//     • drops of evaporated liquid shed into vacuum as T → 1.
//
// Slider semantics:
//   iso_level   → reduced temperature  T/Tm_bulk   ∈ [0, 1]
//   field_mix   → twin phase bias (austenite vs martensite) ∈ [0, 1]
//   kscale      → particle radius R (small R → lower Tm via size law)
//   color_shift → palette hue rotation
//   w_lattice   → core lattice contrast
//   w_motif     → twin habit-plane sharpness
//   w_band      → liquid-shell turbulence amplitude
//   mouse (drag) → drag the particle around the frame

fn nano_phase_hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn nano_phase_vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u_ = f * f * (3.0 - 2.0 * f);
    let a = nano_phase_hash21(i);
    let b = nano_phase_hash21(i + vec2<f32>(1.0, 0.0));
    let c = nano_phase_hash21(i + vec2<f32>(0.0, 1.0));
    let d = nano_phase_hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u_.x), mix(c, d, u_.x), u_.y);
}

fn nano_phase_fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var amp = 0.55;
    var q = p;
    for (var i = 0; i < 4; i = i + 1) {
        v = v + amp * nano_phase_vnoise(q);
        q = q * 2.07 + vec2<f32>(13.7, 7.3);
        amp = amp * 0.5;
    }
    return v;
}

// Soft palette: cold→warm interpolation, hue-rotated by color_shift.
fn nano_phase_palette(x: f32) -> vec3<f32> {
    let h = fract(u.color_shift + 0.0);
    let cold = 0.5 + 0.5 * cos(TAU * (h + vec3<f32>(0.62, 0.70, 0.85)));
    let warm = 0.5 + 0.5 * cos(TAU * (h + vec3<f32>(0.02, 0.18, 0.40)));
    return mix(cold, warm, clamp(x, 0.0, 1.0));
}

fn render_nano_phase(uv: vec2<f32>) -> vec3<f32> {
    let t      = u.time * u.speed;
    // ── Geometry: particle radius R from kscale, centre from mouse drag.
    //   View-space coordinates so we have headroom for evaporated drops outside R.
    let p = uv * (2.0 / max(u.zoom, 0.08));
    var centre = vec2<f32>(0.0, 0.0);
    if u.mouse_down >= 0.5 {
        // Mouse is in [0,1]² — map into the same view-space as p.
        centre = vec2<f32>(
            (u.mouse.x - 0.5) * 2.0 * u.aspect * (2.0 / max(u.zoom, 0.08)),
            (0.5 - u.mouse.y) * 2.0 * (2.0 / max(u.zoom, 0.08)),
        );
    }
    let R = clamp(0.55 + 0.55 * u.kscale, 0.30, 1.40);

    // ── Size-dependent melting point (Gibbs–Thomson). Smaller R → lower Tm.
    //   Tm_eff in [0..1] of the bulk; map iso_level so small particles melt early.
    let tm_eff = clamp(1.0 - 0.32 / max(R, 0.18), 0.30, 1.00);
    let temp   = clamp(u.iso_level, 0.0, 1.0);
    // Reduced temperature relative to *this* particle's Tm:
    let theta  = clamp(temp / max(tm_eff, 0.05), 0.0, 1.30);

    // ── Surface premelting: a liquid skin of thickness δ(T) nucleates well
    //   before the core melts. δ diverges logarithmically near Tm in theory;
    //   we use a smooth ramp from 0 at θ≈0.45 to ~R at θ≈1.
    let delta = R * smoothstep(0.45, 1.00, theta);
    // Radial distance + thermal wobble of the particle interface (capillary waves).
    let d_local = p - centre;
    let r = length(d_local);
    let wobble = 0.04 * R * (nano_phase_fbm(d_local * 4.5 + vec2<f32>(t * 0.6, -t * 0.45)) - 0.5) * smoothstep(0.2, 0.9, theta);
    let r_eff  = r + wobble;
    // Core radius is what's left of the solid:
    let r_core   = max(R - delta, 0.0);
    let in_part  = smoothstep(R + 0.02, R - 0.02, r_eff);            // 1 inside particle
    let in_core  = smoothstep(r_core + 0.04, r_core - 0.04, r_eff);  // 1 inside solid core
    let in_shell = clamp(in_part - in_core, 0.0, 1.0);

    // ── Crystalline core: crystal_field on local coords, scaled so a few unit
    //   cells span the core. Twin/martensitic habit plane sweeps through.
    let twin_dir = vec2<f32>(cos(0.55 + t * 0.18), sin(0.55 + t * 0.18));
    let twin_pos = (u.field_mix - 0.5) * 1.6 * R + 0.25 * R * sin(t * 0.27);
    let twin_s   = dot(d_local, twin_dir) - twin_pos;
    let twin_w   = 0.04 + 0.18 * (1.0 - clamp(u.w_motif, 0.0, 1.0));  // wall thickness
    let twin_mix = smoothstep(-twin_w, twin_w, twin_s);               // 0=austenite, 1=martensite
    let twin_wall = exp(-(twin_s * twin_s) / (twin_w * twin_w * 0.6));
    // Two sublattices: rotate the same crystal field by different angles to fake
    // FCC vs BCT symmetry within one particle.
    let aA = 0.0;
    let aB = 0.78;  // ~45° habit-plane misorientation
    let pA = mat2x2<f32>(vec2<f32>(cos(aA), sin(aA)), vec2<f32>(-sin(aA), cos(aA))) * d_local;
    let pB = mat2x2<f32>(vec2<f32>(cos(aB), sin(aB)), vec2<f32>(-sin(aB), cos(aB))) * (d_local * vec2<f32>(1.0, 0.78));  // BCT c/a≈0.78
    let fA = crystal_field(vec3<f32>(pA * (2.4 / max(R, 0.18)), t * 0.10));
    let fB = cf2(vec3<f32>(pB * (2.4 / max(R, 0.18)), -t * 0.08));
    let f_core = mix(fA, fB, twin_mix);
    // Atomic-row pattern: bright peaks at +1, dark at -1.
    let lat = 0.5 + 0.5 * sin(f_core * 3.2);

    // ── Debye–Waller broadening inside the core: peaks soften as θ→1.
    let dw = exp(-theta * theta * 2.4);
    let core_pattern = mix(0.45, 1.0, lat) * dw + (1.0 - dw) * 0.55;

    // ── Liquid shell: vector-noise turbulence + cf2-driven swirl, no lattice.
    let shell_q = d_local * 3.2 + vec2<f32>(t * 0.7, -t * 0.5);
    let shell_n = nano_phase_fbm(shell_q);
    let shell_swirl = 0.5 + 0.5 * sin(cf2(vec3<f32>(d_local * 1.6, t * 0.3)) * 2.0 + shell_n * 4.0);
    let shell_amp   = mix(0.5, 1.0, clamp(u.w_band, 0.0, 1.0));
    let shell_val   = shell_swirl * shell_amp;

    // ── Vapour: drops of liquid evaporated into vacuum once T > Tm. We render
    //   them as a faint speckled cloud surrounding the particle.
    let vapour_t = smoothstep(0.95, 1.10, theta);
    var vapour   = 0.0;
    if vapour_t > 0.001 {
        let q = (d_local - centre * 0.0) * 2.6;
        let nq = nano_phase_fbm(q + vec2<f32>(t * 0.9, t * 0.6))
               * nano_phase_fbm(q * 1.7 + vec2<f32>(-t * 0.4, t * 1.1));
        vapour = vapour_t * smoothstep(R * 1.05, R * 2.4, r) * pow(nq, 2.0) * 6.0;
    }

    // ── Colour assembly.
    let accent = u.crystal_color.xyz;
    let bg     = vec3<f32>(0.006, 0.008, 0.018);
    // Core: accent-tinted lattice with mild warming as θ→1.
    let core_cold = mix(accent * 0.55, accent * 1.30, core_pattern);
    let core_warm = vec3<f32>(1.40, 0.85, 0.35);
    let core_col  = mix(core_cold, mix(core_cold, core_warm, 0.45), smoothstep(0.6, 0.99, theta))
                  * (0.6 + 0.6 * clamp(u.w_lattice, 0.0, 1.0));
    // Habit plane: bright thin band that flashes at the boundary.
    let habit_col = vec3<f32>(1.20, 1.05, 0.60) * twin_wall * 0.55 * clamp(u.w_motif, 0.0, 1.0);
    // Shell: silvery hot liquid skin.
    let shell_col = nano_phase_palette(shell_val) * (0.55 + 0.55 * shell_val)
                  + vec3<f32>(1.10, 0.55, 0.25) * shell_val * 0.45 * smoothstep(0.4, 1.0, theta);
    // Rim glow: hot edge of the particle.
    let rim       = exp(-pow(max(r - R, 0.0) / max(R * 0.18, 0.04), 2.0));
    let rim_col   = vec3<f32>(1.20, 0.70, 0.30) * rim * (0.35 + 0.65 * theta);
    // Vapour: dim accent-tinted speckle.
    let vapour_col = (accent * 0.40 + vec3<f32>(0.80, 0.55, 0.30)) * vapour;

    var col = bg;
    col = mix(col, core_col + habit_col, in_core);
    col = mix(col, shell_col, in_shell);
    col = col + rim_col * (1.0 - in_core * 0.85);
    col = col + vapour_col;

    // Surface premelting glow: thin ring at r ≈ R, brightens with shell width.
    let surf_ring = exp(-pow((r_eff - R) / 0.05, 2.0));
    col = col + accent * surf_ring * smoothstep(0.0, 0.6, delta / max(R, 0.05)) * 0.55;

    // Cold ambient lattice fringes outside the particle (substrate hint).
    let amb = 0.5 + 0.5 * sin(crystal_field(vec3<f32>(p * 1.6, t * 0.05)) * 2.0);
    col = col + accent * amb * 0.04 * (1.0 - in_part);

    return col;
}
