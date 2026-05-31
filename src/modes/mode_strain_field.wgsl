// Mode 49: STRAIN FIELD — crystal under biaxial strain showing lattice
// deformation and piezoelectric polarization charges.
// Left half: unstrained reference.  Right half: strained crystal.
// Dividing line animates or follows field_mix.
// Polarization charge density ρ_P ∝ div P colourises the transition region.

// Biaxial strain tensor (plane stress, z free to relax)
//   ε_xx = ε_yy = ε,  ε_zz = -2ν/(1-ν)·ε  (Poisson, ν≈0.3)
// Deformation map: (x,y,z) → (x(1+ε_xx), y(1+ε_yy), z(1+ε_zz))

fn strain_deform(p: vec3<f32>, eps: f32) -> vec3<f32> {
    let nu = 0.28;
    let eps_z = -2.0 * nu / (1.0 - nu) * eps;
    return vec3<f32>(p.x * (1.0 + eps), p.y * (1.0 + eps), p.z * (1.0 + eps_z));
}

// Piezoelectric polarization magnitude (e₁₅-like, proportional to shear strain
// which is zero for pure biaxial — so we use the volumetric coupling instead):
// P ∝ e₃₃ · ε_zz + e₃₁ · (ε_xx + ε_yy)
// Effective scalar: just use sign(eps)·f(strained position) as the density proxy.
fn strain_pol_density(p_strained: vec3<f32>, eps: f32) -> f32 {
    let cf_strained = crystal_field(p_strained);
    // Polarisation charge ~ -div(P) ≈ eps * Laplacian proxy
    let dx = 0.04;
    let lap = crystal_field(p_strained + vec3<f32>(dx,0,0))
            + crystal_field(p_strained - vec3<f32>(dx,0,0))
            + crystal_field(p_strained + vec3<f32>(0,dx,0))
            + crystal_field(p_strained - vec3<f32>(dx,0,0))
            - 4.0 * cf_strained;
    return eps * lap / (dx * dx);
}

fn render_strain_field(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * mp(MP_SPEED);
    let zoom = max(mp(MP_ZOOM), 0.1);

    // Animate strain: cycle between compression and tension
    let eps_max = 0.18 + 0.22 * mp(MP_ISO_LEVEL);
    let eps = eps_max * sin(t * 0.55) * (1.0 - mp(MP_FIELD_MIX))
            + eps_max * mp(MP_FIELD_MIX);  // field_mix holds strain at max

    // Dividing line x = split
    let split = 0.0;
    let blend_width = 0.08 / zoom;

    let p_ref = vec3<f32>(uv * 2.2 / zoom, t * 0.08);
    let p_def = strain_deform(p_ref, eps);

    let f_ref = crystal_field(p_ref);
    let f_def = crystal_field(p_def);
    let f2_ref = cf2(p_ref);
    let f2_def = cf2(p_def);

    // Blend across the split line
    let blend = smoothstep(split - blend_width, split + blend_width, uv.x);

    let f_use = mix(f_ref, f_def, blend);
    let f2_use = mix(f2_ref, f2_def, blend);

    // Base crystal colours
    let base = u.crystal_color.xyz;
    let amp = sqrt(f_use * f_use + f2_use * f2_use);
    let phase = atan2(f2_use, f_use);
    let hue_shift = phase / TAU + mp(MP_COLOR_SHIFT);
    // Simple hue rotation
    let hue_k = vec3<f32>(0.57735);
    let cs = cos(hue_shift * TAU);
    let sn = sin(hue_shift * TAU);
    let base_col = base * cs + cross(hue_k, base) * sn + hue_k * dot(hue_k, base) * (1.0 - cs);
    var col = base_col * (0.5 + amp * 1.2);

    // Strained side: warm tint proportional to strain
    let strain_heat = abs(eps) * 1.6;
    let warm = mix(vec3<f32>(0.0), vec3<f32>(0.9, 0.35, 0.05), clamp(strain_heat, 0.0, 1.0));
    col += warm * blend * (0.3 + 0.3 * sin(t * 1.2));

    // Piezo charge density at the split interface
    let interface_mask = exp(-10.0 * (uv.x - split) * (uv.x - split) / (blend_width * blend_width));
    let rho_P = strain_pol_density(p_def, eps);
    let pol_col = mix(vec3<f32>(0.2, 0.4, 1.0), vec3<f32>(1.0, 0.3, 0.1),
                       clamp(0.5 + rho_P * 0.4, 0.0, 1.0));
    col += pol_col * interface_mask * abs(rho_P) * 0.8;

    // Lattice contour lines (iso-surface of f)
    let contour = exp(-40.0 * f_use * f_use);
    col += vec3<f32>(1.0) * contour * 0.25;

    // Split boundary glow
    col += vec3<f32>(0.8, 0.9, 1.0) * exp(-50.0 * (uv.x - split) * (uv.x - split)) * 0.15;

    // Strain gauge text: scale bar at bottom (strain indicator strip)
    let bar_y = -0.85 / zoom;
    let bar_mask = step(-0.04, uv.y - bar_y) * step(uv.y - bar_y, 0.04)
                 * step(-0.5, uv.x) * step(uv.x, 0.5 * (1.0 + eps));
    col += vec3<f32>(1.0, 0.7, 0.1) * bar_mask * 0.9;

    return col;
}
