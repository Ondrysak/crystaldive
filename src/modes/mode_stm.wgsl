// ── Mode 33: STM — top-down topograph with Friedel-ring QPI around impurities ──

fn stm_impurity_pos(i: i32) -> vec2<f32> {
    // 5 fixed pseudorandom impurity positions (A..E).
    if (i == 0) { return vec2<f32>( 0.6,  0.4); }
    if (i == 1) { return vec2<f32>(-0.9,  0.5); }
    if (i == 2) { return vec2<f32>( 0.2, -0.8); }
    if (i == 3) { return vec2<f32>(-0.5, -0.3); }
    return vec2<f32>( 1.1, -0.6);
}

fn stm_impurity_phase(i: i32) -> f32 {
    if (i == 0) { return 0.0; }
    if (i == 1) { return 1.7; }
    if (i == 2) { return 3.1; }
    if (i == 3) { return 4.4; }
    return 2.2;
}

// Density at a point: base |ψ|² plus Friedel-ring QPI from each impurity.
fn stm_density(r: vec2<f32>) -> f32 {
    let psi_real = crystal_field(vec3<f32>(r, 0.0));
    let psi_imag = cf2(vec3<f32>(r, 0.0));
    let rho_base = psi_real * psi_real + psi_imag * psi_imag;

    let kF = u.kscale * (3.0 + u.field_mix * 4.0);
    let qpi_amp = 0.18 * (0.6 + 1.4 * u.iso_level);

    var qpi = 0.0;
    for (var i = 0; i < 5; i++) {
        let imp_p  = stm_impurity_pos(i);
        let imp_ph = stm_impurity_phase(i);
        let d = length(r - imp_p);
        let env = exp(-d * 0.7);
        qpi += cos(2.0 * kF * d + imp_ph) * env / max(d, 0.15) * qpi_amp;
    }

    return rho_base + qpi;
}

// Tip-height proxy — constant-current STM mode.
fn stm_z(r: vec2<f32>) -> f32 {
    let rho = stm_density(r);
    return pow(max(rho, 0.0), 0.6);
}

// Returns vec3(rho, d_rho/dx, d_rho/dy) analytically in two G-vector passes
// (one for crystal_field, one for cf2), replacing 4 finite-difference stm_z calls.
fn stm_density_val_grad(r: vec2<f32>) -> vec3<f32> {
    let p3 = vec3<f32>(r, 0.0);
    // crystal_field pass
    let vg1 = crystal_field_val_grad_xy(p3);
    let f1   = vg1.x;
    // cf2 pass: cf2(p) = crystal_field(p*1.37 + offset)
    let vg2 = crystal_field_val_grad_xy(p3 * 1.37 + vec3<f32>(1.618, 2.718, 3.141));
    let f2   = vg2.x;

    let rho_base  = f1 * f1 + f2 * f2;
    var d_rho_dx  = 2.0 * (f1 * vg1.y + f2 * vg2.y * 1.37);
    var d_rho_dy  = 2.0 * (f1 * vg1.z + f2 * vg2.z * 1.37);

    let kF      = u.kscale * (3.0 + u.field_mix * 4.0);
    let qpi_amp = 0.18 * (0.6 + 1.4 * u.iso_level);

    var qpi = 0.0;
    for (var i = 0; i < 5; i++) {
        let imp_p  = stm_impurity_pos(i);
        let imp_ph = stm_impurity_phase(i);
        let dr     = r - imp_p;
        let d      = length(dr);
        let d_safe = max(d, 0.15);
        let env    = exp(-d * 0.7);
        let phase  = 2.0 * kF * d + imp_ph;
        let cos_ph = cos(phase);
        let sin_ph = sin(phase);

        qpi += cos_ph * env / d_safe * qpi_amp;

        // Analytical gradient of this QPI term w.r.t. r.
        if d > 0.001 {
            let inv_d  = 1.0 / d;
            let df_dd  = (-sin_ph * 2.0 * kF - cos_ph * 0.7) * env / d_safe
                       + select(0.0, -cos_ph * env / (d * d_safe), d > 0.15);
            let d_dr   = df_dd * inv_d * qpi_amp;
            d_rho_dx  += d_dr * dr.x;
            d_rho_dy  += d_dr * dr.y;
        }
    }

    return vec3<f32>(rho_base + qpi, d_rho_dx, d_rho_dy);
}

// Hue rotation around the luma axis — driven by u.color_shift.
fn stm_hue_rotate(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h);
    let sn = sin(h);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

fn render_stm(uv: vec2<f32>) -> vec3<f32> {
    let r = uv * 3.0 / u.zoom;

    // Analytical gradient replaces 4 finite-difference stm_z calls (saves 8 crystal_field evals).
    let rho_vg = stm_density_val_grad(r);
    let rho    = rho_vg.x;
    let z_tip  = pow(max(rho, 0.0), 0.6);
    // ∂z/∂r = 0.6 * rho^{-0.4} * ∂rho/∂r
    let dz_drho = select(0.0, 0.6 * pow(rho, -0.4), rho > 1e-6);
    let dz_dx   = dz_drho * rho_vg.y;
    let dz_dy   = dz_drho * rho_vg.z;
    let n = normalize(vec3<f32>(-dz_dx, -dz_dy, 1.0));

    // Lambert shading with a fixed grazing light source.
    let light = normalize(vec3<f32>(0.7, 0.4, 0.6));
    let diff = max(dot(n, light), 0.0);

    // Warm STM-image palette: dark indigo → tan/copper.
    var palette = mix(vec3<f32>(0.06, 0.04, 0.10),
                      vec3<f32>(0.95, 0.65, 0.30),
                      smoothstep(0.0, 1.5, z_tip));
    // color_shift hue-rotates the palette.
    palette = stm_hue_rotate(palette, u.color_shift * TAU);

    var stm_col = mix(palette, u.crystal_color.xyz, 0.20) * (0.35 + 0.85 * diff);

    // Bright impurity-site dots — small bright spot at each impurity.
    var bright_imp = 0.0;
    for (var i = 0; i < 5; i++) {
        let imp_p = stm_impurity_pos(i);
        let dv = r - imp_p;
        let d2 = dot(dv, dv);
        bright_imp += exp(-d2 * 80.0);
    }
    stm_col += bright_imp * vec3<f32>(1.10, 0.95, 0.70) * 1.2;

    return stm_col;
}
