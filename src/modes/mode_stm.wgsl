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
    // iso_level controls QPI ring contrast (higher → sharper rings).
    let qpi_amp = 0.18 * (0.6 + 1.4 * u.iso_level);

    var qpi = 0.0;
    for (var i = 0; i < 5; i++) {
        let imp_p = stm_impurity_pos(i);
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

// Hue rotation around the luma axis — driven by u.color_shift.
fn stm_hue_rotate(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h);
    let sn = sin(h);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

fn render_stm(uv: vec2<f32>) -> vec3<f32> {
    let r = uv * 3.0 / u.zoom;

    // Tip height + finite-difference gradients for shaded-relief normal.
    let eps = 0.012;
    let z_tip = stm_z(r);
    let zx_p = stm_z(r + vec2<f32>(eps, 0.0));
    let zx_m = stm_z(r - vec2<f32>(eps, 0.0));
    let zy_p = stm_z(r + vec2<f32>(0.0, eps));
    let zy_m = stm_z(r - vec2<f32>(0.0, eps));
    let dz_dx = (zx_p - zx_m) / (2.0 * eps);
    let dz_dy = (zy_p - zy_m) / (2.0 * eps);
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
