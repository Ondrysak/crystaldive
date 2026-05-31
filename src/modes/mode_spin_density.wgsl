// Mode 50: SPIN DENSITY — real-space element-resolved magnetic moment map.
// Renders a 2D supercell where each lattice site is coloured by its local
// spin moment.  field_mix crossfades FM→AFM→ferrimagnetic ordering.
// iso_level controls moment magnitude.  Mouse drags the field of view.

// Sublattice parity for AFM — assign ±1 based on lattice site index.
fn sdens_sublattice(ix: i32, iy: i32) -> f32 {
    let parity = (ix + iy) & 1;
    return select(1.0, -1.0, parity == 1);
}

// Gaussian blob for a spin moment at a lattice site.
fn sdens_moment_blob(p: vec2<f32>, site: vec2<f32>, moment: f32, sigma: f32) -> f32 {
    let d = p - site;
    return moment * exp(-dot(d, d) / (sigma * sigma));
}

// Build the local spin density by summing over a patch of supercell sites.
// Uses the G-vectors to infer the real-space lattice vectors a₁, a₂.
fn sdens_density(p: vec2<f32>, t: f32, fm_weight: f32, afm_weight: f32, ferri_weight: f32) -> f32 {
    let ng = i32(u.num_g);

    // Lattice vectors: use the two smallest non-zero G vectors to derive a₁, a₂.
    // a = 2π / |G|  in the G direction's perpendicular (real-space dual).
    // Simplification: take G[0] and G[1] directly.
    var g0 = vec2<f32>(1.0, 0.0);
    var g1 = vec2<f32>(0.0, 1.0);
    var g0_len = 1.0;
    var g1_len = 1.0;
    if ng >= 1 {
        g0 = g_block.gamp[0].xy;
        g0_len = max(length(g0), 0.01);
        g0 = g0 / g0_len;
    }
    if ng >= 2 {
        g1 = g_block.gamp[1].xy;
        g1_len = max(length(g1), 0.01);
        g1 = g1 / g1_len;
    }
    // Real-space lattice spacing ~ 2π/|G|
    let a0 = TAU / max(g0_len * u.kscale, 0.1);
    let a1 = TAU / max(g1_len * u.kscale, 0.1);

    // Basis vectors
    let b0 = vec2<f32>(-g0.y, g0.x) * a0;
    let b1 = vec2<f32>(-g1.y, g1.x) * a1;

    let sigma = min(a0, a1) * 0.25;
    let m_max = u.iso_level * 2.0 + 0.3;

    var density = 0.0;
    let half = 6;
    for (var ix = -half; ix <= half; ix++) {
        for (var iy = -half; iy <= half; iy++) {
            let site = b0 * f32(ix) + b1 * f32(iy);
            let sl = sdens_sublattice(ix, iy);

            // Moment components
            let m_fm = m_max;
            let m_afm = m_max * sl;
            let ferri_scale = select(0.6, 1.0, (ix & 1) == 0);
            let m_ferri = m_max * sl * ferri_scale;

            let moment_raw = fm_weight * m_fm + afm_weight * m_afm + ferri_weight * m_ferri;
            // Spin dynamics: small oscillation around ordered state
            let osc = 0.08 * sin(t * 1.2 + f32(ix + iy) * 0.6);
            let moment = moment_raw * (1.0 + osc);

            density += sdens_moment_blob(p, site, moment, sigma);
        }
    }
    return density;
}

fn render_spin_density(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed * 0.5;
    let zoom = max(u.zoom, 0.05);

    var pan = vec2<f32>(0.0);
    if u.mouse_down >= 0.5 {
        pan = (u.mouse - 0.5) * 2.0;
    }
    let p = (uv + pan) / zoom;

    // field_mix encodes ordering type
    // 0..0.33 → FM, 0.33..0.67 → crossfade to AFM, 0.67..1 → ferrimagnetic
    let fm_w  = clamp(1.0 - u.field_mix * 3.0, 0.0, 1.0);
    let afm_w = clamp(1.0 - abs(u.field_mix - 0.5) * 6.0, 0.0, 1.0)
              + clamp((u.field_mix - 0.33) * 3.0, 0.0, 1.0) * clamp((0.67 - u.field_mix) * 3.0, 0.0, 1.0);
    let ferri_w = clamp((u.field_mix - 0.67) * 3.0, 0.0, 1.0);

    let density = sdens_density(p, t, fm_w, afm_w, ferri_w);

    // Colour map: negative moment = blue, zero = dark, positive = red
    let d_norm = clamp(density, -2.5, 2.5);
    let pos_col = mix(vec3<f32>(0.05, 0.02, 0.02), vec3<f32>(1.0, 0.15, 0.05),
                       clamp(d_norm / 2.5, 0.0, 1.0));
    let neg_col = mix(vec3<f32>(0.02, 0.02, 0.06), vec3<f32>(0.05, 0.25, 1.0),
                       clamp(-d_norm / 2.5, 0.0, 1.0));
    let zero_col = vec3<f32>(0.02, 0.025, 0.04);

    var col: vec3<f32>;
    if d_norm > 0.0 {
        col = mix(zero_col, pos_col, clamp(d_norm * 0.8, 0.0, 1.0));
    } else {
        col = mix(zero_col, neg_col, clamp(-d_norm * 0.8, 0.0, 1.0));
    }

    // Crystal accent tint
    col += u.crystal_color.xyz * abs(d_norm) * 0.12;

    // color_shift hue-rotates the palette
    let k = vec3<f32>(0.57735);
    let cs = cos(u.color_shift * TAU);
    let sn = sin(u.color_shift * TAU);
    col = col * cs + cross(k, col) * sn + k * dot(k, col) * (1.0 - cs);

    // Lattice grid overlay (faint)
    let ng = i32(u.num_g);
    var grid = 0.0;
    for (var i = 0; i < ng; i++) {
        let ga = g_block.gamp[i];
        let g = ga.xy * u.kscale;
        grid += ga.w * exp(-8.0 * abs(sin(dot(g, p))));
    }
    col += vec3<f32>(0.08) * clamp(grid * 0.05, 0.0, 0.25);

    return col;
}
