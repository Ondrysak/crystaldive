// -- Mode: BAND_SURFACE - electronic energy landscape over k-space --

fn band_surface_hue_shift(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k  = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

fn band_surface_rot(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn band_surface_wave(k: vec2<f32>, phase_mix: f32) -> f32 {
    var e  = 0.0;
    var ns = 0.0;
    let ng_bs = i32(u.num_g);
    for (var i = 0; i < ng_bs; i++) {
        let ga  = g_block.gamp[i];
        let ph  = g_block.phases[i].x;
        let g2  = ga.xy * u.kscale;
        let amp = ga.w;
        let q   = dot(g2, k);

        let lat   = cos(q + ph * phase_mix);
        let motif = cos(q * 1.37 + ph * 1.9 + phase_mix * 1.7);
        let band  = sin(dot(vec2<f32>(-g2.y, g2.x), k) * 0.73 + ph);
        e  += amp * (u.w_lattice * lat + u.w_motif * motif * 0.55 + u.w_band * band * 0.45);
        ns += amp * (u.w_lattice + abs(u.w_motif) * 0.55 + abs(u.w_band) * 0.45);
    }
    return e / max(ns, 0.001);
}

fn band_surface_bands(k: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed;
    let drift = vec2<f32>(cos(t * 0.09), sin(t * 0.07)) * 0.08;
    let kr = band_surface_rot(k - drift, 0.45 + u.field_mix * 0.55);

    let saddle_a = (k.x * k.x - k.y * k.y) * 0.18;
    let saddle_b = (kr.x * kr.y) * 0.34;
    let ripple_a = band_surface_wave(k + drift, 0.4);
    let ripple_b = band_surface_wave(kr * 1.12 - drift.yx, 1.0);

    let e_a = ripple_a * 0.82 + saddle_a - 0.16 * u.iso_level;
    let e_b = -ripple_b * 0.72 + saddle_b + 0.16 * (u.field_mix - 0.5);

    let crossing = abs(e_a - e_b);
    let nodal = sin((k.x * 1.7 - k.y * 1.2) + t * 0.12);
    let coupling = 0.055 + 0.22 * u.field_mix + 0.035 * nodal * nodal;
    let gap = sqrt(crossing * crossing + coupling * coupling);

    return vec3<f32>(0.5 * (e_a + e_b - gap), 0.5 * (e_a + e_b + gap), gap);
}

fn band_surface_contour(e: f32, spacing: f32, width: f32) -> f32 {
    let x = abs(fract(e / spacing + 0.5) - 0.5) * spacing;
    return smoothstep(width, 0.0, x);
}

fn render_band_surface(uv: vec2<f32>) -> vec3<f32> {
    let k = uv * (2.15 / max(u.zoom, 0.05));
    let bands = band_surface_bands(k);
    let e0 = bands.x;
    let e1 = bands.y;
    let gap = bands.z;

    let base_a = band_surface_hue_shift(max(u.crystal_color.xyz, vec3<f32>(0.08)), u.color_shift);
    let base_b = band_surface_hue_shift(vec3<f32>(1.0, 0.78, 0.30), u.color_shift + 0.08);

    let energy_fill = 0.5 + 0.5 * tanh(e0 * 1.35);
    var col = mix(vec3<f32>(0.006, 0.008, 0.016), vec3<f32>(0.035, 0.025, 0.055), energy_fill);
    col += base_a * smoothstep(-0.55, 0.70, e0) * 0.16;
    col += base_b * smoothstep(0.80, -0.45, e1) * 0.10;

    let spacing = mix(0.18, 0.075, clamp(u.iso_level, 0.0, 1.0));
    let width = mix(0.018, 0.007, clamp(u.iso_level, 0.0, 1.0));
    let c0 = band_surface_contour(e0, spacing, width);
    let c1 = band_surface_contour(e1, spacing, width * 1.15);
    col += c0 * base_a * 1.35;
    col += c1 * base_b * 0.95;

    let zero_fermi = smoothstep(width * 1.4, 0.0, abs(e0));
    col += zero_fermi * vec3<f32>(0.86, 0.96, 1.00) * 1.2;

    let avoided = exp(-gap * gap * 28.0);
    col += avoided * mix(vec3<f32>(1.0, 0.45, 0.25), base_b, 0.35) * 1.25;

    let saddle_axes = exp(-abs(k.x * k.y) * 2.2);
    let saddle_core = exp(-dot(k, k) * 0.28);
    let saddle = saddle_axes * saddle_core * smoothstep(0.12, 0.55, abs(e0));
    col += saddle * vec3<f32>(0.95, 0.75, 1.10) * 0.85;

    let contour_density = clamp(c0 * 0.75 + c1 * 0.55 + avoided, 0.0, 1.6);
    let hot = exp(-abs(e0) * 4.5) * saddle_axes * (0.35 + contour_density);
    col += hot * mix(vec3<f32>(1.0, 0.95, 0.48), base_a, 0.25) * 1.6;

    let grid_k = max(abs(fract(k.x * 0.5 + 0.5) - 0.5), abs(fract(k.y * 0.5 + 0.5) - 0.5));
    let grid = smoothstep(0.015, 0.0, grid_k);
    col += grid * vec3<f32>(0.08, 0.10, 0.13);

    return col;
}
