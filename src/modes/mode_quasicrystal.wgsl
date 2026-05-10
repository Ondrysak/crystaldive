// -- QUASICRYSTAL: fivefold reciprocal interference with Penrose-like diffraction --

fn quasicrystal_rot(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn quasicrystal_hue_shift(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k  = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

fn quasicrystal_basis(a: f32) -> vec2<f32> {
    return vec2<f32>(cos(a), sin(a));
}

fn quasicrystal_wave(p: vec2<f32>, k: vec2<f32>, ph: f32) -> f32 {
    return cos(dot(p, k) + ph);
}

fn quasicrystal_spot(q: vec2<f32>, k: vec2<f32>, w: f32) -> f32 {
    let d0 = length(q - k);
    let d1 = length(q + k);
    let d = min(d0, d1);
    return exp(-d * d / max(w * w, 0.00001));
}

fn render_quasicrystal(uv: vec2<f32>) -> vec3<f32> {
    let phi = 1.61803398875;
    let t = u.time * u.speed;

    let zoom = max(u.zoom, 0.05);
    let scale = (7.5 + 5.5 * u.kscale) / zoom;
    let irrational_turn = TAU / (phi * phi + 1.0);
    let base_angle = u.color_shift * TAU * 0.17 + t * 0.025;
    let p = quasicrystal_rot(uv * scale, base_angle);

    var field_a = 0.0;
    var field_b = 0.0;
    var envelope = 0.0;
    var ridge = 0.0;

    for (var i = 0; i < 10; i++) {
        let fi = f32(i);
        let five_angle = TAU * fi / 5.0;
        let ten_angle = TAU * fi / 10.0 + irrational_turn * 0.23;
        let a = five_angle + base_angle * 0.37;
        let b = ten_angle - base_angle * 0.19;

        let k0 = quasicrystal_basis(a);
        let k1 = quasicrystal_basis(b);
        let amp = mix(1.0, 1.0 / phi, select(0.0, 1.0, (i % 2) == 1));
        let ph0 = t * (0.12 + 0.009 * fi) + phi * fi;
        let ph1 = -t * (0.08 + 0.006 * fi) + fi * irrational_turn;

        let w0 = quasicrystal_wave(p, k0, ph0);
        let w1 = quasicrystal_wave(p * phi, k1, ph1);
        field_a += amp * w0;
        field_b += amp * w1;
        envelope += amp;
        ridge += pow(abs(w0 * w1), 5.0) * amp;
    }

    let inv_env = 1.0 / max(envelope, 0.001);
    let fa = field_a * inv_env;
    let fb = field_b * inv_env;
    let interference = fa * fb + 0.45 * (fa + fb);
    let cells = abs(interference);

    let iso = mix(0.18, 0.72, clamp(u.iso_level, 0.0, 1.0));
    let line_w = mix(0.055, 0.014, clamp(u.iso_level, 0.0, 1.0));
    let penrose_edges = smoothstep(line_w, 0.0, abs(cells - iso));
    let star_edges = smoothstep(line_w * 1.4, 0.0, abs(abs(fa) - abs(fb)));

    let q_scale = mix(1.9, 3.1, u.field_mix);
    let q = uv * q_scale;
    var diffraction = 0.0;
    var ring = 0.0;
    for (var j = 0; j < 10; j++) {
        let fj = f32(j);
        let a = TAU * fj / 10.0 + irrational_turn * 0.11 + t * 0.01;
        let k = quasicrystal_basis(a);
        diffraction += quasicrystal_spot(q, k, 0.030);
        diffraction += 0.55 * quasicrystal_spot(q, k * phi, 0.024);
        diffraction += 0.32 * quasicrystal_spot(q, k / phi, 0.035);
        ring += exp(-pow(abs(length(q) - (1.0 + 0.16 * sin(fj * phi))), 2.0) / 0.0025);
    }

    let decagon = pow(clamp(diffraction * 0.22, 0.0, 1.0), 0.65);
    let center_bloom = exp(-dot(q, q) / 0.060);
    let radial_cut = smoothstep(1.9, 0.15, length(uv));

    let base = mix(vec3<f32>(0.010, 0.010, 0.018),
                   u.crystal_color.xyz * 0.10,
                   0.45);
    let cool = quasicrystal_hue_shift(vec3<f32>(0.20, 0.62, 0.95), u.color_shift);
    let warm = quasicrystal_hue_shift(vec3<f32>(1.00, 0.78, 0.32), u.color_shift + 0.09);
    let accent = quasicrystal_hue_shift(u.crystal_color.xyz, u.color_shift * 0.5);

    var col = base;
    col += smoothstep(0.05, 0.95, cells) * cool * 0.35;
    col += penrose_edges * warm * (1.15 + 0.65 * radial_cut);
    col += star_edges * accent * 0.55;
    col += ridge * inv_env * mix(0.18, 0.55, u.field_mix) * accent;
    col += decagon * mix(warm, vec3<f32>(1.0), 0.45) * 2.0;
    col += center_bloom * vec3<f32>(1.0, 0.92, 0.68) * 1.4;
    col += ring * 0.015 * cool;

    return col;
}
