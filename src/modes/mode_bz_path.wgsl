// Mode: BRILLOUIN_PATH - high-symmetry k-path probe through reciprocal space.

fn bz_path_hue_shift(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

fn bz_path_hash(n: f32) -> f32 {
    return fract(sin(n * 127.1 + 19.19) * 43758.5453);
}

fn bz_path_segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let ba = b - a;
    let h = clamp(dot(p - a, ba) / max(dot(ba, ba), 0.0001), 0.0, 1.0);
    return vec2<f32>(length(p - (a + ba * h)), h);
}

fn bz_path_k_point(s: f32) -> vec3<f32> {
    var kx = vec3<f32>(0.55, 0.0, 0.0) * u.kscale;
    var km = vec3<f32>(0.55, 0.55, 0.0) * u.kscale;
    if (u.num_g >= 2u) {
        let g0 = textureLoad(g_tex, vec2<i32>(0, 0), 0).xyz;
        let g1 = textureLoad(g_tex, vec2<i32>(1, 0), 0).xyz;
        kx = g0 * u.kscale * 0.5;
        km = (g0 + g1) * u.kscale * 0.5;
    }

    if (s < 0.3333333) {
        return mix(vec3<f32>(0.0), kx, s * 3.0);
    }
    if (s < 0.6666667) {
        return mix(kx, km, (s - 0.3333333) * 3.0);
    }
    return mix(km, vec3<f32>(0.0), (s - 0.6666667) * 3.0);
}

fn bz_path_screen_point(s: f32) -> vec2<f32> {
    let gamma = vec2<f32>(-0.55, -0.34);
    let xpt = vec2<f32>(0.54, -0.34);
    let mpt = vec2<f32>(0.54, 0.44);

    if (s < 0.3333333) {
        return mix(gamma, xpt, s * 3.0);
    }
    if (s < 0.6666667) {
        return mix(xpt, mpt, (s - 0.3333333) * 3.0);
    }
    return mix(mpt, gamma, (s - 0.6666667) * 3.0);
}

fn bz_path_nearest_path(p: vec2<f32>) -> vec3<f32> {
    let gamma = vec2<f32>(-0.55, -0.34);
    let xpt = vec2<f32>(0.54, -0.34);
    let mpt = vec2<f32>(0.54, 0.44);

    let d0 = bz_path_segment_distance(p, gamma, xpt);
    let d1 = bz_path_segment_distance(p, xpt, mpt);
    let d2 = bz_path_segment_distance(p, mpt, gamma);

    var best_d = d0.x;
    var best_s = d0.y * 0.3333333;
    var seg = 0.0;
    if (d1.x < best_d) {
        best_d = d1.x;
        best_s = 0.3333333 + d1.y * 0.3333333;
        seg = 1.0;
    }
    if (d2.x < best_d) {
        best_d = d2.x;
        best_s = 0.6666667 + d2.y * 0.3333333;
        seg = 2.0;
    }
    return vec3<f32>(best_d, best_s, seg);
}

fn bz_path_response(k: vec3<f32>, phase: f32) -> f32 {
    var v = 0.0;
    var norm = 0.0;
    for (var i = 0; i < 96; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let ph = textureLoad(g_tex, vec2<i32>(i, 1), 0).r;
        let g = ga.xyz * u.kscale;
        let amp = ga.w;
        let band = dot(g, k) * (0.35 + 0.45 * u.w_band);
        v += amp * cos(band + ph + phase * (1.0 + f32(i) * 0.013));
        norm += amp;
    }
    return v / max(norm, 0.001);
}

fn bz_path_node_glow(p: vec2<f32>, c: vec2<f32>, r: f32) -> f32 {
    let d = length(p - c);
    return exp(-(d * d) / max(r * r, 0.0001));
}

fn render_bz_path(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed;
    let aspect = max(u.aspect, 0.0001);
    let p = vec2<f32>(uv.x / aspect, uv.y) * (1.08 / max(u.zoom, 0.2));

    let accent = bz_path_hue_shift(max(u.crystal_color.xyz, vec3<f32>(0.08)), u.color_shift);
    let gold = bz_path_hue_shift(vec3<f32>(1.0, 0.73, 0.28), u.color_shift * 0.45);
    let cyan = bz_path_hue_shift(vec3<f32>(0.25, 0.92, 1.0), u.color_shift * 0.3 + 0.08);

    let radial = dot(p, p);
    var col = vec3<f32>(0.006, 0.008, 0.015);
    col += mix(vec3<f32>(0.014, 0.018, 0.032), accent * 0.09, 0.45) * (1.0 - smoothstep(0.0, 1.9, radial));

    let grid_scale = 7.0 + u.iso_level * 9.0;
    let grid = abs(sin((p.x + p.y * 0.28) * grid_scale)) * abs(sin((p.y - p.x * 0.22) * grid_scale));
    col += pow(1.0 - grid, 18.0) * accent * 0.055;

    let near = bz_path_nearest_path(p);
    let k = bz_path_k_point(near.y);
    let response = bz_path_response(k, t * 0.75);
    let response2 = bz_path_response(k + vec3<f32>(0.21, 0.13, 0.08), t * -0.42);
    let mixed_response = mix(response, response2, u.field_mix);

    let path_core_w = mix(0.020, 0.008, clamp(u.iso_level, 0.0, 1.0));
    let path_core = smoothstep(path_core_w, 0.0, near.x);
    let path_halo = exp(-(near.x * near.x) / 0.018);
    let traveling_s = fract(t * 0.105);
    let dk = abs(near.y - traveling_s);
    let wrapped_dk = min(dk, 1.0 - dk);
    let chase = exp(-wrapped_dk * wrapped_dk * 160.0);
    let pulse = 0.5 + 0.5 * sin(TAU * (near.y * 9.0 - t * 0.85) + mixed_response * 3.0);

    col += path_halo * mix(accent, cyan, 0.35 + 0.25 * mixed_response) * (0.16 + 0.32 * pulse);
    col += path_core * mix(gold, accent, 0.45 + 0.35 * mixed_response) * (1.05 + 2.2 * chase);

    let probe = bz_path_screen_point(traveling_s);
    let probe_k = bz_path_k_point(traveling_s);
    let probe_response = bz_path_response(probe_k, t);
    let probe_glow = bz_path_node_glow(p, probe, 0.075 + 0.035 * abs(probe_response));
    let probe_core = bz_path_node_glow(p, probe, 0.026);
    col += probe_glow * mix(cyan, gold, 0.5 + 0.5 * probe_response) * 1.7;
    col += probe_core * vec3<f32>(1.0, 0.97, 0.82) * 2.4;

    let gamma = vec2<f32>(-0.55, -0.34);
    let xpt = vec2<f32>(0.54, -0.34);
    let mpt = vec2<f32>(0.54, 0.44);
    let node_g = bz_path_node_glow(p, gamma, 0.055);
    let node_x = bz_path_node_glow(p, xpt, 0.050);
    let node_m = bz_path_node_glow(p, mpt, 0.050);
    col += node_g * vec3<f32>(1.0, 0.95, 0.78) * 1.3;
    col += node_x * cyan * 0.95;
    col += node_m * accent * 1.05;

    let band_y = p.y + 0.78;
    let band_x = clamp((p.x + 0.72) / 1.44, 0.0, 1.0);
    let band_k = bz_path_k_point(band_x);
    let e0 = bz_path_response(band_k, t * 0.36);
    let e1 = bz_path_response(band_k + vec3<f32>(0.37, 0.0, 0.19), -t * 0.28);
    let band_a = -0.05 + e0 * 0.18 + 0.035 * sin(TAU * (band_x * 3.0 - t * 0.16));
    let band_b = 0.12 + e1 * 0.16 - 0.030 * cos(TAU * (band_x * 2.0 + t * 0.11));
    let band_w = mix(0.015, 0.006, clamp(u.iso_level, 0.0, 1.0));
    let band_pulse = exp(-min(abs(band_x - traveling_s), 1.0 - abs(band_x - traveling_s)) * 12.0);
    let line_a = smoothstep(band_w, 0.0, abs(band_y - band_a));
    let line_b = smoothstep(band_w, 0.0, abs(band_y - band_b));
    let band_window = smoothstep(-0.72, -0.68, p.x) * smoothstep(0.72, 0.68, p.x) * smoothstep(-0.28, -0.18, band_y) * smoothstep(0.40, 0.30, band_y);
    col += band_window * line_a * gold * (0.9 + 1.4 * band_pulse);
    col += band_window * line_b * cyan * (0.8 + 1.1 * band_pulse);
    col += band_window * exp(-abs(band_y) * 8.0) * accent * 0.035;

    let spark_seed = floor(near.y * 36.0);
    let sparkle = step(0.84, bz_path_hash(spark_seed)) * pow(max(0.0, sin(TAU * (t * 0.9 + bz_path_hash(spark_seed + 4.0)))), 10.0);
    col += sparkle * path_core * vec3<f32>(1.0, 0.92, 0.62) * 1.2;

    return col;
}
