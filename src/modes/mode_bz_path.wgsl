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
        let g0 = g_block.gamp[0].xyz;
        let g1 = g_block.gamp[1].xyz;
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
    let ng_bzp = i32(u.num_g);
    for (var i = 0; i < ng_bzp; i++) {
        let ga = g_block.gamp[i];
        let ph = g_block.phases[i].x;
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

fn bz_path_poly_edge(p: vec2<f32>) -> vec4<f32> {
    let gamma = vec2<f32>(-0.55, -0.34);
    let xpt = vec2<f32>(0.54, -0.34);
    let mpt = vec2<f32>(0.54, 0.44);

    let e0 = bz_path_segment_distance(p, gamma, xpt).x;
    let e1 = bz_path_segment_distance(p, xpt, mpt).x;
    let e2 = bz_path_segment_distance(p, mpt, gamma).x;

    let s0 = sign((xpt.x - gamma.x) * (p.y - gamma.y) - (xpt.y - gamma.y) * (p.x - gamma.x));
    let s1 = sign((mpt.x - xpt.x) * (p.y - xpt.y) - (mpt.y - xpt.y) * (p.x - xpt.x));
    let s2 = sign((gamma.x - mpt.x) * (p.y - mpt.y) - (gamma.y - mpt.y) * (p.x - mpt.x));
    let inside = 1.0 - step(0.5, max(max(abs(s0 - s1), abs(s1 - s2)), abs(s2 - s0)));

    return vec4<f32>(e0, e1, e2, inside);
}

fn bz_path_rot(a: f32) -> mat2x2<f32> {
    let c = cos(a);
    let s = sin(a);
    return mat2x2<f32>(c, -s, s, c);
}

fn bz_path_sd_circle(p: vec2<f32>, r: f32) -> f32 {
    return abs(length(p) - r);
}

fn render_bz_path(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed;
    let aspect = max(u.aspect, 0.0001);
    let p = vec2<f32>(uv.x / aspect, uv.y) * (1.08 / max(u.zoom, 0.2));

    let accent = bz_path_hue_shift(max(u.crystal_color.xyz, vec3<f32>(0.08)), u.color_shift);
    let gold = bz_path_hue_shift(vec3<f32>(1.0, 0.73, 0.28), u.color_shift * 0.45);
    let cyan = bz_path_hue_shift(vec3<f32>(0.25, 0.92, 1.0), u.color_shift * 0.3 + 0.08);

    let radial = dot(p, p);
    var col = vec3<f32>(0.003, 0.005, 0.012);
    col += mix(vec3<f32>(0.010, 0.014, 0.030), accent * 0.10, 0.55) * (1.0 - smoothstep(0.0, 2.15, radial));

    let scene_angle = 0.18 * sin(t * 0.23) + 0.07 * cos(t * 0.41);
    let q = bz_path_rot(scene_angle) * p;
    let parallax = vec2<f32>(0.11 * sin(t * 0.19), 0.07 * cos(t * 0.27));

    let grid_scale = 6.5 + u.iso_level * 9.5;
    let lattice_a = q + parallax;
    let grid_x = abs(sin((lattice_a.x + lattice_a.y * 0.32) * grid_scale + t * 0.45));
    let grid_y = abs(sin((lattice_a.y - lattice_a.x * 0.28) * grid_scale - t * 0.32));
    let grid = min(grid_x, grid_y);
    col += pow(1.0 - grid, 22.0) * mix(accent, cyan, 0.45) * 0.10;
    col += pow(1.0 - grid, 7.0) * vec3<f32>(0.03, 0.05, 0.08) * 0.22;

    let ring0 = exp(-bz_path_sd_circle(q + vec2<f32>(0.02, -0.02), 0.56) * 18.0);
    let ring1 = exp(-bz_path_sd_circle(q - vec2<f32>(0.05, 0.03), 0.82) * 12.0);
    col += (ring0 * cyan + ring1 * gold) * 0.035;

    let skew = mat2x2<f32>(1.0, 0.34, -0.22, 1.0);
    let lp = skew * (q + parallax * 0.35) * 8.0;
    let cell = floor(lp);
    let cell_uv = fract(lp) - 0.5;
    let reciprocal_dot = exp(-dot(cell_uv, cell_uv) * 72.0);
    let dot_gate = step(0.64, bz_path_hash(cell.x * 17.0 + cell.y * 31.0));
    let twinkle = 0.35 + 0.65 * pow(max(0.0, sin(t * 1.8 + bz_path_hash(cell.x + cell.y * 13.0) * TAU)), 6.0);
    col += reciprocal_dot * dot_gate * twinkle * mix(cyan, gold, bz_path_hash(cell.x + 9.0 * cell.y)) * 0.22;

    let pane = bz_path_poly_edge(q);
    let pane_edge = min(min(pane.x, pane.y), pane.z);
    let pane_rim = exp(-(pane_edge * pane_edge) / 0.0028);
    let pane_fill_wave = 0.5 + 0.5 * sin((q.x * 14.0 - q.y * 10.0) + t * 1.25);
    let pane_scan = exp(-abs(dot(q, normalize(vec2<f32>(0.72, -0.34))) - sin(t * 0.7) * 0.55) * 10.0);
    col += pane.w * mix(vec3<f32>(0.02, 0.04, 0.08), accent * 0.22, pane_fill_wave) * 0.55;
    col += pane.w * pane_scan * cyan * 0.22;
    col += pane_rim * vec3<f32>(0.92, 0.97, 1.0) * (0.28 + 0.18 * sin(t * 1.7));

    let near = bz_path_nearest_path(q);
    let k = bz_path_k_point(near.y);
    let response = bz_path_response(k, t * 0.75);
    let response2 = bz_path_response(k + vec3<f32>(0.21, 0.13, 0.08), t * -0.42);
    let mixed_response = mix(response, response2, u.field_mix);

    let path_core_w = mix(0.020, 0.008, clamp(u.iso_level, 0.0, 1.0));
    let path_core = smoothstep(path_core_w, 0.0, near.x);
    let path_halo = exp(-(near.x * near.x) / 0.014);
    let traveling_s = fract(t * 0.155 + 0.025 * sin(t * 0.57));
    let trailing_s0 = fract(traveling_s - 0.055);
    let trailing_s1 = fract(traveling_s - 0.115);
    let dk = abs(near.y - traveling_s);
    let wrapped_dk = min(dk, 1.0 - dk);
    let trail0 = exp(-pow(min(abs(near.y - trailing_s0), 1.0 - abs(near.y - trailing_s0)), 2.0) * 170.0);
    let trail1 = exp(-pow(min(abs(near.y - trailing_s1), 1.0 - abs(near.y - trailing_s1)), 2.0) * 170.0);
    let chase = exp(-wrapped_dk * wrapped_dk * 190.0);
    let pulse = 0.5 + 0.5 * sin(TAU * (near.y * 9.0 - t * 0.85) + mixed_response * 3.0);
    let ticks = pow(max(0.0, cos(TAU * (near.y * 24.0 - t * 0.9))), 20.0);
    let data_gate = 0.55 + 0.45 * step(0.0, sin(TAU * (near.y * 11.0 + mixed_response * 0.7 - t * 0.6)));

    col += path_halo * mix(accent, cyan, 0.35 + 0.25 * mixed_response) * (0.22 + 0.42 * pulse);
    col += path_core * mix(gold, accent, 0.45 + 0.35 * mixed_response) * (1.05 + 2.2 * chase);
    col += path_core * ticks * data_gate * vec3<f32>(1.0, 0.98, 0.82) * 1.1;
    col += path_halo * (trail0 * 0.95 + trail1 * 0.55) * cyan * 1.15;

    let probe = bz_path_screen_point(traveling_s);
    let probe_world = bz_path_rot(-scene_angle) * probe;
    let probe_k = bz_path_k_point(traveling_s);
    let probe_response = bz_path_response(probe_k, t);
    let probe_glow = bz_path_node_glow(q, probe, 0.085 + 0.045 * abs(probe_response));
    let probe_core = bz_path_node_glow(q, probe, 0.024);
    let scan_ray = exp(-abs(dot(q - probe, normalize(vec2<f32>(0.22, 1.0)))) * 34.0) * exp(-length(q - probe) * 2.3);
    col += probe_glow * mix(cyan, gold, 0.5 + 0.5 * probe_response) * 1.7;
    col += probe_core * vec3<f32>(1.0, 0.97, 0.82) * 3.0;
    col += scan_ray * mix(cyan, gold, 0.45 + 0.35 * probe_response) * 0.35;
    col += exp(-length(p - probe_world) * 16.0) * vec3<f32>(0.90, 0.96, 1.0) * 0.18;

    let gamma = vec2<f32>(-0.55, -0.34);
    let xpt = vec2<f32>(0.54, -0.34);
    let mpt = vec2<f32>(0.54, 0.44);
    let node_g = bz_path_node_glow(q, gamma, 0.060);
    let node_x = bz_path_node_glow(q, xpt, 0.055);
    let node_m = bz_path_node_glow(q, mpt, 0.055);
    col += node_g * vec3<f32>(1.0, 0.95, 0.78) * 1.3;
    col += node_x * cyan * 0.95;
    col += node_m * accent * 1.05;
    col += (node_g + node_x + node_m) * (0.35 + 0.35 * sin(t * 2.0)) * vec3<f32>(1.0, 1.0, 0.92);

    let band_y = p.y + 0.80;
    let band_x = clamp((p.x + 0.76) / 1.52, 0.0, 1.0);
    let band_k = bz_path_k_point(band_x);
    let e0 = bz_path_response(band_k, t * 0.36);
    let e1 = bz_path_response(band_k + vec3<f32>(0.37, 0.0, 0.19), -t * 0.28);
    let band_a = -0.05 + e0 * 0.18 + 0.035 * sin(TAU * (band_x * 3.0 - t * 0.16));
    let band_b = 0.12 + e1 * 0.16 - 0.030 * cos(TAU * (band_x * 2.0 + t * 0.11));
    let band_c = 0.26 + (e0 - e1) * 0.12 + 0.024 * sin(TAU * (band_x * 5.0 + t * 0.13));
    let band_d = -0.20 + (e0 + e1) * 0.08 - 0.020 * cos(TAU * (band_x * 4.0 - t * 0.20));
    let band_w = mix(0.015, 0.006, clamp(u.iso_level, 0.0, 1.0));
    let band_pulse = exp(-min(abs(band_x - traveling_s), 1.0 - abs(band_x - traveling_s)) * 12.0);
    let line_a = smoothstep(band_w, 0.0, abs(band_y - band_a));
    let line_b = smoothstep(band_w, 0.0, abs(band_y - band_b));
    let line_c = smoothstep(band_w * 0.82, 0.0, abs(band_y - band_c));
    let line_d = smoothstep(band_w * 0.75, 0.0, abs(band_y - band_d));
    let avoided_crossing = exp(-abs(band_a - band_b) * 18.0) * exp(-abs(band_y - (band_a + band_b) * 0.5) * 24.0);
    let band_window = smoothstep(-0.76, -0.70, p.x) * smoothstep(0.76, 0.70, p.x) * smoothstep(-0.30, -0.20, band_y) * smoothstep(0.42, 0.32, band_y);
    let spectro = pow(max(0.0, sin(TAU * (band_x * 18.0 + band_y * 2.0 - t * 0.75))), 12.0);
    col += band_window * line_a * gold * (0.9 + 1.4 * band_pulse);
    col += band_window * line_b * cyan * (0.8 + 1.1 * band_pulse);
    col += band_window * line_c * mix(accent, gold, 0.45) * (0.48 + 0.8 * band_pulse);
    col += band_window * line_d * mix(cyan, accent, 0.55) * (0.38 + 0.7 * band_pulse);
    col += band_window * avoided_crossing * vec3<f32>(1.0, 0.82, 0.42) * 0.42;
    col += band_window * exp(-abs(band_y) * 8.0) * accent * 0.055;
    col += band_window * exp(-abs(band_y - 0.31) * 45.0) * cyan * 0.06;
    col += band_window * exp(-abs(band_y + 0.24) * 45.0) * gold * 0.06;
    col += band_window * spectro * mix(accent, cyan, band_x) * 0.16;

    let spark_seed = floor(near.y * 36.0);
    let sparkle = step(0.84, bz_path_hash(spark_seed)) * pow(max(0.0, sin(TAU * (t * 0.9 + bz_path_hash(spark_seed + 4.0)))), 10.0);
    col += sparkle * path_core * vec3<f32>(1.0, 0.92, 0.62) * 1.2;

    let burst_phase = fract(t * 0.5);
    let burst_pos = bz_path_screen_point(fract(traveling_s + 0.33));
    let burst = exp(-length(q - burst_pos) * 12.0) * exp(-abs(length(q - burst_pos) - burst_phase * 0.38) * 18.0) * (1.0 - burst_phase);
    col += burst * mix(gold, cyan, burst_phase) * 0.45;

    return col;
}
