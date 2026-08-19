// ── BRAGG KALEIDO — recursive polar folding by reciprocal-lattice stars ───

fn bragg_kaleido_rot(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn bragg_kaleido_palette(h: f32) -> vec3<f32> {
    let c = 0.54 + 0.46 * cos(TAU * (h + vec3<f32>(0.00, 0.36, 0.68)));
    return c * c;
}

fn render_bragg_kaleido(uv: vec2<f32>) -> vec3<f32> {
    if u.num_g == 0u { return vec3<f32>(0.006, 0.002, 0.016); }

    let symmetry = clamp(i32(round(mp(0u))), 3, 12);
    let t = u.time * mp(1u);
    let bend = clamp(mp(2u), 0.0, 1.0);
    let iter_count = clamp(i32(round(mp(3u))), 4, 14);
    let hue = mp(4u);
    let zoom = max(mp(5u), 0.2);
    let inflation = clamp(mp(6u), 1.05, 2.1);
    let crystal_lock = clamp(mp(7u), 0.0, 2.0);
    let beams = clamp(mp(8u), 0.0, 2.0);
    let ng = i32(u.num_g);

    let g0 = g_block.gamp[0];
    let base_angle = atan2(g0.y, g0.x) * crystal_lock + t * 0.006;
    var p = bragg_kaleido_rot(uv * (1.35 / zoom), base_angle * 0.16);
    if u.mouse_down >= 0.5 {
        p -= (u.mouse * 2.0 - 1.0) * vec2<f32>(u.aspect, 1.0) * 0.42;
    }

    var ray_trap = 10.0;
    var ring_trap = 10.0;
    var bragg_sum = 0.0;
    var phase_sum = 0.0;
    for (var i = 0; i < 14; i++) {
        if i >= iter_count { break; }
        let fi = f32(i);
        let idx = (i * 7 + 1) % ng;
        let ga = g_block.gamp[idx];
        let ga_angle = atan2(ga.y, ga.x);

        let sector = TAU / f32(symmetry);
        var angle = atan2(p.y, p.x) + ga_angle * 0.08 * crystal_lock;
        angle = abs(fract(angle / sector + 0.5) * sector - 0.5 * sector);
        let radius = length(p);
        p = radius * vec2<f32>(cos(angle), sin(angle));

        ray_trap = min(ray_trap, abs(p.y) / (1.0 + fi));
        ring_trap = min(ring_trap, abs(radius - (0.42 + 0.08 * sin(fi + ga.w * 2.0))));
        let wave = cos(dot(ga.xy, p) * (1.0 + 0.13 * fi) + ga.z * 0.22 + t * 0.025);
        bragg_sum += wave * ga.w / (1.0 + 0.22 * fi);
        phase_sum += wave * 0.13;

        let offset = vec2<f32>(0.68 + 0.08 * ga.w, 0.37 + 0.05 * sin(ga_angle));
        p = abs(p * inflation - offset);
        p = bragg_kaleido_rot(p + vec2<f32>(wave, -wave) * bend * 0.08,
                              0.17 + ga_angle * 0.06 * crystal_lock);
    }

    let rays = exp(-mix(22.0, 95.0, beams * 0.5) * ray_trap);
    let rings = exp(-44.0 * ring_trap);
    let lattice = 0.5 + 0.5 * sin(bragg_sum * 2.4 + phase_sum);
    let pal = bragg_kaleido_palette(fract(hue + phase_sum * 0.05 + lattice * 0.2 + t * 0.003));

    var col = vec3<f32>(0.005, 0.002, 0.016);
    col += mix(pal, u.crystal_color.xyz, 0.12) * lattice * 0.32;
    col += bragg_kaleido_palette(hue + 0.22 + phase_sum * 0.03) * rays * (0.42 + beams * 0.72);
    col += vec3<f32>(1.0, 0.56, 0.18) * rings * bend * 0.38;
    return col;
}
