// ── APOLLONIAN — crystal-directed recursive circle inversions ─────────────

fn apollonian_rot(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn apollonian_palette(h: f32) -> vec3<f32> {
    let c = 0.5 + 0.5 * cos(TAU * (h + vec3<f32>(0.00, 0.31, 0.63)));
    return c * c;
}

fn render_apollonian(uv: vec2<f32>) -> vec3<f32> {
    if u.num_g == 0u { return vec3<f32>(0.005, 0.002, 0.015); }

    let g0 = g_block.gamp[0];
    let g1 = g_block.gamp[min(1, i32(u.num_g) - 1)];
    let inversion = clamp(mp(0u), 0.65, 1.75);
    let t = u.time * mp(1u);
    let drift = clamp(mp(2u), 0.0, 1.0);
    let iter_count = clamp(i32(round(mp(3u))), 6, 18);
    let hue = mp(4u);
    let zoom = max(mp(5u), 0.2);
    let circle_glow = clamp(mp(6u), 0.0, 2.0);
    let lattice_pull = clamp(mp(7u), 0.0, 2.0);
    let fold = clamp(mp(8u), 0.0, 1.0);

    let a0 = atan2(g0.y, g0.x);
    let a1 = atan2(g1.y, g1.x);
    let dir0 = vec2<f32>(cos(a0), sin(a0));
    let dir1 = vec2<f32>(cos(a1 + 2.094), sin(a1 + 2.094));
    let dir2 = vec2<f32>(cos(a0 - a1 - 2.094), sin(a0 - a1 - 2.094));

    var p = apollonian_rot(uv * (1.55 / zoom), a0 * 0.15 + t * 0.006 * drift);
    if u.mouse_down >= 0.5 {
        p -= (u.mouse * 2.0 - 1.0) * vec2<f32>(u.aspect, 1.0) * 0.55;
    }

    var ring_acc = 0.0;
    var seam_acc = 0.0;
    var phase_sum = 0.0;
    for (var i = 0; i < 18; i++) {
        if i >= iter_count { break; }
        let fi = f32(i);
        let branch = i % 3;
        var centre = dir0;
        if branch == 1 { centre = dir1; }
        if branch == 2 { centre = dir2; }
        centre *= 0.38 + 0.09 * sin(fi * 1.7 + t * 0.023);

        if fold > 0.001 {
            p = mix(p, abs(p), fold * 0.66);
        }
        let q = p - centre;
        let r2 = max(dot(q, q), 0.004);
        let radius = sqrt(r2);
        let ring_r = 0.24 + 0.07 * sin(fi * 1.618 + a0 - a1);
        let level_weight = 1.0 / (1.0 + 0.20 * fi);
        ring_acc += exp(-mix(34.0, 96.0, circle_glow * 0.5)
                        * abs(radius - ring_r)) * level_weight;
        seam_acc += exp(-52.0 * min(abs(q.x), abs(q.y))) * level_weight;
        phase_sum += atan2(q.y, q.x) * 0.055;

        let inv = clamp(inversion / r2, 0.35, 4.8);
        p = centre + q * inv;
        // Keep the inverted orbit bounded while preserving every nested copy.
        p = fract((p + vec2<f32>(1.0)) * 0.5) * 2.0 - vec2<f32>(1.0);
        p = apollonian_rot(p * (0.94 + 0.035 * lattice_pull),
                           (a0 + a1) * 0.035 + fi * 0.19 + t * 0.002 * drift);
    }

    let rings = 1.0 - exp(-ring_acc * (0.58 + circle_glow * 0.35));
    let seams = (1.0 - exp(-seam_acc * 0.42)) * fold;
    let cell = 0.5 + 0.5 * sin(length(p) * 8.0 + phase_sum * 1.7);
    let pal = apollonian_palette(fract(hue + phase_sum / TAU + cell * 0.17 + t * 0.004));

    var col = vec3<f32>(0.004, 0.002, 0.014);
    col += mix(pal, u.crystal_color.xyz, 0.14) * cell * 0.16;
    col += apollonian_palette(hue + 0.16 + phase_sum * 0.03)
           * rings * (0.32 + circle_glow * 0.48);
    col += vec3<f32>(0.25, 0.72, 1.0) * seams * lattice_pull * 0.22;
    return col;
}
