// ── JULIA LATTICE — reciprocal-vector-seeded Julia orbit trap ────────────

fn julia_lattice_rot(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn julia_lattice_palette(h: f32) -> vec3<f32> {
    let c = 0.5 + 0.5 * cos(TAU * (h + vec3<f32>(0.00, 0.34, 0.67)));
    return c * c;
}

fn render_julia_lattice(uv: vec2<f32>) -> vec3<f32> {
    if u.num_g == 0u { return vec3<f32>(0.008, 0.003, 0.02); }

    let g0 = g_block.gamp[0];
    let g1 = g_block.gamp[min(1, i32(u.num_g) - 1)];
    let seed_gain = clamp(mp(0u), 0.0, 2.0);
    let t = u.time * mp(1u);
    let breathe = clamp(mp(2u), 0.0, 1.0);
    let iter_count = clamp(i32(round(mp(3u))), 12, 42);
    let hue = mp(4u);
    let zoom = max(mp(5u), 0.2);
    let trap_mix = clamp(mp(6u), 0.0, 2.0);
    let glow = clamp(mp(7u), 0.0, 2.0);
    let crystal_twist = clamp(mp(8u), 0.0, 2.0);

    let ga = atan2(g0.y, g0.x);
    let gb = atan2(g1.z + 0.001, length(g1.xy));
    let crystal_seed = vec2<f32>(
        sin(ga * 1.7 + gb + g0.w * 2.1),
        cos(ga - gb * 1.3 + g1.w * 1.9)
    );
    let base_c = mix(vec2<f32>(-0.78, 0.145), vec2<f32>(-0.42, 0.61), breathe);
    let c = base_c + crystal_seed * seed_gain * 0.115
            + vec2<f32>(cos(t * 0.071), sin(t * 0.053)) * 0.045 * breathe;

    var z = julia_lattice_rot(uv * (1.42 / zoom), ga * crystal_twist * 0.18 + t * 0.009);
    if u.mouse_down >= 0.5 {
        z += (u.mouse * 2.0 - 1.0) * vec2<f32>(u.aspect, 1.0) * 0.24;
    }

    var escaped = 0.0;
    var smooth_i = 0.0;
    var trap = 10.0;
    var cross_trap = 10.0;
    for (var i = 0; i < 42; i++) {
        if i >= iter_count { break; }
        let x = z.x * z.x - z.y * z.y + c.x;
        let y = 2.0 * z.x * z.y + c.y;
        z = vec2<f32>(x, y);
        let r2 = dot(z, z);
        trap = min(trap, abs(sqrt(r2) - (0.42 + 0.12 * sin(ga * 2.0))));
        cross_trap = min(cross_trap, min(abs(z.x), abs(z.y)));
        if r2 > 64.0 {
            escaped = 1.0;
            smooth_i = f32(i) + 1.0 - log2(max(log2(r2), 0.001));
            break;
        }
        smooth_i = f32(i);
    }

    let fi = smooth_i / max(f32(iter_count), 1.0);
    let orbit = exp(-mix(9.0, 34.0, clamp(trap_mix * 0.5, 0.0, 1.0)) * trap);
    let cross = exp(-38.0 * cross_trap);
    let interior = 1.0 - escaped;
    let pal = julia_lattice_palette(fract(hue + fi * 1.7 + trap * 0.4 + t * 0.006));
    let crystal_pal = mix(pal, u.crystal_color.xyz, 0.16);

    var col = vec3<f32>(0.006, 0.002, 0.018);
    col += crystal_pal * (0.18 + 0.82 * fi) * escaped * 0.72;
    col += julia_lattice_palette(hue + 0.18) * orbit * (0.55 + glow * 0.75);
    col += u.crystal_color.xyz * cross * trap_mix * 0.32;
    col += julia_lattice_palette(hue + t * 0.004) * interior * (0.16 + orbit * 0.75);
    return col;
}
