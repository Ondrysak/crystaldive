// ── NEWTON BASIN — crystal-rotated roots of z^n = a ─────────────────────

fn newton_basin_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn newton_basin_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let d = max(dot(b, b), 1e-8);
    return vec2<f32>(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / d;
}

fn newton_basin_pow(z: vec2<f32>, n: i32) -> vec2<f32> {
    var r = vec2<f32>(1.0, 0.0);
    for (var i = 0; i < 7; i++) {
        if i >= n { break; }
        r = newton_basin_mul(r, z);
    }
    return r;
}

fn newton_basin_palette(h: f32) -> vec3<f32> {
    let c = 0.52 + 0.48 * cos(TAU * (h + vec3<f32>(0.02, 0.35, 0.69)));
    return c * c;
}

fn render_newton_basin(uv: vec2<f32>) -> vec3<f32> {
    if u.num_g == 0u { return vec3<f32>(0.006, 0.002, 0.018); }

    let g0 = g_block.gamp[0];
    let g1 = g_block.gamp[min(1, i32(u.num_g) - 1)];
    let root_count = clamp(i32(round(mp(0u))), 3, 7);
    let t = u.time * mp(1u);
    let relax = clamp(mp(2u), 0.35, 1.35);
    let detail = clamp(i32(round(mp(3u))), 8, 28);
    let hue = mp(4u);
    let zoom = max(mp(5u), 0.2);
    let boundary = clamp(mp(6u), 0.0, 2.0);
    let crystal_bias = clamp(mp(7u), 0.0, 2.0);
    let pulse = clamp(mp(8u), 0.0, 1.0);

    let orient = atan2(g0.y, g0.x) + 0.35 * atan2(g1.z, length(g1.xy) + 0.001);
    let a_angle = orient * crystal_bias + t * 0.018 + g0.w - g1.w * 0.37;
    let a_radius = 0.82 + 0.18 * sin(t * 0.055) * pulse;
    let poly_a = a_radius * vec2<f32>(cos(a_angle), sin(a_angle));

    let ca = cos(orient * 0.17);
    let sa = sin(orient * 0.17);
    var z = vec2<f32>(ca * uv.x - sa * uv.y, sa * uv.x + ca * uv.y) * (1.75 / zoom);
    if u.mouse_down >= 0.5 {
        z += (u.mouse * 2.0 - 1.0) * vec2<f32>(u.aspect, 1.0) * 0.34;
    }

    var residual = 10.0;
    var min_step = 10.0;
    var used = 0.0;
    for (var i = 0; i < 28; i++) {
        if i >= detail { break; }
        let zn = newton_basin_pow(z, root_count);
        let znm1 = newton_basin_pow(z, root_count - 1);
        let f = zn - poly_a;
        let deriv = znm1 * f32(root_count);
        let step = newton_basin_div(f, deriv);
        z -= step * relax;
        residual = length(f);
        min_step = min(min_step, length(step));
        used = f32(i);
        if residual < 0.00008 { break; }
    }

    let root_angle = atan2(z.y, z.x) / TAU;
    let root_id = fract(root_angle * f32(root_count) - a_angle / TAU);
    let convergence = 1.0 - used / max(f32(detail), 1.0);
    let edge = exp(-mix(8.0, 42.0, clamp(boundary * 0.5, 0.0, 1.0)) * min_step);
    let root_glow = exp(-22.0 * abs(length(z) - pow(a_radius, 1.0 / f32(root_count))));

    let pal = newton_basin_palette(fract(hue + root_id + convergence * 0.24 + t * 0.003));
    var col = vec3<f32>(0.005, 0.003, 0.016);
    col += mix(pal, u.crystal_color.xyz, 0.14) * (0.22 + 0.62 * convergence);
    col += newton_basin_palette(hue + root_angle + 0.23) * edge * boundary * 0.62;
    col += vec3<f32>(1.0, 0.72, 0.28) * root_glow * 0.25;
    return col;
}
