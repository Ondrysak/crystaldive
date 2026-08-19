// ── MULTIBROT FIELD — crystal-perturbed z^n + c escape-time fractal ───────

fn multibrot_field_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn multibrot_field_pow(z: vec2<f32>, n: i32) -> vec2<f32> {
    var r = vec2<f32>(1.0, 0.0);
    for (var i = 0; i < 6; i++) {
        if i >= n { break; }
        r = multibrot_field_mul(r, z);
    }
    return r;
}

fn multibrot_field_rot(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn multibrot_field_palette(h: f32) -> vec3<f32> {
    let c = 0.51 + 0.49 * cos(TAU * (h + vec3<f32>(0.00, 0.29, 0.64)));
    return c * c;
}

fn render_multibrot_field(uv: vec2<f32>) -> vec3<f32> {
    if u.num_g == 0u { return vec3<f32>(0.005, 0.002, 0.016); }

    let power = clamp(i32(round(mp(0u))), 2, 6);
    let t = u.time * mp(1u);
    let crystal_warp = clamp(mp(2u), 0.0, 1.0);
    let iter_count = clamp(i32(round(mp(3u))), 16, 48);
    let hue = mp(4u);
    let zoom = max(mp(5u), 0.2);
    let orbit_mix = clamp(mp(6u), 0.0, 2.0);
    let drift = clamp(mp(7u), 0.0, 1.0);
    let interior_glow = clamp(mp(8u), 0.0, 2.0);

    let g0 = g_block.gamp[0];
    let g1 = g_block.gamp[min(1, i32(u.num_g) - 1)];
    let orient = atan2(g0.y, g0.x) + 0.25 * atan2(g1.z, length(g1.xy) + 0.001);
    let crystal_offset = vec2<f32>(
        sin(g0.x * 0.37 + g1.y * 0.21 + g0.w),
        cos(g0.y * 0.31 - g1.x * 0.27 + g1.w)
    ) * crystal_warp * 0.13;

    var c = multibrot_field_rot(uv * (2.15 / zoom), orient * 0.13);
    c += vec2<f32>(-0.42, 0.0) + crystal_offset;
    c += vec2<f32>(cos(t * 0.021), sin(t * 0.017)) * drift * 0.035;
    if u.mouse_down >= 0.5 {
        c += (u.mouse * 2.0 - 1.0) * vec2<f32>(u.aspect, 1.0) * 0.26;
    }

    var z = vec2<f32>(0.0);
    var escaped = 0.0;
    var smooth_i = 0.0;
    var orbit = 10.0;
    var stripe = 0.0;
    for (var i = 0; i < 48; i++) {
        if i >= iter_count { break; }
        z = multibrot_field_pow(z, power) + c;
        let r2 = dot(z, z);
        orbit = min(orbit, abs(length(z) - (0.55 + 0.08 * sin(orient * 2.0))));
        stripe += sin(atan2(z.y, z.x) * f32(power) + t * 0.01);
        if r2 > 256.0 {
            escaped = 1.0;
            let log_zn = log(max(r2, 1.001)) * 0.5;
            smooth_i = f32(i) + 1.0 - log(max(log_zn, 0.001)) / log(f32(power));
            break;
        }
        smooth_i = f32(i);
    }

    let fi = smooth_i / max(f32(iter_count), 1.0);
    let trap = exp(-mix(7.0, 34.0, orbit_mix * 0.5) * orbit);
    let stripe_col = 0.5 + 0.5 * sin(stripe * 0.18);
    let interior = 1.0 - escaped;
    let pal = multibrot_field_palette(fract(hue + fi * 1.9 + stripe_col * 0.16 + t * 0.002));

    var col = vec3<f32>(0.004, 0.002, 0.014);
    col += mix(pal, u.crystal_color.xyz, 0.14) * escaped * (0.12 + 0.66 * fi);
    col += multibrot_field_palette(hue + orbit + 0.2) * trap * orbit_mix * 0.72;
    col += multibrot_field_palette(hue + orient / TAU) * interior * interior_glow * (0.12 + trap * 0.5);
    return col;
}
