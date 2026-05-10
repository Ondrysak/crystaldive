// Mode 27: DOMAIN_WALL - ferroic domains separated by moving phase walls.

fn domain_wall_rot(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn domain_wall_palette(x: f32, wall: f32) -> vec3<f32> {
    let a = vec3<f32>(0.16, 0.38, 1.05);
    let b = vec3<f32>(1.05, 0.28, 0.20);
    let c = vec3<f32>(0.92, 0.82, 0.34);
    return mix(mix(a, b, x), c, wall * 0.45);
}

fn render_domain_wall(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed * 0.45;
    let p = uv * (2.4 / max(u.zoom, 0.08));
    let wall_field =
        sin(p.x * (1.15 + u.field_mix) + sin(p.y * 1.7 + t) * 0.7 + t) +
        0.55 * sin(dot(p, vec2<f32>(-0.8, 1.25)) * 1.5 - t * 0.7);
    let domain = smoothstep(-0.09, 0.09, wall_field);
    let wall = 1.0 - smoothstep(0.0, 0.18 + 0.10 * u.iso_level, abs(wall_field));

    let pa = domain_wall_rot(p, 0.33 + u.color_shift * TAU * 0.20);
    let pb = domain_wall_rot(p, -0.55 - u.color_shift * TAU * 0.15);
    let fa = crystal_field(vec3<f32>(pa, t * 0.18));
    let fb = cf2(vec3<f32>(pb + vec2<f32>(0.35, -0.2), -t * 0.12));
    let order = mix(fa, fb, domain);
    let stripes = 0.5 + 0.5 * sin(order * 3.0 + p.x * 3.0 - p.y * 1.5);

    var col = domain_wall_palette(domain, wall);
    col *= 0.35 + 0.65 * smoothstep(-0.8, 0.9, order);
    col += wall * vec3<f32>(1.15, 0.95, 0.62) * (0.8 + 0.5 * stripes);
    col += u.crystal_color.xyz * (0.18 + 0.35 * stripes) * (1.0 - wall * 0.3);
    return col;
}
