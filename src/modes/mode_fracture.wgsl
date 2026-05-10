// Mode 28: FRACTURE - strained lattice around a crack front.

fn fracture_crack_y(x: f32, t: f32) -> f32 {
    return 0.16 * sin(x * 2.6 + t) + 0.05 * sin(x * 7.0 - t * 1.7);
}

fn fracture_strain(p: vec2<f32>, tip: vec2<f32>) -> vec2<f32> {
    let d = p - tip;
    let r = max(length(d), 0.035);
    let th = atan2(d.y, d.x);
    let amp = 0.18 / sqrt(r);
    return amp * vec2<f32>(cos(0.5 * th), -sin(0.5 * th));
}

fn render_fracture(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed * 0.35;
    let p = uv * (2.2 / max(u.zoom, 0.08));
    let tip = vec2<f32>(mix(-1.15, 0.65, u.field_mix), fracture_crack_y(mix(-1.15, 0.65, u.field_mix), t));
    let cy = fracture_crack_y(p.x, t);
    let behind = smoothstep(tip.x + 0.05, tip.x - 0.15, p.x);
    let crack = behind * (1.0 - smoothstep(0.015, 0.085 + 0.08 * u.iso_level, abs(p.y - cy)));

    let strain = fracture_strain(p, tip) * (1.0 - crack * 0.7);
    let warped = p + strain + vec2<f32>(0.0, sign(p.y - cy) * crack * 0.18);
    let f = crystal_field(vec3<f32>(warped, t * 0.15));
    let f2 = cf2(vec3<f32>(warped * vec2<f32>(1.08, 0.92), -t * 0.1));
    let lattice = smoothstep(0.15, 1.0, abs(sin(f * 2.6) * cos(f2 * 2.1)));

    let rtip = length(p - tip);
    let stress = exp(-rtip * 1.6) * (0.55 + 0.45 * sin(rtip * 28.0 - t * 5.0));
    var col = mix(vec3<f32>(0.015, 0.014, 0.020), u.crystal_color.xyz * 0.55, lattice);
    col += stress * vec3<f32>(1.2, 0.65, 0.22) * (0.7 + u.iso_level);
    col = mix(col, vec3<f32>(0.0, 0.0, 0.0), crack * 0.88);
    col += crack * vec3<f32>(0.18, 0.30, 0.45) * smoothstep(0.08, 0.0, abs(p.y - cy));
    return col;
}
