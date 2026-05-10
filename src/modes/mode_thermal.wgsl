// Mode 26: THERMAL - Debye-Waller blur and heat-haze softened lattice peaks.

fn thermal_hue_shift(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

fn thermal_peak(k: vec2<f32>, center: vec2<f32>, width: f32) -> f32 {
    let d = k - center;
    return exp(-dot(d, d) / max(width * width, 1e-4));
}

fn render_thermal(uv: vec2<f32>) -> vec3<f32> {
    let temp = clamp(u.iso_level, 0.0, 1.0);
    let t = u.time * u.speed;
    let k = uv * (3.4 / max(u.zoom, 0.08));
    let haze = vec2<f32>(
        sin(uv.y * 18.0 + t * 1.7),
        cos(uv.x * 16.0 - t * 1.3)
    ) * (0.015 + 0.055 * temp);
    let kh = k + haze;

    var intensity = 0.0;
    var sharp = 0.0;
    var soft = 0.0;
    for (var i = 0; i < 96; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let g = ga.xy * u.kscale * 0.38;
        let amp = ga.w;
        let g2 = dot(ga.xyz, ga.xyz);
        let dw = exp(-temp * g2 * 0.065);
        let breathe = 1.0 + 0.12 * sin(t * 1.4 + f32(i) * 0.73);
        let width = (0.018 + temp * 0.080) * breathe;
        let p1 = thermal_peak(kh, g, width);
        let p2 = thermal_peak(kh, -g, width);
        let pk = (p1 + p2) * amp * amp * dw;
        intensity += pk;
        sharp += (p1 + p2) * amp * amp * (1.0 - temp);
        soft += pk * temp;
    }

    let field = crystal_field(vec3<f32>(uv * 1.9 / max(u.zoom, 0.08), t * 0.12));
    let shimmer = 0.5 + 0.5 * sin(field * 3.0 + t * 2.0);
    let base = thermal_hue_shift(u.crystal_color.xyz, u.color_shift);
    var col = vec3<f32>(0.010, 0.012, 0.020);
    col += base * intensity * (1.5 + temp);
    col += vec3<f32>(0.95, 0.62, 0.24) * soft * 1.3;
    col += vec3<f32>(0.72, 0.90, 1.00) * sharp * 0.8;
    col += shimmer * temp * vec3<f32>(0.16, 0.10, 0.04);
    col *= 1.0 + 0.25 * sin(t + length(uv) * 8.0) * temp;
    return col;
}
