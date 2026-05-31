// Mode: CHARGE_DENSITY_WAVE - lattice-locked stripes with domains and phase slips

fn cdw_rot(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn cdw_safe_dir(g: vec2<f32>, fallback: vec2<f32>) -> vec2<f32> {
    let l = length(g);
    if l > 1e-4 {
        return g / l;
    }
    return fallback;
}

fn cdw_gdir(idx: i32, fallback: vec2<f32>) -> vec2<f32> {
    if idx < i32(u.num_g) {
        let g = g_block.gamp[idx].xy;
        return cdw_safe_dir(g, fallback);
    }
    return fallback;
}

fn cdw_hash(p: vec2<f32>) -> f32 {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)),
                      dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q.x + q.y) * 43758.5453);
}

fn cdw_domain(p: vec2<f32>) -> f32 {
    let slow = 0.55 + 0.45 * mp(MP_FIELD_MIX);
    let a = sin(p.x * (1.3 + slow) + 0.7 * sin(p.y * 1.1));
    let b = sin(dot(p, vec2<f32>(-0.8, 1.15)) * (1.0 + slow) + 1.7);
    let c = 0.35 * sin(crystal_field(vec3<f32>(p * 0.22, u.time * mp(MP_SPEED) * 0.025)));
    return a + 0.75 * b + c;
}

fn cdw_vortex_phase(p: vec2<f32>, center: vec2<f32>, charge: f32) -> f32 {
    let d = p - center;
    let core = smoothstep(0.02, 0.35, dot(d, d));
    return atan2(d.y, d.x) * charge * core;
}

fn cdw_palette(x: f32) -> vec3<f32> {
    let h = fract(x + mp(MP_COLOR_SHIFT));
    let pal = 0.54 + 0.46 * cos(TAU * (h + vec3<f32>(0.0, 0.31, 0.64)));
    return mix(pal, u.crystal_color.xyz, 0.34);
}

fn render_cdw(uv: vec2<f32>) -> vec3<f32> {
    let scale = (5.2 + 4.0 * mp(MP_KSCALE)) / max(mp(MP_ZOOM), 0.2);
    let t = u.time * mp(MP_SPEED);
    let p = uv * scale;

    let g0 = cdw_gdir(0, vec2<f32>(1.0, 0.0));
    let g1_raw = cdw_gdir(1, vec2<f32>(0.0, 1.0));
    let g1 = cdw_safe_dir(g1_raw - g0 * dot(g1_raw, g0), vec2<f32>(-g0.y, g0.x));
    let g2 = cdw_safe_dir(g0 + g1 * (0.65 + 0.35 * mp(MP_FIELD_MIX)), normalize(vec2<f32>(1.0, 1.0)));

    let domain_raw = cdw_domain(p * 0.36);
    let domain_sign = sign(domain_raw + 1e-4);
    let wall = 1.0 - smoothstep(0.02, 0.32, abs(domain_raw));

    let slip_line_a = dot(p, cdw_rot(g1, 0.2)) + 1.1 * sin(dot(p, g0) * 0.18);
    let slip_line_b = dot(p, cdw_rot(g0, -0.35)) - 0.8 * cos(dot(p, g1) * 0.16 + 1.4);
    let slip_a = smoothstep(-0.18, 0.18, slip_line_a) - 0.5;
    let slip_b = smoothstep(-0.16, 0.16, slip_line_b) - 0.5;

    let drift = t * 0.08;
    let phase_bias = TAU * (0.08 * mp(MP_ISO_LEVEL) + 0.04 * sin(t * 0.17));
    var phase = phase_bias;
    phase += domain_sign * (0.75 + 1.35 * mp(MP_FIELD_MIX));
    phase += (slip_a - 0.65 * slip_b) * TAU * (0.28 + 0.42 * mp(MP_ISO_LEVEL));
    phase += cdw_vortex_phase(p, vec2<f32>(-2.1, 1.0) + 0.25 * vec2<f32>(sin(drift), cos(drift * 1.3)), 1.0);
    phase += cdw_vortex_phase(p, vec2<f32>(1.7, -0.8) + 0.20 * vec2<f32>(cos(drift * 0.7), sin(drift)), -1.0);
    phase += 0.18 * cf2(vec3<f32>(p * 0.18, t * 0.035));

    let k = 2.8 + 4.2 * mp(MP_KSCALE);
    let primary = sin(dot(p, g0) * k + phase);
    let harmonic = sin(dot(p, g0) * k * 2.0 + phase * 1.9) * 0.28;
    let cross = sin(dot(p, g1) * k * (0.72 + 0.22 * mp(MP_FIELD_MIX)) - phase * 0.65) * 0.33;
    let lock = sin(dot(p, g2) * k * 0.54 + phase * 0.35) * 0.18;
    let density = primary + harmonic + cross + lock;

    let stripe = smoothstep(0.38, 0.98, density);
    let trough = smoothstep(0.72, 0.98, -density);
    let node = smoothstep(0.045, 0.0, abs(density));
    let compression = smoothstep(0.78, 1.0, primary * sin(dot(p, g1) * k * 0.44 - phase));

    let defect_a = exp(-dot(p - vec2<f32>(-2.1, 1.0), p - vec2<f32>(-2.1, 1.0)) * 2.2);
    let defect_b = exp(-dot(p - vec2<f32>(1.7, -0.8), p - vec2<f32>(1.7, -0.8)) * 2.0);
    let core = defect_a + defect_b;

    let bg = mix(vec3<f32>(0.010, 0.008, 0.016), u.crystal_color.xyz * 0.055, 0.55);
    let cold = mix(vec3<f32>(0.03, 0.05, 0.09), u.crystal_color.xyz * 0.22, 0.45);
    let warm = cdw_palette(density * 0.09 + domain_raw * 0.05);

    var col = bg;
    col = mix(col, cold, trough * 0.70);
    col = mix(col, warm * (0.65 + 0.60 * stripe), stripe);
    col += node * vec3<f32>(0.95, 0.88, 0.62) * (0.45 + 0.75 * mp(MP_ISO_LEVEL));
    col += wall * mix(vec3<f32>(0.15, 0.65, 0.95), u.crystal_color.xyz, 0.35) * 0.85;
    col += (smoothstep(0.19, 0.0, abs(slip_line_a)) + smoothstep(0.17, 0.0, abs(slip_line_b)))
         * vec3<f32>(1.00, 0.82, 0.44) * 0.35;
    col += compression * vec3<f32>(1.15, 0.96, 0.66) * 0.55;
    col += core * vec3<f32>(0.60, 0.92, 1.15) * 1.25;

    let grains = cdw_hash(floor(p * 1.7));
    col *= 0.92 + 0.14 * grains;

    return col;
}
