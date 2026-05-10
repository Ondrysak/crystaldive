// Mode: ORBITAL - real-space atomic orbital lobes shaped by crystal field

fn orbital_rot_y(p: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

fn orbital_rot_x(p: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

fn orbital_basis(p: vec3<f32>) -> vec4<f32> {
    let r2 = max(dot(p, p), 1e-5);
    let inv_r = inverseSqrt(r2);
    let n = p * inv_r;

    let px = n.x;
    let py = n.y;
    let dz2 = 0.5 * (3.0 * n.z * n.z - 1.0);
    let dx2y2 = n.x * n.x - n.y * n.y;

    return vec4<f32>(px, py, dz2, dx2y2);
}

fn orbital_wave(p: vec3<f32>) -> f32 {
    let f = crystal_field(p * 0.85);
    let f2 = cf2(p * 0.65 + vec3<f32>(0.17, -0.11, 0.23));
    let hybrid = clamp(u.field_mix, 0.0, 1.0);
    let shimmer = 0.5 + 0.5 * sin(u.time * u.speed * 2.2 + f * 2.4 + f2 * 1.7);

    let b = orbital_basis(p);
    let p_like = mix(b.x, b.y, 0.35 + 0.30 * sin(f2 + u.color_shift));
    let d_like = mix(b.z, b.w, 0.5 + 0.5 * sin(f + u.color_shift * TAU));
    let psi = mix(p_like, d_like, hybrid);

    let r2 = dot(p, p);
    let radial_node = 0.58 - 0.19 * r2 + 0.045 * f;
    let envelope = exp(-r2 * (0.48 + 0.22 * hybrid));
    let cf_mod = 1.0 + 0.22 * f + 0.12 * shimmer;

    return psi * radial_node * envelope * cf_mod;
}

fn orbital_density(p: vec3<f32>) -> f32 {
    let psi = orbital_wave(p);
    let shell = 0.08 * exp(-dot(p, p) * 0.22) * abs(crystal_field(p * 0.5));
    return psi * psi + shell;
}

fn orbital_sdf(p: vec3<f32>) -> f32 {
    let iso = mix(0.025, 0.18, clamp(u.iso_level, 0.0, 1.0));
    return iso - orbital_density(p);
}

fn orbital_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.004;
    return normalize(vec3<f32>(
        orbital_sdf(p + vec3<f32>(e, 0.0, 0.0)) - orbital_sdf(p - vec3<f32>(e, 0.0, 0.0)),
        orbital_sdf(p + vec3<f32>(0.0, e, 0.0)) - orbital_sdf(p - vec3<f32>(0.0, e, 0.0)),
        orbital_sdf(p + vec3<f32>(0.0, 0.0, e)) - orbital_sdf(p - vec3<f32>(0.0, 0.0, e))
    ));
}

fn render_orbital(uv: vec2<f32>) -> vec3<f32> {
    var az = u.time * u.speed * 0.15;
    var el = 0.4;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.5;
    }

    let cp = vec3<f32>(sin(az) * cos(el), sin(el), cos(az) * cos(el)) * (3.4 / u.zoom);
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), fwd));
    let up = cross(fwd, rgt);
    let rd = normalize(fwd + uv.x * rgt + uv.y * up);

    var t = 0.0;
    var hit = false;
    var hit_p = vec3<f32>(0.0);
    var hit_i = 0;
    for (var i = 0; i < 84; i++) {
        var p = cp + rd * t;
        p = orbital_rot_y(p, 0.22 * sin(u.time * u.speed * 0.35));
        p = orbital_rot_x(p, 0.18 * cos(u.time * u.speed * 0.27));

        let d = orbital_sdf(p);
        if d < 0.0025 {
            hit = true;
            hit_p = p;
            hit_i = i;
            break;
        }
        if t > 8.5 { break; }
        t += max(d * 0.72, 0.006);
    }

    var col = vec3<f32>(0.006, 0.008, 0.020);

    if hit {
        let nm = orbital_normal(hit_p);
        let psi = orbital_wave(hit_p);
        let f = crystal_field(hit_p * 0.9);
        let phase_mix = smoothstep(-0.035, 0.035, psi);
        let neg_col = vec3<f32>(0.12, 0.48, 1.15);
        let pos_col = vec3<f32>(1.20, 0.26, 0.18);
        var surf = mix(neg_col, pos_col, phase_mix);
        surf = mix(surf, u.crystal_color.xyz, 0.22 + 0.12 * abs(f));

        let light = normalize(vec3<f32>(1.2, 1.8, 0.9));
        let diff = max(dot(nm, light), 0.0);
        let rim = pow(1.0 - abs(dot(nm, -rd)), 2.2);
        let nodal = smoothstep(0.055, 0.0, abs(psi));
        let shimmer = 0.5 + 0.5 * sin(u.time * u.speed * 5.0 + f * 4.0);

        col = surf * (0.25 + 0.85 * diff);
        col += vec3<f32>(0.74, 0.94, 1.15) * rim * (0.35 + 0.30 * shimmer);
        col += vec3<f32>(1.0, 0.96, 0.78) * nodal * (0.28 + 0.25 * shimmer);
        col *= 0.62 + 0.38 * (1.0 - f32(hit_i) / 84.0);
    } else {
        var glow = 0.0;
        var node_trace = 0.0;
        for (var k = 0; k < 5; k++) {
            let tg = 0.55 + f32(k) * 0.75;
            let p = cp + rd * tg;
            let d = orbital_density(p);
            let w = orbital_wave(p);
            glow += d;
            node_trace += smoothstep(0.035, 0.0, abs(w)) * d;
        }
        col += u.crystal_color.xyz * glow * 0.12;
        col += vec3<f32>(0.18, 0.50, 1.00) * glow * 0.10;
        col += vec3<f32>(0.95, 0.88, 0.62) * node_trace * 0.20;
    }

    return col;
}
