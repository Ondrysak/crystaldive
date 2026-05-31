// Mode: SPIN_TEXTURE - reciprocal-space spin orientation field with Rashba winding.

fn spin_texture_hue(h: f32) -> vec3<f32> {
    return 0.5 + 0.5 * cos(TAU * (h + mp(MP_COLOR_SHIFT) + vec3<f32>(0.00, 0.33, 0.67)));
}

fn spin_texture_hash21(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(123.34, 456.21));
    return fract((q.x + q.y) * (q.x + 34.45 + q.y));
}

fn spin_texture_spin(k: vec2<f32>) -> vec3<f32> {
    let t = u.time * mp(MP_SPEED) * 0.18;
    let r = length(k);
    let inv_r = 1.0 / max(r, 1e-4);
    let radial = k * inv_r;
    let tangent = vec2<f32>(-radial.y, radial.x);

    let p = vec3<f32>(k, t);
    let f0 = crystal_field(p * 0.72 + vec3<f32>(0.0, 0.0, 0.2 * t));
    let f1 = cf2(p * 0.58 + vec3<f32>(0.31, 0.17, 0.0));

    let winding = 1.0 + floor(mp(MP_FIELD_MIX) * 3.99);
    let a = atan2(k.y, k.x) * winding
          + 0.75 * f0
          + 0.20 * sin(r * (3.5 + 7.0 * mp(MP_ISO_LEVEL)) - t);
    let rashba = vec2<f32>(-sin(a), cos(a));

    let bloch_mix = 0.25 + 0.65 * mp(MP_W_MOTIF) / max(mp(MP_W_LATTICE) + mp(MP_W_MOTIF) + mp(MP_W_BAND), 1e-3);
    let xy = normalize(mix(tangent, mix(rashba, radial, 0.30 + 0.35 * mp(MP_FIELD_MIX)), bloch_mix)
                     + 0.12 * vec2<f32>(f1, -f0));

    let core_phase = r * (5.5 + 7.0 * mp(MP_ISO_LEVEL)) - t * 1.4 + f0 * 2.0;
    let z = tanh((0.42 * cos(core_phase) + 0.58 * f1) * 2.1);
    let in_plane = sqrt(max(0.0, 1.0 - z * z));
    return normalize(vec3<f32>(xy * in_plane, z));
}

fn spin_texture_dash(k: vec2<f32>, spin: vec3<f32>) -> f32 {
    let density = 11.0 + 13.0 * mp(MP_ISO_LEVEL);
    let grid = k * density;
    let cell = floor(grid);
    let jitter = vec2<f32>(
        spin_texture_hash21(cell + vec2<f32>(17.0, 3.0)),
        spin_texture_hash21(cell + vec2<f32>(5.0, 29.0))
    ) - vec2<f32>(0.5);
    let center = (cell + vec2<f32>(0.5) + jitter * 0.38) / density;

    let local = spin_texture_spin(center);
    let dir = normalize(local.xy + vec2<f32>(1e-4, 0.0));
    let side = vec2<f32>(-dir.y, dir.x);
    let d = k - center;
    let along = dot(d, dir);
    let across = abs(dot(d, side));

    let half_len = (0.035 + 0.025 * mp(MP_FIELD_MIX)) / max(mp(MP_ZOOM), 0.25);
    let half_wid = 0.0065 / max(mp(MP_ZOOM), 0.25);
    let body = smoothstep(half_len, half_len * 0.45, abs(along));
    let width = 1.0 - smoothstep(half_wid * 0.35, half_wid, across);
    let phase = 0.65 + 0.35 * sin((along * density * 9.0) + u.time * mp(MP_SPEED) * 1.7);
    return body * width * phase * smoothstep(0.10, 0.45, length(spin.xy));
}

fn render_spin_texture(uv: vec2<f32>) -> vec3<f32> {
    let k = uv * (2.2 / max(mp(MP_ZOOM), 0.05));
    let spin = spin_texture_spin(k);

    let angle = atan2(spin.y, spin.x) / TAU + 0.5;
    let angle_col = spin_texture_hue(angle);
    let up_col = vec3<f32>(1.00, 0.88, 0.48);
    let down_col = vec3<f32>(0.20, 0.72, 1.05);
    let polarity_col = mix(down_col, up_col, 0.5 + 0.5 * spin.z);

    let r = length(k);
    let ring = 0.5 + 0.5 * sin(r * (13.0 + 12.0 * mp(MP_ISO_LEVEL)) - u.time * mp(MP_SPEED) * 0.55);
    let contour = smoothstep(0.76, 1.0, ring) * 0.18;
    let chirality = abs(dot(normalize(k + vec2<f32>(1e-4, 0.0)), vec2<f32>(spin.y, -spin.x)));

    var col = mix(angle_col, polarity_col, 0.38 + 0.34 * abs(spin.z));
    col = mix(col, u.crystal_color.xyz, 0.18);
    col *= 0.42 + 0.50 * smoothstep(0.0, 1.0, length(spin.xy)) + contour;
    col += chirality * vec3<f32>(0.05, 0.16, 0.18);

    let dash = spin_texture_dash(k, spin);
    let ink = mix(vec3<f32>(0.02, 0.03, 0.04), vec3<f32>(1.05, 1.00, 0.82), smoothstep(-0.20, 0.55, spin.z));
    col = mix(col, ink, dash * (0.58 + 0.22 * abs(spin.z)));

    let core = smoothstep(0.80, 0.98, abs(spin.z)) * (1.0 - smoothstep(0.0, 2.6, r));
    col += core * mix(down_col, up_col, 0.5 + 0.5 * spin.z) * 0.75;

    let bg = mix(vec3<f32>(0.006, 0.010, 0.018), u.crystal_color.xyz * 0.055, 0.45);
    let mask = smoothstep(3.0, 0.45, r);
    return mix(bg, col, mask);
}
