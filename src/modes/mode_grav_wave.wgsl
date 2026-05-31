// Mode: GRAV WAVE -- binary inspiral strain field and interferometer fringes
//
// A compact binary emits a transverse-traceless metric perturbation. We draw
// the plus/cross strain as an animated quadrupole that distorts a test grid,
// with Michelson-style fringes showing the differential arm-length shift.

fn grav_wave_hue_shift(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k  = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

fn grav_wave_binary(p: vec2<f32>, phase: f32, sep: f32) -> vec3<f32> {
    let ax = vec2<f32>(cos(phase), sin(phase));
    let a  = ax * sep;
    let d1 = length(p - a);
    let d2 = length(p + a);
    let m1 = smoothstep(0.12, 0.0, d1);
    let m2 = smoothstep(0.12, 0.0, d2);
    let bridge = exp(-abs(dot(p, vec2<f32>(-ax.y, ax.x))) * 7.0)
               * smoothstep(sep * 1.15, 0.0, abs(dot(p, ax)));
    return vec3<f32>(m1 + m2 + bridge * 0.22, m1, m2);
}

fn render_grav_wave(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * mp(MP_SPEED);
    let p = uv * (2.35 / max(mp(MP_ZOOM), 0.08));
    let r = length(p);
    let ang = atan2(p.y, p.x);

    let cycle = fract(t * 0.032);
    let chirp = cycle * cycle * (3.0 - 2.0 * cycle);
    let freq = mix(5.0, 24.0, chirp) * mix(0.75, 1.45, mp(MP_FIELD_MIX));
    let phase = t * mix(0.75, 3.5, chirp) + mp(MP_COLOR_SHIFT) * TAU;
    let sep = mix(0.78, 0.18, chirp);
    let amp = (0.035 + 0.16 * mp(MP_ISO_LEVEL)) * (0.35 + 1.2 * chirp);

    let retarded = phase - r * freq * 0.42;
    let envelope = 1.0 / (1.0 + r * 0.85);
    let hp = amp * envelope * cos(2.0 * (retarded - ang));
    let hx = amp * envelope * sin(2.0 * (retarded - ang));

    let q = p + vec2<f32>(hp * p.x + hx * p.y, hx * p.x - hp * p.y);

    let grid_x = smoothstep(0.030, 0.0, abs(fract(q.x * 4.5 + 0.5) - 0.5));
    let grid_y = smoothstep(0.030, 0.0, abs(fract(q.y * 4.5 + 0.5) - 0.5));
    let grid = max(grid_x, grid_y) * smoothstep(2.9, 0.3, r);

    let wave = 0.5 + 0.5 * cos(freq * r - phase * 2.0);
    let wave_ring = pow(wave, mix(5.0, 14.0, mp(MP_ISO_LEVEL))) * smoothstep(2.8, 0.25, r);

    let arm_w = 0.035 + 0.025 * mp(MP_FIELD_MIX);
    let arm_x = smoothstep(arm_w, 0.0, abs(p.y)) * smoothstep(1.75, 0.10, abs(p.x));
    let arm_y = smoothstep(arm_w, 0.0, abs(p.x)) * smoothstep(1.75, 0.10, abs(p.y));
    let diff = hp * cos(2.0 * mp(MP_COLOR_SHIFT) * TAU) + hx * sin(2.0 * mp(MP_COLOR_SHIFT) * TAU);
    let fringes = 0.5 + 0.5 * cos((p.x - p.y) * 22.0 + diff * 220.0 + t * 0.7);
    let interferometer = (arm_x + arm_y) * (0.28 + 0.72 * fringes);

    let bin = grav_wave_binary(p, phase, sep);
    let merge_flash = exp(-pow((cycle - 0.92) * 18.0, 2.0)) * exp(-r * 2.0);

    let cf = 0.5 + 0.5 * crystal_field(vec3<f32>(p * 1.7, t * 0.08));
    var base = mix(vec3<f32>(0.010, 0.014, 0.030), vec3<f32>(0.035, 0.018, 0.050), cf * 0.45);
    var strain_col = mix(vec3<f32>(0.22, 0.58, 1.0), vec3<f32>(1.0, 0.58, 0.22), smoothstep(-amp, amp, hp));
    strain_col = grav_wave_hue_shift(strain_col, mp(MP_COLOR_SHIFT) * 0.35);

    var col = base;
    col += grid * vec3<f32>(0.10, 0.16, 0.22);
    col += wave_ring * strain_col * 0.55;
    col += interferometer * mix(vec3<f32>(0.45, 0.95, 1.0), vec3<f32>(1.0, 0.78, 0.35), fringes) * 0.75;
    col += bin.x * vec3<f32>(1.0, 0.72, 0.42) * 1.15;
    col += bin.y * u.crystal_color.xyz * 0.45;
    col += bin.z * grav_wave_hue_shift(u.crystal_color.xyz, 0.33) * 0.45;
    col += merge_flash * vec3<f32>(1.0, 0.86, 0.58) * 1.4;
    col += abs(diff) * 4.0 * (arm_x + arm_y) * vec3<f32>(0.55, 0.75, 1.0);

    return col;
}
