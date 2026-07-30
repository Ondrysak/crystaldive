// ── THERMAL DIFFUSE — phenomenological one-phonon diffuse scattering ──────
//
// The loaded reciprocal lattice supplies the Bragg centres. Around each centre
// we draw a temperature-weighted, anisotropic halo. Its longitudinal/transverse
// shape is a visual proxy for the one-phonon factor |Q·e_{qν}|²/ων; a full
// material calculation would replace this proxy with DFPT phonon eigenvectors.

fn thermal_diffuse_rotate(g: vec3<f32>, az: f32, el: f32) -> vec3<f32> {
    let ca = cos(az); let sa = sin(az);
    let ce = cos(el); let se = sin(el);
    let x = g.x * ca - g.z * sa;
    let z = g.x * sa + g.z * ca;
    return vec3<f32>(x, g.y * ce - z * se, g.y * se + z * ce);
}

fn render_thermal_diffuse(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * mp(MP_SPEED);
    var az = t * 0.17;
    var el = 0.34 + 0.12 * sin(t * 0.11);
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.3;
    }

    let temperature = mp(MP_FIELD_MIX);
    let soft_mode   = mp(MP_ISO_LEVEL);
    let detector    = 0.31 * mp(MP_KSCALE) / max(mp(MP_ZOOM), 0.05);
    let longitudinal = mp(MP_W_LATTICE);
    let diffuse_gain = mp(MP_W_MOTIF);
    let sharpness    = mp(MP_W_BAND);
    let q = uv;

    var diffuse = 0.0;
    var bragg = 0.0;
    var direction = vec3<f32>(0.0);
    let ng_td = i32(u.num_g);
    for (var i = 0i; i < ng_td; i++) {
        let ga = g_block.gamp[i];
        let g = thermal_diffuse_rotate(ga.xyz, az, el);
        let spot = g.xy * detector;
        let d = q - spot;
        let gxy_len = max(length(g.xy), 1e-4);
        let ghat = g.xy / gxy_len;
        let parallel = dot(d, ghat);
        let transverse = dot(d, vec2<f32>(-ghat.y, ghat.x));

        // A soft branch is bright and broad. Directional widths emulate the
        // Q·e polarization selection rule without claiming material-specific
        // phonon eigenvectors from the structural input alone.
        let omega = 0.10 + soft_mode * 0.72 + 0.14 * length(g);
        let w_parallel = 0.010 + 0.050 * longitudinal / max(omega, 0.05);
        let w_transverse = 0.007 + 0.038 * (2.0 - longitudinal) / max(omega, 0.05);
        let halo = exp(-0.5 * (
            parallel * parallel / max(w_parallel * w_parallel, 1e-6) +
            transverse * transverse / max(w_transverse * w_transverse, 1e-6)
        ));
        let core_w = 0.0025 + 0.009 * (1.0 - sharpness * 0.5);
        let core = exp(-dot(d, d) / max(core_w * core_w, 1e-7));
        let population = 0.08 + 2.8 * temperature;
        let intensity = ga.w * population / omega;
        diffuse += intensity * halo * diffuse_gain;
        bragg += ga.w * core * (0.25 + 0.75 * sharpness);
        direction += vec3<f32>(abs(ghat.x), abs(ghat.y), abs(g.z) / max(length(g), 1e-4)) * halo * ga.w;
    }

    let dir = normalize(direction + vec3<f32>(1e-4));
    let cool = mix(vec3<f32>(0.04, 0.22, 0.85), u.crystal_color.xyz, 0.42);
    let warm = mix(vec3<f32>(1.00, 0.20, 0.06), u.crystal_color.xyz, 0.25);
    let spectral = mix(cool, warm, dir.x * 0.55 + dir.z * 0.35);
    let bg = vec3<f32>(0.003, 0.005, 0.016) + u.crystal_color.xyz * 0.010;
    var col = bg + spectral * diffuse * 0.78;
    col += vec3<f32>(1.0, 0.93, 0.78) * bragg * 1.15;

    // A detector rim and beam-stop make the reciprocal-space readout explicit.
    let r = length(uv);
    let rim = exp(-pow((r - 0.94) * 24.0, 2.0));
    col += u.crystal_color.xyz * rim * 0.035;
    col *= smoothstep(0.018, 0.042, r);
    return col;
}
