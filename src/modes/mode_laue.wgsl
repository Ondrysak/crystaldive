// ── LAUE CONSTELLATION — polychromatic reciprocal-lattice spot detector ───
//
// Each loaded G-vector is rotated with the crystal and tested against a broad
// Ewald-shell proxy. The detector receives discrete Laue-like spots instead of
// Kikuchi bands, making crystal symmetry legible as a moving constellation.

fn laue_rotate(g: vec3<f32>, az: f32, el: f32) -> vec3<f32> {
    let ca = cos(az); let sa = sin(az);
    let ce = cos(el); let se = sin(el);
    let x = g.x * ca - g.z * sa;
    let z = g.x * sa + g.z * ca;
    return vec3<f32>(x, g.y * ce - z * se, g.y * se + z * ce);
}

fn render_laue(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * mp(MP_SPEED);
    var az = t * 0.21;
    var el = 0.24 + 0.16 * sin(t * 0.13);
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.2;
    }

    // A broad polychromatic band samples many Ewald radii simultaneously.
    let energy = 4.0 + 12.0 * mp(MP_KSCALE);
    let bandwidth = 0.025 + 0.38 * mp(MP_FIELD_MIX);
    let mosaicity = 0.0025 + 0.024 * mp(MP_ISO_LEVEL);
    let focal = 1.25 / max(mp(MP_ZOOM), 0.05);
    let spot_gain = mp(MP_W_LATTICE);
    let streak_gain = mp(MP_W_MOTIF);
    let exposure = 0.20 + mp(MP_W_BAND);
    let beam = vec3<f32>(0.0, 0.0, energy);

    var spots = vec3<f32>(0.0);
    var streaks = vec3<f32>(0.0);
    var beam_sum = 0.0;
    let ng_laue = i32(u.num_g);
    for (var i = 0i; i < ng_laue; i++) {
        let ga = g_block.gamp[i];
        let g = laue_rotate(ga.xyz, az, el);
        let kout = beam + g;
        if kout.z <= 0.05 { continue; }

        // |k_out| = |k_in| is the monochromatic Bragg condition. A Laue image
        // accepts a band of incident energies, hence the finite shell width.
        let mismatch = abs(length(kout) - energy);
        let shell = exp(-mismatch * mismatch / max(bandwidth * bandwidth, 1e-6));
        let detector = kout.xy / kout.z * focal;
        let delta = uv - detector;
        let r2 = dot(delta, delta);
        let width = mosaicity * (0.75 + 0.45 * length(g));
        let point = exp(-r2 / max(width * width, 1e-7));

        let detector_len = max(length(detector), 1e-4);
        let radial = detector / detector_len;
        let along = dot(delta, radial);
        let across = dot(delta, vec2<f32>(-radial.y, radial.x));
        let tail = exp(-(
            along * along / max((width * 8.0) * (width * 8.0), 1e-7) +
            across * across / max((width * 1.35) * (width * 1.35), 1e-7)
        ));

        let angle = fract(0.17 * f32(i) + 0.11 * g.z + mp(MP_COLOR_SHIFT));
        let cyan = vec3<f32>(0.12, 0.72, 1.00);
        let amber = vec3<f32>(1.00, 0.34, 0.07);
        let colour = mix(cyan, amber, angle);
        let intensity = ga.w * shell * exposure;
        spots += colour * intensity * point * spot_gain * 3.2;
        streaks += mix(colour, u.crystal_color.xyz, 0.42) * intensity * tail * streak_gain * 0.11;
        beam_sum += intensity * point;
    }

    let r = length(uv);
    let bg = vec3<f32>(0.0015, 0.002, 0.008) + u.crystal_color.xyz * 0.008;
    var col = bg + streaks + spots;
    col += exp(-r * r * 1700.0) * vec3<f32>(1.0, 0.95, 0.80) * (0.18 + beam_sum * 0.05);
    let rim = exp(-pow((r - 0.93) * 30.0, 2.0));
    col += u.crystal_color.xyz * rim * 0.05;
    col *= smoothstep(0.022, 0.046, r);
    return col;
}
