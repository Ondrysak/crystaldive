// ── Mode 38: NEMATIC — liquid-crystal director field n̂≡−n̂ with ±1/2 disclinations ──

fn nematic_director(p: vec3<f32>) -> vec2<f32> {
    let h1 = crystal_field(p);
    let h2 = cf2(vec3<f32>(p.x, p.y, p.z * 1.4 + 0.37));
    return vec2<f32>(h1, h2);
}

fn render_nematic(uv: vec2<f32>) -> vec3<f32> {
    let r = uv * 2.0 / mp(MP_ZOOM);
    let t = u.time * mp(MP_SPEED);

    // Two scalar fields whose phase encodes 2θ (so θ is unique mod π → nematic).
    // Analytical gradient replaces 8 finite-difference crystal_field calls for Jacobian.
    let p1  = vec3<f32>(r, t * 0.05);
    let vg1 = crystal_field_val_grad_xy(p1);
    let h1  = vg1.x;
    // cf2 transforms the input: cf2(x) = crystal_field(x*1.37 + offset)
    // Chain rule: ∇cf2 w.r.t. r = 1.37 * ∇crystal_field at transformed point
    let p2_inner = vec3<f32>(r, t * 0.07) * 1.37 + vec3<f32>(1.618, 2.718, 3.141);
    let vg2 = crystal_field_val_grad_xy(p2_inner);
    let h2  = vg2.x;
    let h1x = vg1.y; let h1y = vg1.z;
    let h2x = vg2.y * 1.37; let h2y = vg2.z * 1.37;

    // Director angle: working with 2θ keeps n̂ ≡ −n̂ symmetry intact.
    let two_theta = atan2(h2, h1);          // ∈ (-π, π]
    let theta     = 0.5 * two_theta;        // ∈ (-π/2, π/2]

    // Alternative director construction (90°-rotated) for field_mix interpolation.
    let theta_alt = 0.5 * atan2(h1, -h2);
    let theta_eff = mix(theta, theta_alt, mp(MP_FIELD_MIX));

    // Order parameter magnitude: zero at disclination cores.
    let m    = sqrt(h1 * h1 + h2 * h2);
    let core = exp(-m * 4.0);

    // Hairy streaks projected along the director (uses theta_eff so streaks
    // line up cleanly across cores and fan into ±1/2 comet/triangle shapes).
    let s      = uv.x * cos(theta_eff) + uv.y * sin(theta_eff);
    let freq   = 50.0 + 60.0 * mp(MP_ISO_LEVEL);
    let streak = 0.5 + 0.5 * sin(s * freq);
    let dash   = smoothstep(0.45, 0.65, streak);

    // Defect sign from analytical Jacobian (was: 8 finite-difference crystal_field calls).
    let defect_sign = h1x * h2y - h1y * h2x;

    // Background — cool, glowing toward defect cores where m → 0.
    let bg = mix(vec3<f32>(0.02, 0.02, 0.05),
                 u.crystal_color.xyz * 0.30,
                 smoothstep(0.0, 1.5, m));

    // Warm streak ink, biased by crystal accent.
    let streak_col = mix(vec3<f32>(1.0, 0.95, 0.80),
                         u.crystal_color.xyz,
                         0.30);

    // Compose: streak overlay fades out where m is tiny (i.e. at cores), so
    // hairs fan rather than swarm into the singularity.
    var col = mix(bg, streak_col, dash * smoothstep(0.0, 0.2, m) * 0.70);

    // Magenta defect-core glow (classical nematic visualization convention).
    col += core * vec3<f32>(1.10, 0.50, 0.95) * 1.4;

    // Quadrupolar tint distinguishing +1/2 (warm) from −1/2 (cool) cores.
    col += core * mix(vec3<f32>(0.6, 0.9, 1.1),
                      vec3<f32>(1.1, 0.7, 0.4),
                      step(0.0, defect_sign)) * 0.5;

    return col;
}
