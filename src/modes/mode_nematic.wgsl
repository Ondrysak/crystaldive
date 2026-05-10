// ── Mode 38: NEMATIC — liquid-crystal director field n̂≡−n̂ with ±1/2 disclinations ──

fn nematic_director(p: vec3<f32>) -> vec2<f32> {
    let h1 = crystal_field(p);
    let h2 = cf2(vec3<f32>(p.x, p.y, p.z * 1.4 + 0.37));
    return vec2<f32>(h1, h2);
}

fn render_nematic(uv: vec2<f32>) -> vec3<f32> {
    let r = uv * 2.0 / u.zoom;
    let t = u.time * u.speed;

    // Two scalar fields whose phase encodes 2θ (so θ is unique mod π → nematic).
    let h1 = crystal_field(vec3<f32>(r, t * 0.05));
    let h2 = cf2(vec3<f32>(r, t * 0.07));

    // Director angle: working with 2θ keeps n̂ ≡ −n̂ symmetry intact.
    let two_theta = atan2(h2, h1);          // ∈ (-π, π]
    let theta     = 0.5 * two_theta;        // ∈ (-π/2, π/2]

    // Alternative director construction (90°-rotated) for field_mix interpolation.
    let theta_alt = 0.5 * atan2(h1, -h2);
    let theta_eff = mix(theta, theta_alt, u.field_mix);

    // Order parameter magnitude: zero at disclination cores.
    let m    = sqrt(h1 * h1 + h2 * h2);
    let core = exp(-m * 4.0);

    // Hairy streaks projected along the director (uses theta_eff so streaks
    // line up cleanly across cores and fan into ±1/2 comet/triangle shapes).
    let s      = uv.x * cos(theta_eff) + uv.y * sin(theta_eff);
    let freq   = 50.0 + 60.0 * u.iso_level;
    let streak = 0.5 + 0.5 * sin(s * freq);
    let dash   = smoothstep(0.45, 0.65, streak);

    // Determine ±1/2 defect sign via finite-difference Jacobian:
    // sign of ∂h1/∂x · ∂h2/∂y − ∂h1/∂y · ∂h2/∂x near a zero of (h1, h2).
    let e   = 0.04;
    let h1x = crystal_field(vec3<f32>(r + vec2<f32>(e, 0.0), t * 0.05))
            - crystal_field(vec3<f32>(r - vec2<f32>(e, 0.0), t * 0.05));
    let h1y = crystal_field(vec3<f32>(r + vec2<f32>(0.0, e), t * 0.05))
            - crystal_field(vec3<f32>(r - vec2<f32>(0.0, e), t * 0.05));
    let h2x = cf2(vec3<f32>(r + vec2<f32>(e, 0.0), t * 0.07))
            - cf2(vec3<f32>(r - vec2<f32>(e, 0.0), t * 0.07));
    let h2y = cf2(vec3<f32>(r + vec2<f32>(0.0, e), t * 0.07))
            - cf2(vec3<f32>(r - vec2<f32>(0.0, e), t * 0.07));
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
