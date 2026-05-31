// ── Mode 46: LATTICE WALK — endless first-person flight through the crystal isosurface ──

fn lw_path(fly: f32) -> vec3<f32> {
    // Gentle helix-like drift through the periodic lattice
    return vec3<f32>(
        0.55 * sin(fly * 0.31) + 0.28 * sin(fly * 0.19),
        0.45 * cos(fly * 0.23) + 0.20 * cos(fly * 0.41),
        fly
    );
}

fn render_zoom(uv: vec2<f32>) -> vec3<f32> {
    let t   = u.time * u.speed;
    let fly = t * 0.30;

    // Camera on forward-moving path; forward = numeric derivative of path
    let pos  = lw_path(fly);
    let pos2 = lw_path(fly + 0.05);
    let fwd  = normalize(pos2 - pos);
    let rgt  = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), fwd));
    let up   = cross(fwd, rgt);

    // FPS look: mouse steers gaze direction while the camera still flies forward
    var look = fwd;
    if u.mouse_down >= 0.5 {
        let yaw = (u.mouse.x - 0.5) * 2.8;
        let pit = (u.mouse.y - 0.5) * 1.6;
        look = normalize(fwd + rgt * yaw + up * pit);
    }

    // zoom acts as inverse-FOV (larger zoom = more telephoto = narrower tunnel view)
    let fov = 1.0 / max(u.zoom, 0.1);
    let rd  = normalize(look + uv.x * rgt * fov + uv.y * up * fov);

    var col = vec3<f32>(0.005, 0.008, 0.022);

    // Ray-march the crystal isosurface; start just ahead to avoid self-hits.
    // Larger step factor + min step and a tighter cap keep the worst-case
    // (grazing rays inside the dense field) from pinning the iteration limit.
    var ray_t = 0.04;
    var hit   = false;
    var hit_p = vec3<f32>(0.0);
    var hit_i = 0;
    let lw_steps = 56;
    for (var i = 0; i < lw_steps; i++) {
        let p = pos + rd * ray_t;
        let d = sdf(p);
        if abs(d) < 0.004 { hit = true; hit_p = p; hit_i = i; break; }
        if ray_t > 7.0 { break; }
        ray_t += max(abs(d) * 0.85, 0.010);
    }

    if hit {
        // Analytical normal, oriented toward the camera (inside-out rendering)
        let nm_raw = calc_normal(hit_p);
        let nm     = select(nm_raw, -nm_raw, dot(nm_raw, rd) > 0.0);

        // Colour from the crystal field value at the hit point
        let cf_val = crystal_field(hit_p);
        let hue    = fract(cf_val * 0.45 + u.color_shift + t * 0.022);
        var surf   = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
        surf = mix(surf, u.crystal_color.xyz, 0.25);

        // Head-lamp light moves with camera forward direction
        let light = normalize(fwd + vec3<f32>(0.35, 0.60, 0.15));
        let diff  = max(dot(nm, light), 0.0);
        let rim   = pow(1.0 - abs(dot(nm, -rd)), 3.0);
        let spec  = pow(max(dot(reflect(-light, nm), -rd), 0.0), 22.0);

        col  = surf * (0.20 + 0.80 * diff)
             + vec3<f32>(1.0, 0.90, 0.70) * spec * 0.40
             + u.crystal_color.xyz * rim * 0.55;
        // AO-style darkening for deep marches
        col *= 0.45 + 0.55 * (1.0 - f32(hit_i) / f32(lw_steps));

        // Depth fog into the tunnel
        let fog = 1.0 - exp(-ray_t * 0.25);
        let fgc = u.crystal_color.xyz * 0.08 + vec3<f32>(0.004, 0.008, 0.020);
        col = mix(col, fgc, fog * 0.50);
    } else {
        // No direct hit — volumetric glow from nearby surfaces along the ray
        var glow = 0.0;
        for (var k = 0; k < 6; k++) {
            let tg  = 0.2 + f32(k) * 0.9;
            let p   = pos + rd * tg;
            glow   += exp(-abs(sdf(p)) * 9.0) * 0.20;
        }
        col += u.crystal_color.xyz * glow * 0.85;
        col += vec3<f32>(0.25, 0.55, 1.00) * glow * 0.30;
    }

    return col;
}
