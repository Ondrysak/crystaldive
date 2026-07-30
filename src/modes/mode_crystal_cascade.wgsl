// ── CRYSTAL CASCADE — recursive reciprocal-field fractal ─────────────────
//
// The strongest loaded reciprocal vectors are sampled at successively inflated
// coordinates.  Their quadrature field becomes both an orbit trap and the warp
// for the next level, so lattice symmetry, systematic absences, strain, and
// crystal morphs alter the whole hierarchy instead of merely tinting it.

fn crystal_cascade_rot(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn crystal_cascade_palette(h: f32) -> vec3<f32> {
    let wave = 0.52 + 0.48 * cos(TAU * (h + vec3<f32>(0.00, 0.33, 0.67)));
    return wave * wave;
}

fn render_crystal_cascade(uv: vec2<f32>) -> vec3<f32> {
    if u.num_g == 0u {
        return vec3<f32>(0.01, 0.005, 0.025);
    }

    let inflation = clamp(mp(0u), 1.15, 2.45);
    let speed = mp(1u);
    let warp = clamp(mp(2u), 0.0, 1.0);
    let sharpness = clamp(mp(3u), 3.0, 24.0);
    let hue_shift = mp(4u);
    let zoom = max(mp(5u), 0.2);
    let g_limit = clamp(i32(round(mp(6u))), 4, 16);
    let phase_mix = clamp(mp(7u), 0.0, 2.0);
    let depth = clamp(i32(round(mp(8u))), 3, 7);
    let ng = min(i32(u.num_g), g_limit);
    let t = u.time * speed;

    var p = uv * (2.15 / zoom);
    if u.mouse_down >= 0.5 {
        let pointer = (u.mouse * 2.0 - 1.0) * vec2<f32>(u.aspect, 1.0);
        p -= pointer * 0.72;
    }

    var col = vec3<f32>(0.006, 0.003, 0.018) + u.crystal_color.xyz * 0.018;
    var orbit_glow = 0.0;
    var phase_memory = 0.0;

    for (var level = 0; level < 7; level++) {
        if level >= depth { break; }
        let fl = f32(level);

        var re = 0.0;
        var im = 0.0;
        var norm = 0.0;
        for (var j = 0; j < 16; j++) {
            if j >= ng { break; }
            // A coprime stride prevents every scale from reading the same star
            // in the same order while keeping interpolation through g_block smooth.
            let idx = (j * 5 + level * 3) % ng;
            let ga = g_block.gamp[idx];
            let ph = g_block.phases[idx].x;
            let fj = f32(j);
            let z = 0.34 * sin(0.37 * fl + t * 0.08) + 0.08 * phase_memory;
            let theta = dot(ga.xyz, vec3<f32>(p, z)) * (0.72 + 0.09 * fl)
                        + ph * (0.35 + phase_mix)
                        + t * (0.075 + 0.006 * fj)
                        + fl * phase_mix * 0.41;
            re += ga.w * cos(theta);
            im += ga.w * sin(theta + ga.z * 0.31 + fj * 0.037 * phase_mix);
            norm += ga.w;
        }

        let field = vec2<f32>(re, im) / max(norm, 0.001);
        let mag = length(field);
        let phase = atan2(field.y, field.x) / TAU;
        let level_frac = fl / max(f32(depth - 1), 1.0);

        // Repeated zeros are the dark/bright fractal skeleton; a moving shell
        // around them keeps the image alive without destroying its structure.
        let core = exp(-sharpness * mag);
        let shell_radius = 0.13 + 0.09 * sin(t * 0.11 - fl * 0.83 + phase_memory);
        let shell = exp(-sharpness * 1.45 * abs(mag - shell_radius));
        let signed_ridge = pow(clamp(0.5 + 0.5 * field.x, 0.0, 1.0),
                               mix(2.0, 7.0, warp));
        let energy = core * 1.55 + shell * 0.82 + signed_ridge * 0.22;

        let hue = fract(hue_shift + phase * (0.75 + 0.45 * phase_mix)
                        + level_frac * 0.31 + phase_memory * 0.045);
        let pal = crystal_cascade_palette(hue);
        let crystal_pal = mix(pal, u.crystal_color.xyz, 0.10 + 0.12 * core);
        let level_weight = 0.36 / (1.0 + 0.28 * fl);
        col += crystal_pal * energy * level_weight;

        orbit_glow = orbit_glow * 0.63 + energy;
        phase_memory += phase + field.x * 0.23 - field.y * 0.17;

        // Inverse-fold style recursion: each crystal vector star supplies the
        // rotation, while the complex field bends the next inflated domain.
        let orient_idx = (level * 7 + 1) % ng;
        let orient_g = g_block.gamp[orient_idx].xy;
        let orient = atan2(orient_g.y, orient_g.x);
        let spin = orient * (0.12 + 0.10 * phase_mix) + 0.21 * fl + t * 0.012;
        let folded = abs(p + vec2<f32>(0.17 * field.y, -0.17 * field.x) * warp);
        p = crystal_cascade_rot(folded * inflation - vec2<f32>(0.73, 0.61), spin);
        p += vec2<f32>(field.y, -field.x) * warp * (0.22 + 0.035 * fl);
    }

    let centre = exp(-dot(uv, uv) * 2.4 / max(zoom, 0.2));
    let final_hue = fract(hue_shift + phase_memory * 0.035 + t * 0.008);
    col += crystal_cascade_palette(final_hue) * orbit_glow * 0.022;
    col += u.crystal_color.xyz * centre * 0.07;
    return col;
}
