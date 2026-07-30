// Mode: BRAGG DIVE — volumetric flight through crystal-seeded morphing octagram cells.
//
// Unlike LATTICE WALK's opaque crystal-field isosurface, this is an emissive
// volume: each repeated unit cell contains an open gate made from signed box
// fields. The gate continuously changes topology from a connected octagram to
// four separated petals. A travelling phase offsets that morph per cell, so the
// structure opens ahead of the camera instead of switching between fixed scenes.

const BRAGG_DIVE_STEPS: i32 = 60;

fn bragg_dive_rot(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn bragg_dive_rot_xy(p: vec3<f32>, a: f32) -> vec3<f32> {
    let xy = bragg_dive_rot(p.xy, a);
    return vec3<f32>(xy, p.z);
}

fn bragg_dive_rot_xz(p: vec3<f32>, a: f32) -> vec3<f32> {
    let xz = bragg_dive_rot(p.xz, a);
    return vec3<f32>(xz.x, p.y, xz.y);
}

fn bragg_dive_sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn bragg_dive_path(fly: f32, crystal_phase: f32) -> vec3<f32> {
    return vec3<f32>(
        0.24 * sin(fly * 0.31 + crystal_phase),
        0.19 * cos(fly * 0.23 - crystal_phase * 0.7),
        fly
    );
}

// Signed-field blend between two different topologies. The connected state is
// an eight-point gate with four longitudinal rails; the separated state pulls
// four bars away from the axis and leaves a hollow diamond at the centre.
fn bragg_dive_cell(p: vec3<f32>, morph: f32, twist: f32, beam: f32) -> f32 {
    var q = bragg_dive_rot_xy(p, twist + morph * 0.28);
    q = bragg_dive_rot_xz(q, 0.12 * sin(twist * 1.7));

    let width = 0.045 + 0.095 * beam;
    let gate_z = 0.075 + 0.060 * beam;
    let arm = 0.82 - 0.10 * morph;

    let bar0 = bragg_dive_sd_box(q, vec3<f32>(arm, width, gate_z));
    let bar1 = bragg_dive_sd_box(bragg_dive_rot_xy(q, TAU * 0.125), vec3<f32>(arm, width, gate_z));
    let bar2 = bragg_dive_sd_box(bragg_dive_rot_xy(q, TAU * 0.250), vec3<f32>(arm, width, gate_z));
    let bar3 = bragg_dive_sd_box(bragg_dive_rot_xy(q, TAU * 0.375), vec3<f32>(arm, width, gate_z));
    let octagram = min(min(bar0, bar1), min(bar2, bar3));

    let rail_p = vec3<f32>(abs(q.x) - 0.61, abs(q.y) - 0.61, q.z);
    let rails = bragg_dive_sd_box(rail_p, vec3<f32>(width * 0.72, width * 0.72, 1.22));
    let connected = min(octagram, rails);

    let spread = 0.20 + 0.48 * morph;
    let petal_b = vec3<f32>(0.42, width * 1.12, gate_z * 1.12);
    let petal_xp = bragg_dive_sd_box(bragg_dive_rot_xy(q - vec3<f32>(spread, 0.0, 0.0), 0.32), petal_b);
    let petal_xn = bragg_dive_sd_box(bragg_dive_rot_xy(q + vec3<f32>(spread, 0.0, 0.0), 0.32), petal_b);
    let petal_yp = bragg_dive_sd_box(bragg_dive_rot_xy(q - vec3<f32>(0.0, spread, 0.0), TAU * 0.25 - 0.32), petal_b);
    let petal_yn = bragg_dive_sd_box(bragg_dive_rot_xy(q + vec3<f32>(0.0, spread, 0.0), TAU * 0.25 - 0.32), petal_b);
    let petals = min(min(petal_xp, petal_xn), min(petal_yp, petal_yn));

    let diamond_q = bragg_dive_rot_xy(q, TAU * 0.125);
    let outer = bragg_dive_sd_box(diamond_q, vec3<f32>(0.30, 0.30, gate_z * 0.82));
    let inner = bragg_dive_sd_box(diamond_q, vec3<f32>(0.17, 0.17, gate_z * 1.30));
    let diamond = max(outer, -inner);
    let separated = min(min(petals, diamond), rails);

    return mix(connected, separated, smoothstep(0.08, 0.92, morph));
}

fn bragg_dive_palette(h: f32) -> vec3<f32> {
    return 0.52 + 0.48 * cos(TAU * (h + vec3<f32>(0.02, 0.34, 0.67)));
}

fn render_bragg_dive(uv: vec2<f32>) -> vec3<f32> {
    let cell_scale = clamp(mp(0u), 0.35, 3.0);
    let speed = mp(1u);
    let morph_bias = mp(2u);
    let beam = clamp(mp(3u), 0.0, 1.0);
    let hue_shift = mp(4u);
    let fov = 1.0 / max(mp(5u), 0.12);
    let crystal_twist = mp(6u);
    let glow = 0.45 + 1.25 * mp(7u);
    let view_depth = 6.0 + 7.0 * mp(8u);

    // The first reciprocal vector and phase orient every gate. Different loaded
    // crystals therefore retain the same visual grammar but not the same tunnel.
    var crystal_angle = 0.0;
    var crystal_elev = 0.0;
    var crystal_phase = 0.0;
    if u.num_g > 0u {
        let g = g_block.gamp[0].xyz;
        crystal_angle = atan2(g.y, g.x);
        crystal_elev = atan2(g.z, max(length(g.xy), 1e-4));
        crystal_phase = g_block.phases[0].x;
    }

    let anim = u.time * speed;
    let fly = anim * 1.15;
    let ro = bragg_dive_path(fly, crystal_phase);
    let ahead = bragg_dive_path(fly + 0.06, crystal_phase);
    let forward = normalize(ahead - ro);
    var right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), forward));
    var up = cross(forward, right);

    let roll = 0.18 * sin(anim * 0.17) + crystal_angle * 0.10 * crystal_twist;
    let rolled_right = right * cos(roll) + up * sin(roll);
    up = -right * sin(roll) + up * cos(roll);
    right = rolled_right;

    var look = forward;
    if u.mouse_down >= 0.5 {
        let yaw = (u.mouse.x - 0.5) * 2.4;
        let pitch = (u.mouse.y - 0.5) * 1.5;
        look = normalize(forward + right * yaw + up * pitch);
    }
    let rd = normalize(look + uv.x * right * fov + uv.y * up * fov);

    let period = vec3<f32>(2.55, 2.55, 2.35);
    var ray_t = 0.05;
    var col = vec3<f32>(0.002, 0.004, 0.014);
    var total = 0.0;

    for (var i = 0; i < BRAGG_DIVE_STEPS; i = i + 1) {
        if ray_t > view_depth { break; }
        let world = ro + rd * ray_t;
        let scaled = world * cell_scale;
        let cell_id = floor(scaled / period + vec3<f32>(0.5));
        let local = scaled - cell_id * period;
        let wave = anim * 0.82 - cell_id.z * 0.74
                 + cell_id.x * 0.29 + cell_id.y * 0.41 + crystal_phase;
        let auto_morph = 0.5 + 0.5 * sin(wave);
        let morph = clamp(auto_morph + (morph_bias - 0.5) * 0.85, 0.0, 1.0);
        let twist = crystal_angle * crystal_twist
                  + crystal_elev * 0.35
                  + 0.12 * sin(wave * 0.7);
        let field_d = bragg_dive_cell(local, morph, twist, beam);
        let world_d = field_d / cell_scale;

        let density = exp(-abs(world_d) * (28.0 + 22.0 * beam));
        let fade = exp(-ray_t * 0.075);
        let hue = fract(hue_shift + crystal_phase / TAU
                      + dot(cell_id, vec3<f32>(0.071, 0.113, 0.047))
                      + morph * 0.18 + ray_t * 0.012);
        var sample_col = bragg_dive_palette(hue);
        sample_col = mix(sample_col, u.crystal_color.xyz, 0.30);
        col += sample_col * density * fade * (0.007 + 0.012 * glow);
        col += vec3<f32>(1.0, 0.88, 0.68) * pow(density, 5.0) * fade * 0.007 * glow;
        total += density * fade;

        ray_t += max(abs(world_d) * 0.58, 0.026);
    }

    // Blue distance haze keeps open cells legible when no beam crosses a ray;
    // accumulated density adds a restrained bloom without relying on feedback.
    col += mix(vec3<f32>(0.003, 0.007, 0.022), u.crystal_color.xyz * 0.055,
               clamp(total * 0.010, 0.0, 0.42));
    col += vec3<f32>(0.014, 0.050, 0.13) * (1.0 - exp(-total * 0.030)) * glow;
    return col;
}
