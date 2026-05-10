// ── Mode 13: PHONON — animated 5x5x5 atom lattice oscillating along a phonon eigenmode ──

fn phonon_atom_pos(R: vec3<f32>, q: vec3<f32>, e_hat: vec3<f32>, amp: f32, omega: f32, t: f32) -> vec3<f32> {
    let phase = dot(q, R) - omega * t;
    return R + e_hat * amp * cos(phase);
}

fn render_phonon(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time;

    // ── Camera (orbit) ────────────────────────────────────────────────────
    var az = t * u.speed * 0.18;
    var el = 0.35;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.5;
    }
    let cp  = vec3<f32>(sin(az)*cos(el), sin(el), cos(az)*cos(el)) * (5.0 / u.zoom);
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x*rgt + uv.y*up);

    // ── Pick q vector from g_tex ─────────────────────────────────────────
    let q_idx = clamp(i32(u.iso_level * f32(u.num_g)), 0, i32(u.num_g) - 1);
    let q     = textureLoad(g_tex, vec2<i32>(q_idx, 0), 0).xyz * u.kscale;

    // ── Eigenvector ê (mix transverse → longitudinal) ────────────────────
    let q_len = max(length(q), 1e-5);
    let q_hat = q / q_len;
    var t_hat: vec3<f32>;
    if abs(dot(q_hat, vec3<f32>(0.0, 1.0, 0.0))) > 0.95 {
        t_hat = normalize(cross(q_hat, vec3<f32>(1.0, 0.0, 0.0)));
    } else {
        t_hat = normalize(cross(q_hat, vec3<f32>(0.0, 1.0, 0.0)));
    }
    let e_hat = normalize(mix(t_hat, q_hat, u.field_mix));

    // ── Acoustic-like dispersion ─────────────────────────────────────────
    let omega = sqrt(length(q) + 0.1) * u.speed * 1.5;
    let amp_A = 0.18 + 0.12 * u.iso_level;
    let sphere_r = 0.18;

    let light = normalize(vec3<f32>(1.6, 2.2, 1.4));

    // ── Background: dark navy + subtle crystal_field tint ────────────────
    let bg_f = crystal_field(rd * 1.5);
    var col  = vec3<f32>(0.008, 0.005, 0.022)
             + u.crystal_color.xyz * bg_f * 0.04;

    // ── Pass 1: ray-cast spheres at displaced atoms (frontmost wins) ─────
    var best_t   = 1e9;
    var best_col = vec3<f32>(0.0);
    var got_hit  = false;

    for (var ix = -2; ix <= 2; ix++) {
        for (var iy = -2; iy <= 2; iy++) {
            for (var iz = -2; iz <= 2; iz++) {
                let R   = vec3<f32>(f32(ix), f32(iy), f32(iz));
                let phs = dot(q, R) - omega * t;
                let R_t = R + e_hat * amp_A * cos(phs);

                let oc = R_t - cp;
                let b  = dot(oc, rd);
                let h  = b*b - dot(oc, oc) + sphere_r * sphere_r;
                if h < 0.0 { continue; }
                let th = b - sqrt(h);
                if th <= 0.0 || th >= best_t { continue; }

                best_t  = th;
                got_hit = true;

                let p  = cp + rd * th;
                let nm = normalize(p - R_t);

                // Color by phase → hue cycle
                let hue  = fract(phs / TAU + u.color_shift);
                var surf = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
                surf = mix(surf, u.crystal_color.xyz, 0.28);

                let diff = max(dot(nm, light), 0.0);
                let spec = pow(max(dot(reflect(-light, nm), -rd), 0.0), 30.0);
                let rim  = pow(1.0 - abs(dot(nm, -rd)), 2.4);

                best_col = surf * (0.28 + 0.72 * diff)
                         + spec * vec3<f32>(1.0, 0.95, 0.85) * 0.55
                         + rim  * u.crystal_color.xyz * 0.55;
            }
        }
    }

    if got_hit { col = best_col; }

    // ── Pass 2: soft halo glow per atom (depth cue) ──────────────────────
    let halo_r = sphere_r * 2.6;
    for (var ix = -2; ix <= 2; ix++) {
        for (var iy = -2; iy <= 2; iy++) {
            for (var iz = -2; iz <= 2; iz++) {
                let R   = vec3<f32>(f32(ix), f32(iy), f32(iz));
                let phs = dot(q, R) - omega * t;
                let R_t = R + e_hat * amp_A * cos(phs);

                let oc   = R_t - cp;
                let proj = dot(oc, rd);
                if proj < 0.05 { continue; }
                let cl    = cp + rd * proj;
                let dist2 = dot(R_t - cl, R_t - cl);

                let hue  = fract(phs / TAU + u.color_shift);
                var hcol = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
                hcol = mix(hcol, u.crystal_color.xyz, 0.45);

                col += hcol * exp(-dist2 / (halo_r * halo_r)) * 0.07;
            }
        }
    }

    return col;
}
