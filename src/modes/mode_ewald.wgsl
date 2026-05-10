// ── Mode 15: EWALD — Ewald sphere sweeping through 3D reciprocal lattice ──
// Educational hybrid of XRD + RECIP3D: a translucent Ewald sphere of radius
// |k_i| = 1/λ is centered at -k_i so it passes through the origin. Whenever a
// G point lies on the shell, the Laue/Bragg condition |k_i + G| = |k_i| is met
// and the point flares — visualizing the diffraction condition geometrically.

fn ewald_sphere_hit(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32) -> vec2<f32> {
    // Returns (t_near, t_far). If miss, t_far < t_near.
    let oc = ro - c;
    let b  = dot(oc, rd);
    let h  = b*b - dot(oc, oc) + r*r;
    if h < 0.0 { return vec2<f32>(1.0, -1.0); }
    let s = sqrt(h);
    return vec2<f32>(-b - s, -b + s);
}

fn ewald_point_hit(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
    // Nearest forward intersection with sphere centered at c, radius r. -1 if miss.
    let oc = c - ro;
    let b  = dot(oc, rd);
    let h  = b*b - dot(oc, oc) + r*r;
    if h < 0.0 { return -1.0; }
    let th = b - sqrt(h);
    if th <= 0.0 { return -1.0; }
    return th;
}

fn render_ewald(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed;

    // ── Incidence direction (mouse-controlled, otherwise slow precession) ──
    var az_inc: f32 = u.time * u.speed * 0.1;
    var el_inc: f32 = 0.2;
    if u.mouse_down >= 0.5 {
        az_inc = u.mouse.x * TAU;
        el_inc = (u.mouse.y - 0.5) * 2.5;
    }
    let k0 = 0.4 + u.iso_level * 2.5;
    let k_dir = vec3<f32>(sin(az_inc)*cos(el_inc), sin(el_inc), cos(az_inc)*cos(el_inc));
    let k_i  = k0 * k_dir;
    let c_ewald = -k_i;
    let R_ewald = k0;

    // ── Camera (independent slow orbit; mouse drives the sphere, not the cam) ──
    let cam_az = u.time * u.speed * 0.07;
    let cam_el = 0.35;
    let cp  = vec3<f32>(sin(cam_az)*cos(cam_el), sin(cam_el), cos(cam_az)*cos(cam_el)) * (5.5 / u.zoom);
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x*rgt + uv.y*up);

    let G_scale = u.kscale * 0.42;
    let pt_r    = 0.04;

    var best_t   = 1e9;
    var best_col = vec3<f32>(0.0);
    var got_hit  = false;

    // ── Bright origin marker (white) ──
    {
        let r_o = 0.055;
        let oc  = -cp;
        let b   = dot(oc, rd);
        let h   = b*b - dot(oc, oc) + r_o*r_o;
        if h >= 0.0 {
            let th = b - sqrt(h);
            if th > 0.0 {
                best_t   = th;
                got_hit  = true;
                let nm   = normalize(cp + rd*th);
                let diff = max(dot(nm, normalize(vec3<f32>(1.0, 2.0, 1.5))), 0.0);
                best_col = vec3<f32>(1.0, 0.97, 0.85) * (0.45 + 0.65*diff);
            }
        }
    }

    // ── G-point spheres, tinted by Ewald-shell proximity ──
    for (var i = 0; i < 128; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga  = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let amp = ga.w;
        let G   = ga.xyz * G_scale;

        let r_g = pt_r * (0.6 + amp * 0.9);
        let th  = ewald_point_hit(cp, rd, G, r_g);
        if th < 0.0 || th >= best_t { continue; }

        // Ewald flash: how close is G to the sphere shell?
        let d_shell = abs(length(G - c_ewald) - R_ewald);
        let flash   = exp(-d_shell * d_shell * 60.0);

        best_t  = th;
        got_hit = true;

        let p   = cp + rd * th;
        let nm  = normalize(p - G);

        let base   = mix(vec3<f32>(0.55, 0.85, 0.95), vec3<f32>(1.0, 1.0, 1.0), flash);
        var surf   = mix(base, u.crystal_color.xyz, 0.30);

        let light = normalize(vec3<f32>(2.0, 3.0, 1.5));
        let diff  = max(dot(nm, light), 0.0);
        let spec  = pow(max(dot(reflect(-light, nm), -rd), 0.0), 28.0);
        let rim   = pow(1.0 - abs(dot(nm, -rd)), 2.2);

        var c_pt  = surf * (0.25 + 0.75*diff)
                  + rim  * vec3<f32>(0.4, 0.85, 1.0) * 0.35
                  + spec * 0.55;
        c_pt    *= (0.2 + 1.5 * flash);
        c_pt    *= (0.45 + 0.7 * amp);
        best_col = c_pt;
    }

    // ── Background ──
    var col = vec3<f32>(0.004, 0.005, 0.012);
    if got_hit { col = best_col; }

    // ── Translucent Ewald sphere shell (Beer–Lambert through volume) ──
    {
        let tt = ewald_sphere_hit(cp, rd, c_ewald, R_ewald);
        let t_enter = max(tt.x, 0.0);
        let t_exit  = min(tt.y, best_t);
        if tt.y >= tt.x && t_exit > t_enter {
            let path = t_exit - t_enter;
            let shell_thickness = path * 0.18;
            let alpha = (1.0 - exp(-shell_thickness)) * 0.15;
            // Rim brighten where ray grazes sphere (thin path = small alpha; boost rim instead)
            let mid_t = 0.5 * (t_enter + t_exit);
            let mid_p = cp + rd * mid_t;
            let nm_s  = normalize(mid_p - c_ewald);
            let rim_s = pow(1.0 - abs(dot(nm_s, -rd)), 2.0);
            let shell_col = mix(vec3<f32>(0.5, 0.9, 1.0), u.crystal_color.xyz, 0.25);
            col = col * (1.0 - alpha) + shell_col * (alpha * 1.4 + rim_s * 0.18);
        }
    }

    // ── Incident-beam line through c_ewald and origin (extends both directions) ──
    {
        // Distance from ray to infinite line through c_ewald with direction k_dir.
        let oa = cp - c_ewald;
        let cross_d = cross(rd, k_dir);
        let cd2 = dot(cross_d, cross_d);
        if cd2 > 1e-8 {
            let line_d = abs(dot(oa, cross_d)) / sqrt(cd2);
            let beam_w = 0.012;
            // Forward parameter along ray (so we don't draw behind camera).
            let proj_t = dot(-oa - k_dir * dot(-oa, k_dir), rd) / max(dot(rd, rd) - dot(rd, k_dir)*dot(rd, k_dir), 1e-6);
            if proj_t > 0.05 && proj_t < best_t {
                let glow = exp(-(line_d*line_d) / (beam_w*beam_w));
                col += glow * vec3<f32>(1.0, 0.95, 0.6) * 1.4;
            }
        }
    }

    // ── Halo glows around each G (screen-space Gaussian via closest-approach) ──
    for (var i = 0; i < 128; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga    = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let amp   = ga.w;
        let G     = ga.xyz * G_scale;
        let oc    = G - cp;
        let proj  = dot(oc, rd);
        if proj < 0.1 { continue; }
        let cl    = cp + rd*proj;
        let dist2 = dot(G - cl, G - cl);

        let d_shell = abs(length(G - c_ewald) - R_ewald);
        let flash   = exp(-d_shell * d_shell * 60.0);

        let gr      = pt_r * (1.0 + amp) * 2.5;
        let halo    = amp*amp * exp(-dist2 / (gr*gr));
        let halo_c  = mix(u.crystal_color.xyz, vec3<f32>(1.0, 1.0, 0.8), flash);
        col += halo * halo_c * (0.12 + 1.4 * flash);
    }

    return col;
}
