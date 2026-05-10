// ── Mode 35: VORTEX_KNOT — luminous tube around the nodal line of the complex wavefunction ──

fn vortex_psi(r: vec3<f32>) -> vec2<f32> {
    var re = 0.0;
    var im = 0.0;
    for (var i = 0i; i < 64i; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let ph = textureLoad(g_tex, vec2<i32>(i, 1), 0).r;
        let G  = ga.xyz * u.kscale;
        let amp = ga.w;
        let arg = dot(G, r) + ph + u.time * u.speed * 0.15 * f32(i + 1);
        re += amp * cos(arg);
        im += amp * sin(arg);
    }
    return vec2<f32>(re, im);
}

fn vortex_field(r: vec3<f32>) -> vec2<f32> {
    let r_eff = r - vec3<f32>((u.field_mix - 0.5) * 0.8, 0.0, 0.0);
    return vortex_psi(r_eff);
}

fn vortex_sdf(r: vec3<f32>) -> f32 {
    let R_tube = 0.04 + 0.06 * u.iso_level;
    let psi = vortex_field(r);
    return length(psi) - R_tube;
}

fn vortex_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.004;
    return normalize(vec3<f32>(
        vortex_sdf(p + vec3<f32>(e, 0.0, 0.0)) - vortex_sdf(p - vec3<f32>(e, 0.0, 0.0)),
        vortex_sdf(p + vec3<f32>(0.0, e, 0.0)) - vortex_sdf(p - vec3<f32>(0.0, e, 0.0)),
        vortex_sdf(p + vec3<f32>(0.0, 0.0, e)) - vortex_sdf(p - vec3<f32>(0.0, 0.0, e)),
    ));
}

fn render_vortex_knot(uv: vec2<f32>) -> vec3<f32> {
    let t  = u.time * u.speed;
    var az = t * 0.15;
    var el = 0.4;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.5;
    }
    let cp  = vec3<f32>(sin(az)*cos(el), sin(el), cos(az)*cos(el)) * (4.0 / u.zoom);
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x*rgt + uv.y*up);

    // background: very dark navy
    var col = vec3<f32>(0.005, 0.008, 0.025);

    // sphere-trace the nodal-line tube
    var ray_t  = 0.0;
    var hit    = false;
    var hit_p  = vec3<f32>(0.0);
    var hit_i  = 0;
    for (var i = 0; i < 100; i++) {
        let p = cp + rd * ray_t;
        // bound to cube [-2.5, 2.5]^3 (only after we've entered the domain)
        if (ray_t > 0.5 && (abs(p.x) > 2.5 || abs(p.y) > 2.5 || abs(p.z) > 2.5)) {
            break;
        }
        let d = vortex_sdf(p);
        if d < 0.003 {
            hit   = true;
            hit_p = p;
            hit_i = i;
            break;
        }
        if ray_t > 9.0 { break; }
        // damp the step — vortex_sdf isn't a true Euclidean SDF
        ray_t += max(d, 0.005) * 0.5;
    }

    if hit {
        let nm = vortex_normal(hit_p);

        // sample psi slightly off the line to get a defined phase
        let off     = nm * 0.06;
        let psi_off = vortex_field(hit_p + off);
        let phase   = atan2(psi_off.y, psi_off.x);

        let hue = fract(phase / TAU + u.color_shift + u.time * u.speed * 0.04);
        var surf = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
        surf = mix(surf, u.crystal_color.xyz, 0.20);

        let light = normalize(vec3<f32>(1.5, 2.0, 1.0));
        let diff  = max(dot(nm, light), 0.0);
        let rim   = pow(1.0 - abs(dot(nm, -rd)), 2.5);
        let spec  = pow(max(dot(reflect(-light, nm), -rd), 0.0), 32.0);

        col = surf * (0.30 + 0.75 * diff)
            + vec3<f32>(1.0, 0.85, 0.55) * spec * 0.55              // warm specular
            + vec3<f32>(0.20, 0.85, 0.95) * rim * 0.55;             // teal rim
        col *= 0.45 + 0.55 * (1.0 - f32(hit_i) / 100.0);
    }

    // cheap volumetric halo: sample 6 points along the ray, accumulate where |psi| is small
    var halo = vec3<f32>(0.0);
    let halo_end = select(6.0, ray_t, hit);
    for (var k = 0; k < 6; k++) {
        let s = (f32(k) + 0.5) / 6.0;
        let p = cp + rd * (s * halo_end);
        if (abs(p.x) > 2.5 || abs(p.y) > 2.5 || abs(p.z) > 2.5) { continue; }
        let psi  = vortex_field(p);
        let m2   = dot(psi, psi);
        let glow = exp(-m2 * 30.0);
        let phase = atan2(psi.y, psi.x);
        let hue   = fract(phase / TAU + u.color_shift + u.time * u.speed * 0.04);
        var hc    = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
        hc = mix(hc, u.crystal_color.xyz, 0.35);
        halo += hc * glow * 0.06;
    }
    col += halo;

    return col;
}
