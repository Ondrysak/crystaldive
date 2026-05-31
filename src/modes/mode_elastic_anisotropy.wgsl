// Mode 47: ELASTIC ANISO — directional Young's modulus polar surface.
// Crystal-derived effective stiffness C(n̂) = Σᵢ aᵢ² (n̂·Ĝᵢ)⁴ plotted as a
// 3D ray-cast radial surface; shape encodes mechanical anisotropy.

fn eaniso_rot(v: vec3<f32>, az: f32, el: f32) -> vec3<f32> {
    let ca = cos(az); let sa = sin(az);
    let ce = cos(el); let se = sin(el);
    let ry = vec3<f32>(ca*v.x - sa*v.z, v.y, sa*v.x + ca*v.z);
    return vec3<f32>(ry.x, ce*ry.y - se*ry.z, se*ry.y + ce*ry.z);
}

// Effective stiffness in direction n̂ (normalised).
// Uses (n̂·Ĝᵢ)⁴ — captures 4th-rank elastic tensor projection.
fn eaniso_stiffness(n: vec3<f32>) -> f32 {
    let ng = i32(u.num_g);
    var s = 0.0;
    var norm = 0.0;
    for (var i = 0; i < ng; i++) {
        let ga = g_block.gamp[i];
        let glen = length(ga.xyz);
        if glen < 1e-6 { continue; }
        let ghat = ga.xyz / glen;
        let proj = dot(n, ghat);
        let w = ga.w * ga.w;
        s += w * proj * proj * proj * proj;
        norm += w;
    }
    if norm < 1e-9 { return 1.0; }
    return s / norm;
}

// Ray–radial-surface intersection: find t such that ray(t)/|ray(t)| gives a
// direction n̂, then r_surface = stiffness(n̂).  We sphere-march: shrink step
// when r_ray > r_surf to locate the crossing.
fn eaniso_march(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    let scale = 0.55 + 0.55 * u.field_mix;
    // Stiffness(n̂) ≤ 1, so the whole surface fits in a sphere of radius `scale`.
    // Analytically clip the ray to that bounding sphere and only march the
    // segment inside — background rays (screen corners) cost ~nothing.
    let b = dot(ro, rd);
    let c = dot(ro, ro) - scale * scale;
    let disc = b * b - c;
    if disc < 0.0 { return vec4<f32>(0.0); }
    let sq = sqrt(disc);
    let t_near = max(-b - sq, 0.001);
    let t_far = -b + sq;
    if t_far <= t_near { return vec4<f32>(0.0); }

    let steps = 48;
    let dt = (t_far - t_near) / f32(steps);
    var t = t_near;
    var hit = false;
    var prev_inside = false;
    var t_hit = t_near;
    for (var i = 0; i < steps; i++) {
        let p = ro + rd * t;
        let r = length(p);
        if r < 1e-4 { t += dt; continue; }
        let inside = r < eaniso_stiffness(p / r) * scale;
        if i > 0 && inside != prev_inside {
            t_hit = t;
            hit = true;
            break;
        }
        prev_inside = inside;
        t += dt;
    }
    if !hit { return vec4<f32>(0.0); }
    let p = ro + rd * t_hit;
    let r = length(p);
    let n = p / r;
    let stiff_c = eaniso_stiffness(n);
    // Cheap forward-difference normal of f(p) = |p| − r_surf(p/|p|): 3 extra
    // evals reusing the centre sample instead of a 6-call central difference.
    let eps = 0.012;
    let gx = eaniso_stiffness(normalize(p + vec3<f32>(eps, 0.0, 0.0))) - stiff_c;
    let gy = eaniso_stiffness(normalize(p + vec3<f32>(0.0, eps, 0.0))) - stiff_c;
    let gz = eaniso_stiffness(normalize(p + vec3<f32>(0.0, 0.0, eps))) - stiff_c;
    let grad_surf = -vec3<f32>(gx, gy, gz) * scale;
    let surf_norm = normalize(n + grad_surf * 0.5);
    return vec4<f32>(surf_norm, stiff_c * scale);
}

fn render_elastic_anisotropy(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed;

    var az = t * 0.22;
    var el = 0.30;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.5;
    }

    // Camera
    let cam_dist = 2.8 / max(u.zoom, 0.1);
    let cam_raw = vec3<f32>(0.0, 0.0, cam_dist);
    let cam = eaniso_rot(cam_raw, az, el);
    let look_at = vec3<f32>(0.0);
    let fwd = normalize(look_at - cam);
    let right = normalize(cross(fwd, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, fwd);
    let rd = normalize(fwd + uv.x * right * 0.5 + uv.y * up * 0.5);

    let res = eaniso_march(cam, rd);

    if res.w < 1e-6 {
        // Background: faint reference sphere wireframe
        let r_bg = length(cam + rd * 1.8);
        let bg_ring = exp(-20.0 * abs(r_bg - 0.55));
        return vec3<f32>(0.04, 0.06, 0.12) + bg_ring * vec3<f32>(0.08, 0.12, 0.22);
    }

    let norm = res.xyz;
    let stiff = res.w;

    // Light
    let light = normalize(vec3<f32>(0.8, 1.2, 0.6));
    let diff = max(dot(norm, light), 0.0);
    let spec = pow(max(dot(reflect(-light, norm), -rd), 0.0), 24.0);

    // Stiffness → hue: soft = cool blue, stiff = warm amber
    let iso_val = u.iso_level;
    // isotropic reference at C_iso ≈ 1/8 (uniform distribution of (n̂·Ĝ)⁴)
    let c_ref = 0.125;
    let aniso_ratio = clamp(stiff / (c_ref * (0.55 + 0.55 * u.field_mix) + 1e-6), 0.0, 3.0);

    let base_warm = u.crystal_color.xyz;
    let col_soft = vec3<f32>(0.18, 0.42, 0.90);
    let col_stiff = mix(base_warm, vec3<f32>(1.0, 0.72, 0.18), 0.5);
    let surface_col = mix(col_soft, col_stiff, clamp(aniso_ratio - 0.5, 0.0, 1.0));

    var col = surface_col * (0.15 + 0.85 * diff) + vec3<f32>(1.0) * spec * 0.6;

    // Zener-ratio ring: highlight where aniso_ratio ≈ 1 (isotropic locus)
    let ring = exp(-30.0 * (aniso_ratio - 1.0) * (aniso_ratio - 1.0));
    col += vec3<f32>(0.6, 1.0, 0.6) * ring * iso_val * 0.8;

    return col;
}
