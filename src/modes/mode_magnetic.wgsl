// ── Mode 17: MAGNETIC — 2D spin-texture / skyrmion lattice with oriented streaks ──

fn magnetic_spin(p: vec3<f32>) -> vec3<f32> {
    let e  = 0.05;
    let nx = crystal_field(p);
    let ny = cf2(p);
    let nz = (crystal_field(p + vec3<f32>(0.0, 0.0, e))
            - crystal_field(p - vec3<f32>(0.0, 0.0, e))) / (2.0 * e);
    let v  = vec3<f32>(nx, ny, nz);
    let l  = max(length(v), 1e-5);
    return v / l;
}

fn render_magnetic(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed * 0.1;
    let p = vec3<f32>(uv * 2.5 / u.zoom, t);

    // Raw spin components (use undiff field for nx/ny so we keep the gradient field for the streak modulation).
    let nx_raw = crystal_field(p);
    let ny_raw = cf2(p);
    let e      = 0.05;
    let nz_raw = (crystal_field(p + vec3<f32>(0.0, 0.0, e))
                - crystal_field(p - vec3<f32>(0.0, 0.0, e))) / (2.0 * e);
    let raw_len = max(sqrt(nx_raw*nx_raw + ny_raw*ny_raw + nz_raw*nz_raw), 1e-5);
    let nx = nx_raw / raw_len;
    let ny = ny_raw / raw_len;
    let nz = nz_raw / raw_len;

    // Hedgehog vs Bloch rotation in the (nx, ny) plane.
    let phi_rot = atan2(u.w_motif, u.w_lattice + 1e-3);
    let cph = cos(phi_rot);
    let sph = sin(phi_rot);
    let nx_r = nx * cph - ny * sph;
    let ny_r = nx * sph + ny * cph;

    // Polar dome color: red ↑ to blue ↓.
    var pol = mix(vec3<f32>(0.95, 0.30, 0.20),
                  vec3<f32>(0.20, 0.55, 1.00),
                  0.5 + 0.5 * nz);
    pol = mix(pol, u.crystal_color.xyz, 0.25);

    // Brightness modulator from in-plane gradient magnitude (skyrmion edges shimmer).
    let grad_mag = sqrt(nx_raw * nx_raw + ny_raw * ny_raw);
    let edge     = smoothstep(0.0, 1.2, grad_mag);
    var col      = pol * (0.55 + 0.55 * edge);

    // Streaks aligned with in-plane direction.
    let in_plane_len = sqrt(nx_r * nx_r + ny_r * ny_r);
    let theta_p = atan2(ny_r, nx_r);
    let s       = uv.x * cos(theta_p) + uv.y * sin(theta_p);
    let streak  = 0.5 + 0.5 * sin(s * (40.0 + 60.0 * u.iso_level));
    let dash    = smoothstep(0.55, 0.75, streak);

    // Brightness signal of the dome to flip streak contrast (dark on light, light on dark).
    let dome_lum = dot(col, vec3<f32>(0.30, 0.59, 0.11));
    let streak_color = mix(vec3<f32>(1.0, 0.97, 0.85),
                           vec3<f32>(0.02, 0.02, 0.04),
                           smoothstep(0.25, 0.85, dome_lum));
    col = mix(col, streak_color, dash * in_plane_len * 0.55);

    // Skyrmion-core glow: bright cyan/white halo on downward cores, faint spot on upward.
    let core_dn = smoothstep(0.85, 1.00, -nz);
    col += core_dn * vec3<f32>(0.55, 0.95, 1.10) * 1.6;
    let core_up = smoothstep(0.85, 1.00, nz);
    col += core_up * vec3<f32>(1.10, 0.95, 0.80) * 0.35;

    // Dark background tint biased by crystal accent color.
    let bg = mix(vec3<f32>(0.01, 0.005, 0.02),
                 u.crystal_color.xyz * 0.06,
                 0.5);
    col = mix(bg, col, smoothstep(0.0, 0.25, in_plane_len + 0.15 * abs(nz)));

    return col;
}
