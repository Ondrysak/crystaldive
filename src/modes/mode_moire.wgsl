// ── Mode 14: MOIRE — twisted-bilayer interference of two crystal fields ──
fn render_moire(uv: vec2<f32>) -> vec3<f32> {
    let t      = u.time * u.speed * 0.1;
    let theta  = u.field_mix * 0.524;          // 0..30°
    let ca     = cos(theta);
    let sa     = sin(theta);

    let s      = 2.5 / u.zoom;
    let p1     = vec3<f32>(uv * s, t);
    let uv_r   = vec2<f32>(uv.x*ca - uv.y*sa, uv.x*sa + uv.y*ca);
    let p2     = vec3<f32>(uv_r * s, t);

    let f1     = crystal_field(p1);
    let f2_rot = crystal_field(p2);

    let m_prod = f1 * f2_rot;
    let m_sum  = f1 + f2_rot;
    let m_diff = f1 - f2_rot;

    // Sharpness of contour lines from iso_level (smaller = sharper)
    let line_w = mix(0.05, 0.008, clamp(u.iso_level, 0.0, 1.0));

    // Hue from sum, drifting slowly
    let hue   = fract(m_sum * 0.45 + u.color_shift + u.time*u.speed*0.03);
    var pal   = 0.5 + 0.5*cos(TAU*(hue + vec3<f32>(0.0, 0.333, 0.667)));

    // Dark background, slight crystal-color tint
    var col   = mix(vec3<f32>(0.012, 0.006, 0.022),
                    u.crystal_color.xyz * 0.07,
                    0.5);

    // Soft fill: brightness from |product| (moiré peaks)
    let peak  = clamp(abs(m_prod) * 1.6, 0.0, 1.5);
    col = mix(col,
              pal * mix(vec3<f32>(1.0), u.crystal_color.xyz, 0.45),
              smoothstep(0.0, 0.9, peak));

    // Contour lines where f1 + f2_rot ≈ 0 (zero-set of the sum)
    col += smoothstep(line_w, 0.0, abs(m_sum)) * vec3<f32>(1.0, 0.92, 0.55) * 1.6;

    // Bright lines near product extrema — shifts of |m_prod| against iso_level
    let prod_iso = u.iso_level * 0.45 + 0.05;
    col += smoothstep(line_w*1.2, 0.0, abs(abs(m_prod) - prod_iso))
         * mix(vec3<f32>(1.0, 1.0, 0.85), u.crystal_color.xyz, 0.55) * 1.4;

    // Pinwheel ripple at the moiré-cell scale from the difference
    let pin    = 0.5 + 0.5*sin(m_diff * TAU);
    col += pow(pin, 6.0) * u.crystal_color.xyz * 0.55;

    // Favour the crystal accent in the brightest peaks
    let hot    = smoothstep(0.6, 1.4, peak);
    col       += hot * u.crystal_color.xyz * 0.35;

    return col;
}
