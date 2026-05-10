// ── Mode 19: KIKUCHI — EBSD-style Kikuchi band pattern from reciprocal lattice ──
fn render_kikuchi(uv: vec2<f32>) -> vec3<f32> {
    let t  = u.time * u.speed * 0.06;
    var az = t;
    var el = 0.15;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.0;
    }
    let ca = cos(az); let sa = sin(az);
    let ce = cos(el); let se = sin(el);

    let band_w = 0.005 + 0.012 * (1.0 - u.iso_level);
    let inv_bw2 = 1.0 / max(band_w * band_w, 1e-8);

    var bands  = 0.0;
    var excess = 0.0;

    for (var i = 0; i < 128; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga  = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let amp = ga.w;
        let G   = ga.xyz * u.kscale;

        let gx  = G.x*ca - G.z*sa;
        let gyr = G.x*sa + G.z*ca;
        let gy  = G.y*ce - gyr*se;
        let gz  = G.y*se + gyr*ce;

        let g2d   = vec2<f32>(gx, gy);
        let g_len = length(g2d);
        if g_len < 0.05 { continue; }

        let g_hat = g2d / g_len;
        let n     = vec2<f32>(-g_hat.y, g_hat.x);

        let d_par = dot(uv, g_hat);
        let d_perp = dot(uv, n);

        let G_len = length(G);
        let D     = G_len * 0.18 / max(u.zoom, 1e-4);

        let dp = d_par - D;
        let dn = d_par + D;
        let b_pos = exp(-(dp*dp) * inv_bw2);
        let b_neg = exp(-(dn*dn) * inv_bw2);

        let ew = exp(-gz*gz * 8.0);

        bands += amp * ew * (b_pos + b_neg);

        // Faint excess line: thin streak along band-parallel direction
        // through the origin (perpendicular distance from origin = d_par).
        let exc_w = band_w * 1.6;
        let exc   = exp(-(d_par*d_par) / max(exc_w*exc_w, 1e-8));
        excess   += amp * ew * exc * 0.05;

        // Suppress contributions far along the band (n direction) to keep
        // the network tidy near the screen centre.
        let _unused = d_perp;
    }

    // Background: near-black with subtle crystal_color tint where bands are bright.
    let tint = u.crystal_color.xyz;
    var bg   = vec3<f32>(0.006, 0.004, 0.014);
    bg      += tint * 0.04 * smoothstep(0.0, 1.5, bands);

    // Band color: warm white shifted toward crystal_color.
    let warm_white = vec3<f32>(0.95, 0.92, 0.78);
    let band_col   = mix(warm_white, tint, 0.35);

    var col = bg + band_col * bands + band_col * excess;

    // Soft central beam-stop dark disk.
    let r = length(uv);
    col *= smoothstep(0.020, 0.040, r);

    // Tiny central glint (suggests detector centre).
    col += exp(-r*r * 600.0) * vec3<f32>(1.0, 0.92, 0.70) * 0.18;

    // Circular detector boundary fade (vignette is disabled for this mode).
    let edge = 1.0 - smoothstep(0.85, 0.98, r * u.zoom);
    col *= edge;

    return col;
}
