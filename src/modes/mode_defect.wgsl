// ── Mode 20: DEFECT — interactive vacancy probe with Friedel oscillations ──
fn render_defect(uv: vec2<f32>) -> vec3<f32> {
    // Defect position: follows mouse when pressed, otherwise slow precession.
    var defect_pos: vec2<f32>;
    if u.mouse_down >= 0.5 {
        defect_pos = vec2<f32>((u.mouse.x * 2.0 - 1.0) * u.aspect,
                               1.0 - u.mouse.y * 2.0);
    } else {
        defect_pos = vec2<f32>(cos(u.time * u.speed * 0.3),
                               sin(u.time * u.speed * 0.3)) * 0.4;
    }

    // Base crystal field sampled in the same fashion as render_bz / render_warp.
    let p   = vec3<f32>(uv * 2.5 / u.zoom, u.time * u.speed * 0.1);
    let f0  = crystal_field(p);
    let f02 = cf2(p);

    // Defect contribution — Gaussian "scoop" subtracted at the cursor.
    let r           = length(uv - defect_pos);
    let defect_sigma = 0.10 + (1.0 - u.iso_level) * 0.20;
    let defect_A    = 0.5 + u.iso_level * 1.5;
    let defect_scoop = defect_A * exp(-r * r / (defect_sigma * defect_sigma));

    // Friedel oscillations — 1/r-enveloped ringing tied to k_F.
    let defect_kF       = u.kscale * (3.0 + u.field_mix * 6.0);
    let defect_envelope = exp(-r * 1.5);
    let defect_cos      = cos(2.0 * defect_kF * r);
    let defect_friedel  = (defect_cos / max(r, 0.05)) * defect_envelope * 0.25;

    // Modified field: lattice depressed at defect, with ringing skirt.
    let f = f0 - defect_scoop + defect_friedel;

    // ── Coloring (echoes render_bz) ──────────────────────────────────────
    var c = mix(vec3<f32>(0.02, 0.01, 0.04),
                u.crystal_color.xyz * 0.6,
                smoothstep(-0.8, 0.8, f));

    // Bright zero-contour, warm white.
    c += smoothstep(0.025, 0.0, abs(f)) * vec3<f32>(1.0, 0.95, 0.7) * 1.4;

    // Iso-level contour tinted by crystal color.
    c += smoothstep(0.03, 0.0, abs(f - u.iso_level * 0.5))
       * u.crystal_color.xyz * 1.5;

    // Subtle interference shimmer from f / f02 (keeps it in the render_bz family).
    c += (0.5 + 0.5 * sin(f * TAU * 3.0) * cos(f02 * TAU * 2.0))
       * 0.07 * u.crystal_color.xyz;

    // Defect core glow — sharp cyan/white blob right at the probe position.
    let defect_core = exp(-r * r * 60.0) * 1.6;
    c += defect_core * vec3<f32>(0.75, 0.95, 1.10);

    // Friedel ring highlights — bright crests of the oscillation, away from core.
    let defect_ring_mask = smoothstep(0.85, 1.0, defect_cos);
    let defect_ring_gate = smoothstep(0.15, 0.20, r);
    c += defect_ring_mask * defect_ring_gate * defect_envelope
       * u.crystal_color.xyz * 1.1;

    // Outer faint envelope — soft red "you are probing here" hint.
    c += defect_scoop * 0.05 * vec3<f32>(1.0, 0.25, 0.20);

    return c;
}
