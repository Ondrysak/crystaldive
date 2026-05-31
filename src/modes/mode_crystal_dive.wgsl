// ── Mode 37: CRYSTAL DIVE — endless zoom into the lattice via point-group symmetry ──
//
// Idea:
//   Real-space translation invariance means crystal_field(p) is periodic on
//   the lattice. Point-group rotations (4-fold for cubic, 6-fold for hex,
//   …) take the lattice into itself. So a spiral camera that ZOOMS by 2× and
//   ROTATES by 2π/n_fold per "octave" lands on a structurally equivalent view
//   every octave — perfect for an infinite-zoom illusion.
//
//   We render TWO crossfading octaves at a time:
//
//       phase = fract(log2(zoom))
//       view_A = sample at zoom 2^floor(log2(zoom))
//       view_B = sample at zoom 2× view_A
//       col    = mix(view_A, view_B, smoothstep(0, 1, phase))
//
//   Because the crystal is periodic AND the lattice point-group is honoured
//   on every octave boundary, view_B at phase=1 matches view_A at the next
//   octave's phase=0 — seamless loop. Same trick the Fraksl feedback path
//   uses (sample previous frame at scaled uv), but folded into one pass.
//
// Physics in:
//   • Reciprocal basis G1, G2 read from g_tex (the loaded crystal's actual lattice).
//   • Symmetry inferred from |cos(G1, G2)|:
//       ≈0    → 4-fold (cubic / tetragonal — square lattice)
//       ≈0.5  → 6-fold (hex / trigonal)
//       other → 2-fold (oblique / monoclinic — half-spiral per octave)
//   • crystal_field() sampled at the rotated/scaled point — the actual electron
//     density Fourier series for this material.
//
// Slider semantics:
//   zoom         → base zoom level (camera distance from sample plane)
//   speed        → zoom rate; speed=0 freezes the dive
//   color_shift  → hue rotation of the final colour
//   field_mix    → blend between crystal_field and the shifted cf2 (motif twin)
//   iso_level    → contrast / saturation lift
//   w_lattice    → strength of the "vignette tunnel" that focuses the centre
//   w_motif      → strength of an outer counter-spiral overlay (extra trippy)
//   w_band       → strength of an inner phase-singularity sparkle at zoom centres

fn crystal_dive_basis() -> mat2x2<f32> {
    // Same two-shortest-in-plane-G heuristic as the ABRIKOSOV mode so the two
    // lattice-driven modes agree on what "this crystal's 2D primitive cell" is.
    var b1 = vec2<f32>(1.0, 0.0);
    var b2 = vec2<f32>(0.0, 1.0);
    var l1 = 1e9;
    var l2 = 1e9;
    let ng_cd = i32(u.num_g);
    for (var i = 0i; i < ng_cd; i++) {
        let g3 = g_block.gamp[i].xyz;
        let gxy = g3.xy;
        let lxy = length(gxy);
        if (lxy < 0.05) { continue; }
        if (abs(g3.z) > lxy * 1.2) { continue; }
        if (lxy < l1) {
            l2 = l1; b2 = b1;
            l1 = lxy; b1 = gxy;
        } else if (lxy < l2) {
            let par = abs(dot(gxy, b1) / (lxy * l1 + 1e-9));
            if (par < 0.92) { l2 = lxy; b2 = gxy; }
        }
    }
    if (l2 > 9e8) {
        // Fallback to triangular — picks up gracefully when no crystal is loaded yet.
        b1 = vec2<f32>(1.0, 0.0);
        b2 = vec2<f32>(0.5, 0.8660254);
    }
    return mat2x2<f32>(b1, b2);
}

fn crystal_dive_n_fold(b1: vec2<f32>, b2: vec2<f32>) -> f32 {
    // |cos θ| between the two shortest reciprocal-lattice vectors picks the
    // point-group rotation order used for the spiral camera.
    let cosg = abs(dot(b1, b2) / max(length(b1) * length(b2), 1e-6));
    // Square-ish (≈0) → 4-fold; hex-ish (≈0.5) → 6-fold; otherwise 2-fold.
    if (cosg < 0.15)              { return 4.0; }
    if (abs(cosg - 0.5) < 0.15)   { return 6.0; }
    return 2.0;
}

// Sample crystal_field at a transformed sample plane and return a colour.
fn crystal_dive_sample(uv: vec2<f32>, scale: f32, rot: f32, t: f32) -> vec3<f32> {
    let cr = cos(rot);
    let sr = sin(rot);
    let R  = mat2x2<f32>(vec2<f32>(cr, sr), vec2<f32>(-sr, cr));
    let p2 = R * uv * scale;
    // Drift the z-coord with t so even a static zoom level still evolves —
    // gives the dive a sense of "we're not just looking at a still frame".
    let p3 = vec3<f32>(p2, t * 0.03);
    let f1 = crystal_field(p3);
    let f2 = cf2(p3);
    return cfield_col(f1, f2, vec3<f32>(0.0, 0.0, 1.0));
}

fn render_crystal_dive(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * mp(MP_SPEED);

    // Crystal-derived spiral geometry.
    let cb     = crystal_dive_basis();
    let b1     = cb[0];
    let b2     = cb[1];
    let n_fold = crystal_dive_n_fold(b1, b2);

    // Slow dive rate — ~0.16 octaves per second at speed=1.
    let zoom_log    = t * 0.16;
    let level       = floor(zoom_log);
    let phase       = zoom_log - level;
    let smooth_ph   = phase * phase * (3.0 - 2.0 * phase);   // smoothstep

    // Rotation: one full point-group rotation per octave, so octave N+1 lands
    // on the same point-group orbit as octave N.
    let rot_per_octave = TAU / n_fold;
    let rot_a = rot_per_octave * level;
    let rot_b = rot_per_octave * (level + 1.0);

    // Sub-octave rotation: blend the two endpoints' rotation by `phase` so the
    // camera *continuously* spirals, not jumps each octave.
    let rot_cont = mix(rot_a, rot_b, smooth_ph);

    // Two crossfading sample scales — scale_a halves over the octave so view_A
    // zooms IN, and the next-octave view_B (already half-scale lower) fades in
    // to take over. The fade is gaussian-ish via smooth_ph.
    let base_scale = 2.4 / max(mp(MP_ZOOM), 0.08);
    let scale_a = base_scale / exp2(level);
    let scale_b = scale_a * 0.5;

    let view_a = crystal_dive_sample(uv, scale_a, rot_cont,                    t);
    let view_b = crystal_dive_sample(uv, scale_b, rot_cont + rot_per_octave,   t);

    var col = mix(view_a, view_b, smooth_ph);

    // ── Outer counter-spiral (w_motif): a second, opposite-handed dive at a
    //    different phase. Lets you stack two infinite zooms — when both are
    //    cranked, the screen feels like two crystals nested inside each other.
    let motif_w = clamp(mp(MP_W_MOTIF), 0.0, 1.0);
    if (motif_w > 0.001) {
        let scale_c = base_scale / exp2(level + 0.5);
        let counter = crystal_dive_sample(uv, scale_c, -rot_cont * 0.7, t * 1.31);
        col = mix(col, max(col, counter), 0.55 * motif_w);
    }

    // ── Phase-singularity sparkle (w_band): pull the centre toward the
    //    crystal accent colour weighted by 1/r², so each zoom centre lights
    //    up briefly. Reads as the camera "punching through" a unit cell.
    let band_w = clamp(mp(MP_W_BAND), 0.0, 1.0);
    if (band_w > 0.001) {
        let r2     = dot(uv, uv);
        let glow   = exp(-r2 * 8.0) * (1.0 - smooth_ph) * band_w;
        col        = col + u.crystal_color.xyz * 1.4 * glow;
    }

    // ── Tunnel vignette (w_lattice): darkens the rim, focusing attention on
    //    the dive. Magnitude rises with w_lattice; 0 means flat field.
    let tunnel_w = clamp(mp(MP_W_LATTICE), 0.0, 1.0);
    let r        = length(uv);
    col          = col * mix(1.0, 1.0 - 0.55 * smoothstep(0.0, 1.4, r), tunnel_w);

    // ── Contrast / saturation lift driven by iso_level — makes the field
    //    snap rather than smear, useful when the sequencer crossfades.
    let lift = 0.6 + 0.9 * clamp(mp(MP_ISO_LEVEL), 0.0, 1.0);
    col      = pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / lift));

    // ── Hue rotation by color_shift — keeps every mode reactable from the
    //    sequencer / LFO routing for color_shift.
    let kvec  = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs    = cos(mp(MP_COLOR_SHIFT) * TAU);
    let sn    = sin(mp(MP_COLOR_SHIFT) * TAU);
    col       = col * cs + cross(kvec, col) * sn + kvec * dot(kvec, col) * (1.0 - cs);

    // ── field_mix: blend the colour with a desaturated version. At
    //    field_mix=0 we get the original; at 1, a near-monochrome dive.
    let mono  = vec3<f32>(dot(col, vec3<f32>(0.30, 0.59, 0.11)));
    col       = mix(col, mono, clamp(mp(MP_FIELD_MIX) - 0.5, 0.0, 0.5));

    return col;
}
