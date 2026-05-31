// ── Mode 36: ABRIKOSOV — vortex lattice on a type-II superconductor surface ─
//
// Physics:
//   Abrikosov, JETP 5, 1174 (1957)         — triangular flux-line lattice
//   Brandt, PRB 37, 2349 (1988)            — Fourier formula for B(r), |ψ|²
//   Hess et al., PRL 62, 214 (1989)        — first STM image of the lattice
//   Hess et al., PRL 64, 2711 (1990)       — 6-fold star at vortex cores
//   Hayashi et al., PRL 80, 2921 (1998)    — theory of CdGM star pattern
//
// v3 — the vortex lattice is now DERIVED from the loaded crystal's reciprocal
//      lattice (from g_tex), not hard-coded. Picking a different material in
//      the library actually changes the lattice symmetry, orientation, and
//      aspect ratio:
//
//        cubic        → square vortex lattice
//        hexagonal    → triangular (classic Abrikosov)
//        tetragonal   → square / rectangular
//        orthorhombic → rhombic
//        monoclinic   → oblique
//
//      The order parameter is additionally modulated by the actual crystal_field()
//      so the periodic content of the material (the G-vectors' phases) leaks
//      into the vortex-core depth pattern.
//
// Slider semantics:
//   iso_level   → reduced temperature t = T/Tc ∈ [0, 0.99] (wider visible range)
//   field_mix   → applied field B (lattice density: a₀ ∝ 1/√B)
//   zoom        → scan window on sample plane
//   color_shift → phase-winding hue rotation
//   w_lattice   → strength of the CdGM 6-fold star anisotropy
//   w_motif     → strength of phase-singularity rainbow ring
//   w_band      → mixing weight for crystal_field modulation of core depth

fn abrikosov_hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn abrikosov_vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let a = abrikosov_hash21(i);
    let b = abrikosov_hash21(i + vec2<f32>(1.0, 0.0));
    let c = abrikosov_hash21(i + vec2<f32>(0.0, 1.0));
    let d = abrikosov_hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn abrikosov_fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var amp = 0.55;
    var q = p;
    for (var i = 0; i < 4; i = i + 1) {
        v = v + amp * abrikosov_vnoise(q);
        q = q * 2.07 + vec2<f32>(13.7, 7.3);
        amp = amp * 0.5;
    }
    return v;
}

// Find the two shortest non-parallel in-plane reciprocal-lattice vectors of
// the currently loaded crystal. These are read from g_tex (populated by the
// Rust side from the crystal's lattice). Fallback to a canonical triangular
// pair if the crystal has too few in-plane G-vectors.
//
// Returns mat2x2(b1, b2). The mode uses these to set the SYMMETRY and
// ORIENTATION of the vortex lattice — so a hexagonal crystal gives the
// classic Abrikosov triangular lattice, cubic gives a square lattice, etc.
fn abrikosov_crystal_basis() -> mat2x2<f32> {
    var b1 = vec2<f32>(1.0, 0.0);
    var b2 = vec2<f32>(0.0, 1.0);
    var l1 = 1e9;
    var l2 = 1e9;
    let ng_ab = i32(u.num_g);
    for (var i = 0i; i < ng_ab; i++) {
        let g3 = g_block.gamp[i].xyz;
        // Project into the (x,y) plane; ignore strongly out-of-plane G's so we
        // get a clean 2D basis. (Out-of-plane = |gz| > |gxy|.)
        let gxy = g3.xy;
        let lxy = length(gxy);
        if lxy < 0.05 { continue; }
        if abs(g3.z) > lxy * 1.2 { continue; }
        if lxy < l1 {
            l2 = l1; b2 = b1;
            l1 = lxy; b1 = gxy;
        } else if lxy < l2 {
            // Reject if nearly parallel to b1 (we want a real 2D basis).
            let par = abs(dot(gxy, b1) / (lxy * l1 + 1e-9));
            if par < 0.92 {
                l2 = lxy; b2 = gxy;
            }
        }
    }
    // Fallback to triangular if we didn't find two good vectors.
    if l2 > 9e8 {
        b1 = vec2<f32>(1.0, 0.0);
        b2 = vec2<f32>(0.5, 0.8660254);
    }
    return mat2x2<f32>(b1, b2);
}

// Brandt-style Fourier modulation of |B(r) - B̄|: sum over the 6 shortest
// reciprocal-lattice K's of the *vortex lattice*. Smooth & periodic.
// Per-K accumulation. Unrolled (×6) — WGSL forbids runtime indexing of a
// function-local array<>, and there are only 6 nearest K's anyway.
fn abrikosov_brandt_term(K: vec2<f32>, p: vec2<f32>, xi2: f32, lam2: f32) -> f32 {
    let Kmag2 = dot(K, K);
    let form  = exp(-Kmag2 * xi2 * 0.5) / (1.0 + Kmag2 * lam2);
    return form * cos(dot(K, p));
}

fn abrikosov_brandt_Bz(p: vec2<f32>, K1: vec2<f32>, K2: vec2<f32>,
                       xi2: f32, lam2: f32) -> f32 {
    // 6 nearest K-vectors: ±K1, ±K2, ±(K1-K2) for a generic oblique lattice.
    return abrikosov_brandt_term( K1,      p, xi2, lam2)
         + abrikosov_brandt_term(-K1,      p, xi2, lam2)
         + abrikosov_brandt_term( K2,      p, xi2, lam2)
         + abrikosov_brandt_term(-K2,      p, xi2, lam2)
         + abrikosov_brandt_term( K1 - K2, p, xi2, lam2)
         + abrikosov_brandt_term( K2 - K1, p, xi2, lam2);
}

fn render_abrikosov(uv: vec2<f32>) -> vec3<f32> {
    // ── reduced temperature, GL parameters (wider visible response than v2)
    let t_red   = clamp(mp(MP_ISO_LEVEL), 0.0, 0.99);
    let amp_T   = sqrt(max(1.0 - t_red * t_red, 0.0));
    let one_mt  = max(1.0 - t_red,                  0.015);
    let one_mt4 = max(1.0 - t_red * t_red * t_red * t_red, 0.015);
    let xi      = 0.022 * inverseSqrt(one_mt);
    let lambda  = 0.16  * inverseSqrt(one_mt4);
    let xi2     = xi * xi;
    let lam2    = lambda * lambda;

    // ── B-field control with AC breathing.
    // Real SCs in a slowly-varying applied field show the lattice constant
    // pulsing as a₀ ∝ 1/√B. Add a ~12 % AC envelope so the lattice visibly
    // breathes; speed slider scales the oscillation frequency.
    let t_main   = u.time * mp(MP_SPEED);
    let B_static = mix(0.6, 14.0, clamp(mp(MP_FIELD_MIX), 0.0, 1.0));
    let B_field  = B_static * (1.0 + 0.12 * sin(t_main * 0.43));

    // ── CRYSTAL-DERIVED LATTICE GEOMETRY ─────────────────────────────────
    // Read the crystal's own in-plane reciprocal basis. The vortex lattice
    // inherits the same SYMMETRY (cubic→square, hex→triangular, ortho→rhombic)
    // and ORIENTATION (rotation angle from G₁). We scale the real-space period
    // by 1/√B as physical vortex lattices do.
    let cb = abrikosov_crystal_basis();
    let G1 = cb[0];
    let G2 = cb[1];
    // Unit basis (direction + relative length only — the absolute scale is
    // set by B). |b1| = 1, |b2| = |G2|/|G1|.
    let g1l = max(length(G1), 1e-4);
    let b1u = G1 / g1l;
    let b2u = G2 / g1l;
    // Lattice rotation — *much* faster than v2 so motion is actually visible.
    // 0.35 rad/s at speed = 1 → full revolution every ~18 s.
    let phi0 = t_main * 0.35;
    let cph  = cos(phi0);
    let sph  = sin(phi0);
    let R    = mat2x2<f32>(vec2<f32>(cph, sph), vec2<f32>(-sph, cph));
    // Real-space vortex lattice constants — scale ∝ 1/√B.
    let a0   = 0.40 * inverseSqrt(B_field);
    let e1   = a0 * (R * b1u);
    let e2   = a0 * (R * b2u);
    // Reciprocal of the vortex lattice (K_i · e_j = 2π δ_ij), used for Brandt.
    let det  = e1.x * e2.y - e1.y * e2.x;
    let inv_det = 1.0 / det;
    let K1 = TAU * inv_det * vec2<f32>( e2.y, -e2.x);
    let K2 = TAU * inv_det * vec2<f32>(-e1.y,  e1.x);

    // ── sample plane, with flux-flow drift + thermal librations.
    //   Flux flow:  v_L = (J × Φ₀)/(n_s e). We don't model J explicitly; instead
    //   give the drift its own Lissajous so the whole lattice slides smoothly
    //   across the frame. Magnitude scales with melt_t (driven harder near Tc).
    //   Thermal librations: smooth ~2D vector noise modulating individual vortex
    //   positions; amplitude rises with T (vortex liquid → big wobble).
    let drift_amp = 0.20 + 0.50 * smoothstep(0.0, 0.95, t_red);
    let drift     = drift_amp * vec2<f32>(
        sin(t_main * 0.27 + 1.7),
        cos(t_main * 0.31 + 0.4),
    );
    let lib_amp = 0.025 + 0.18 * smoothstep(0.3, 0.95, t_red);
    let lib = lib_amp * vec2<f32>(
        abrikosov_vnoise(uv * 1.7 + vec2<f32>(t_main * 0.9, 0.0)) - 0.5,
        abrikosov_vnoise(uv * 1.7 + vec2<f32>(0.0, t_main * 1.1)) - 0.5,
    );
    let p = uv * (2.2 / max(mp(MP_ZOOM), 0.08)) - drift + lib;

    // ── 1. Globally smooth magnetic field via Brandt sum.
    let Bz_mod = abrikosov_brandt_Bz(p, K1, K2, xi2, lam2);

    // ── 2. Order parameter from a Brandt-style Fourier sum on |ψ|².
    //      |ψ|²(r) = 1 - Σ_K f(K)·cos(K·r), truncated to the 6 nearest K's.
    //      Smooth, periodic, has zeros where ϑ-ansatz does, and stays well-defined
    //      for any (b1, b2) — including non-triangular cells. Unrolled to dodge
    //      WGSL's restriction on runtime-indexed function arrays.
    let Kmag2_1   = dot(K1, K1);
    let Kmag2_2   = dot(K2, K2);
    let K_diff    = K1 - K2;
    let Kmag2_d   = dot(K_diff, K_diff);
    let f1        = exp(-Kmag2_1 * xi2 * 0.35);
    let f2        = exp(-Kmag2_2 * xi2 * 0.35);
    let fd        = exp(-Kmag2_d * xi2 * 0.35);
    let amp_acc   = 2.0 * (f1 + f2 + fd);
    let sum       = 2.0 * (f1 * cos(dot( K1, p))
                         + f2 * cos(dot( K2, p))
                         + fd * cos(dot(K_diff, p)));
    let psi2_fourier = clamp(0.5 - 0.5 * sum / max(amp_acc, 1e-4), 0.0, 1.0);

    // ── 3. Nearest-vortex lookup for CdGM 6-fold star + phase winding.
    // Solve p = i·e1 + j·e2 via Cramer.
    let i_f = ( p.x * e2.y - p.y * e2.x) * inv_det;
    let j_f = (-p.x * e1.y + p.y * e1.x) * inv_det;
    let i0 = floor(i_f);
    let j0 = floor(j_f);
    var d2_min    = 1e9;
    var d_to_core = vec2<f32>(0.0);
    var phase_acc = 0.0;
    for (var di = -1; di <= 1; di = di + 1) {
        for (var dj = -1; dj <= 1; dj = dj + 1) {
            let cell   = vec2<f32>(i0 + f32(di), j0 + f32(dj));
            let centre = cell.x * e1 + cell.y * e2;
            let d      = p - centre;
            let d2     = dot(d, d);
            if d2 < d2_min {
                d2_min    = d2;
                d_to_core = d;
            }
            phase_acc = phase_acc + atan2(d.y, d.x);
        }
    }
    let theta_local = atan2(d_to_core.y, d_to_core.x);
    // CdGM star arms point along the lattice's nearest-neighbour direction.
    let star_axis   = atan2(b1u.y, b1u.x) + phi0;
    let n_arms      = 6.0;                  // hexagonal/triangular materials
    let star_amp    = clamp(mp(MP_W_LATTICE), 0.0, 1.0) * (0.55 * amp_T);
    let star        = 1.0 + star_amp * cos(n_arms * (theta_local - star_axis));
    let core_env    = exp(-d2_min / (2.0 * 1.8 * 1.8 * xi2));

    // ── 4. Material fingerprint: modulate the core depth by the actual
    //      crystal_field() of the loaded material. Different G-vector phases →
    //      different core-to-core intensity pattern. Strength is controlled by
    //      w_band so the user can dial it in.
    let mat_mod_raw = crystal_field(vec3<f32>(p * 1.3, u.time * mp(MP_SPEED) * 0.05));
    let mat_mod     = 0.5 + 0.5 * sin(mat_mod_raw * 1.5);
    let mat_w       = clamp(mp(MP_W_BAND), 0.0, 1.0) * 0.7;

    // ── 5. Vortex-liquid melt
    let melt_t     = smoothstep(0.55, 0.93, t_red);
    let liquid_t   = smoothstep(0.85, 0.99, t_red);
    let wash       = abrikosov_fbm(p * 4.5 + vec2<f32>(0.0, u.time * 0.25));
    let psi2_solid = mix(psi2_fourier, psi2_fourier * (0.6 + 0.4 * mat_mod), mat_w);
    let psi2_final = mix(psi2_solid, 0.35 + 0.30 * wash, liquid_t);

    // ── 6. Colour assembly. Crystal accent colour now drives BOTH background
    //      and core hue so different materials really look different.
    let bg_mix     = 0.35;
    let meissner   = mix(vec3<f32>(0.010, 0.022, 0.052),
                         u.crystal_color.xyz * 0.18,
                         bg_mix);
    let core_warm  = mix(vec3<f32>(1.55, 0.95, 0.40),
                         vec3<f32>(0.70, 0.55, 0.45),
                         t_red);
    // Tint the core slightly by the crystal accent so each material is recognisable.
    let core_col   = mix(core_warm,
                         core_warm * (0.4 + 0.9 * u.crystal_color.xyz),
                         0.35);
    let hue   = fract(phase_acc / TAU + mp(MP_COLOR_SHIFT) + u.time * mp(MP_SPEED) * 0.05);
    var ring  = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
    ring      = mix(ring, u.crystal_color.xyz, 0.30);

    let suppress    = 1.0 - psi2_final;
    let ring_weight = clamp(mp(MP_W_MOTIF), 0.0, 1.0);
    var col = meissner
            + core_col * (suppress * (0.55 + 0.65 * amp_T) + core_env * star * 0.65)
            + ring     * suppress * 0.30 * ring_weight
            + u.crystal_color.xyz * Bz_mod * 0.18 * amp_T;

    // Subtle thermal shimmer near melting (FBM, not hash — no pixelation).
    if melt_t > 0.0 {
        let shim = (abrikosov_fbm(p * 12.0 + u.time * vec2<f32>(0.3, -0.2)) - 0.5);
        col = col + vec3<f32>(shim) * 0.10 * melt_t;
    }

    return col;
}
