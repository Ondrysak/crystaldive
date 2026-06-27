// Mode: KURAMOTO -- generative coupled-oscillator field (Un-0 inspired),
// rendered in the repo's phase-field visual language.
//
// The dynamics follow Unconventional AI's Un-0 image model: a population of
// Kuramoto oscillators is released from a random seed, biased toward a class,
// and integrated to a snapshot time T, where the synchronisation order
// parameter r·e^{iψ} = (1/N) Σ aⱼ e^{iθⱼ} measures how locked the population is.
//
// The READOUT then treats every *pixel* as an oscillator entrained by that mean
// field (Adler equation θ̇ = ω + K·r·sin(ψ−θ)), solved in closed form so the
// field is continuous and high-resolution rather than a coarse upsampled grid:
//
//   • Crystal-based init.  The initial phase is the argument of the complex
//     crystal field  ψ_c(x) = cf₀(x) + i·cf₁(x).  Its zeros are genuine phase
//     vortices — the topological skeleton each material imprints on the seed.
//   • Mean-field locking.  Pixels whose crystal detuning ω(x) is small compared
//     to the drive K·r lock to the (class-warped) mean phase; the rest keep
//     drifting on the crystal phase. Locked + drifting regions coexisting is a
//     chimera state, the hallmark of coupled-oscillator media.
//   • As the cycle plays, r grows, locking spreads, and the image settles.
//
// Rendering borrows three sibling modes so the look connects to the rest of the
// catalog: PHASE (cyclic domain colour + bright vortex cores), SPIN TEXTURE
// (oriented oscillator "hands" on a lattice — the metronome arms), and NEMATIC
// (magenta defect-core glow). Each 6 s cycle re-seeds for a fresh generation.

const KURA_SIDE:   i32 = 4;      // conditioning grid is KURA_SIDE x KURA_SIDE
const KURA_N:      i32 = 16;     // KURA_SIDE * KURA_SIDE population oscillators
const KURA_STEPS:  i32 = 10;     // max Euler integration steps to snapshot T
const KURA_DT:     f32 = 0.22;   // Euler step
const KURA_NCLASS: f32 = 6.0;    // number of selectable "classes"
const KURA_OSPREAD: f32 = 3.6;   // natural-frequency (detuning) spread

// Free-slot accessor with a fallback (preset/random generators leave free
// slots at 0; fall back to the default so the mode never goes dead).
fn kura_slot(s: u32, def: f32) -> f32 { let v = mp(s); return select(def, v, v > 1e-4); }

fn kura_hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

// Conditioning-grid position of oscillator i, in [-1, 1]^2 (cell centres).
fn kura_pos(i: i32) -> vec2<f32> {
    let col = f32(i % KURA_SIDE);
    let row = f32(i / KURA_SIDE);
    return (vec2<f32>(col, row) + 0.5) / f32(KURA_SIDE) * 2.0 - 1.0;
}

// Class-conditional spatial template: different classes settle into different
// arrangements (radial blooms, directional bands, ...).
fn kura_class_target(pos: vec2<f32>, c: f32) -> f32 {
    let ang  = c * 1.7;                               // class-dependent orientation
    let dir  = vec2<f32>(cos(ang), sin(ang));
    let freq = 1.5 + c * 0.6;
    let w_dir = dot(pos, dir) * freq * TAU * 0.5;      // directional plane wave
    let w_rad = length(pos) * (2.0 + c) * TAU * 0.5;   // radial bloom
    return mix(w_dir, w_rad, fract(c * 0.5));          // alternate pattern families
}

fn kura_palette(h: f32) -> vec3<f32> {
    return 0.5 + 0.5 * cos(TAU * (h + vec3<f32>(0.0, 0.33, 0.67)));
}

// SPIN-TEXTURE-style oriented glyph: a little "hand" through each lattice cell
// pointing along the local oscillator phase, with a pivot dot (a metronome arm).
fn kura_hand(uv: vec2<f32>, ang: f32, density: f32, locked: f32) -> f32 {
    let cell   = floor(uv * density);
    let center = (cell + vec2<f32>(0.5)) / density;
    let dir    = vec2<f32>(cos(ang), sin(ang));
    let side   = vec2<f32>(-dir.y, dir.x);
    let d      = uv - center;
    let along  = dot(d, dir);
    let across = dot(d, side);
    let hl     = (0.30 + 0.12 * locked) / density;     // longer arm when locked
    let hw     = 0.05 / density;
    let body   = smoothstep(hl, hl * 0.35, abs(along));
    let wid    = 1.0 - smoothstep(hw * 0.4, hw, abs(across));
    let pivot  = exp(-dot(d, d) * density * density * 20.0);
    return clamp(body * wid + pivot * 0.7, 0.0, 1.0);
}

fn render_kuramoto(uv: vec2<f32>) -> vec3<f32> {
    // ── controls ─────────────────────────────────────────────────────────
    let kscale   = mp(MP_KSCALE);                 // crystal sampling scale
    let speed    = mp(MP_SPEED);
    let zoom     = max(mp(MP_ZOOM), 0.08);
    let cshift   = mp(MP_COLOR_SHIFT);            // hue (slot 4)
    let coupling = mp(2u) * 8.0;                  // K   (slot 2, "coupling")
    let kclass   = mp(3u);                        // Kc  (slot 3, "class_bias", 0..1)
    var cls      = floor(clamp(mp(13u), 0.0, 0.999) * KURA_NCLASS); // class (free 13)
    let crys_w   = clamp(kura_slot(12u, 0.6), 0.0, 1.0);  // crystal structure (free 12)
    let hands_a  = kura_slot(9u, 0.5);            // oscillator-hand visibility (free 9)
    let cores_a  = kura_slot(10u, 0.5);           // defect-core glow (free 10)
    let seed0    = floor(kura_slot(11u, 0.5) * 64.0); // base seed (free 11)

    // Mouse held → pick the class by cursor x.
    if (u.mouse_down >= 0.5) {
        cls = floor(clamp(u.mouse.x, 0.0, 0.999) * KURA_NCLASS);
    }

    // ── looping playback: re-seed each cycle, integrate to a growing T ─────
    let cyc_len  = 6.0;                           // seconds per generative dive
    let prog     = u.time * speed / cyc_len;
    let cycle    = floor(prog);
    let phase_t  = fract(prog);                   // 0 → random, 1 → settled
    let seed     = seed0 + cycle * 7.0;
    let nsteps   = i32(clamp(phase_t * f32(KURA_STEPS) + 0.001, 0.0, f32(KURA_STEPS)));
    let ng       = i32(u.num_g);

    // ── 1+2. seed & integrate the population → order parameter r·e^{iψ} ────
    var th: array<f32, 16>;
    var om: array<f32, 16>;
    var am: array<f32, 16>;
    for (var i = 0i; i < KURA_N; i = i + 1i) {
        let fi  = f32(i);
        let rnd = kura_hash(vec2<f32>(fi, seed));
        var cph = 0.0; var gmag = fi * 0.37; var amp = 1.0;
        if (i < ng) {                                   // crystal prior (O(1))
            let ga = g_block.gamp[i];
            cph    = g_block.phases[i].x;
            gmag   = length(ga.xyz) * kscale;
            amp    = max(ga.w, 0.05);
        }
        th[i] = mix(rnd * TAU, cph + gmag * 0.5, crys_w);
        om[i] = 0.6 * sin(gmag * 1.3 + cph) + 0.25 * (kura_hash(vec2<f32>(fi, seed + 17.0)) - 0.5);
        am[i] = amp;
    }
    let nu_c = 0.4 + cls * 0.13;
    for (var s = 0i; s < KURA_STEPS; s = s + 1i) {
        if (s >= nsteps) { break; }
        var cc = 0.0; var ss = 0.0; var wn = 0.0;
        for (var i = 0i; i < KURA_N; i = i + 1i) {
            cc = cc + am[i] * cos(th[i]); ss = ss + am[i] * sin(th[i]); wn = wn + am[i];
        }
        let inv  = 1.0 / max(wn, 1e-4);
        let psi_s = atan2(ss * inv, cc * inv);
        let rmag_s = sqrt(cc * cc + ss * ss) * inv;
        let phi_s = TAU * (cls / KURA_NCLASS) + f32(s) * KURA_DT * nu_c;
        for (var i = 0i; i < KURA_N; i = i + 1i) {
            let tgt = kura_class_target(kura_pos(i), cls) + phi_s;
            th[i] = th[i] + KURA_DT * (om[i] + coupling * rmag_s * sin(psi_s - th[i])
                                            + kclass * 5.0 * sin(tgt - th[i]));
        }
    }
    var cc = 0.0; var ss = 0.0; var wn = 0.0;
    for (var i = 0i; i < KURA_N; i = i + 1i) {
        cc = cc + am[i] * cos(th[i]); ss = ss + am[i] * sin(th[i]); wn = wn + am[i];
    }
    let inv  = 1.0 / max(wn, 1e-4);
    let psi  = atan2(ss * inv, cc * inv);
    let rmag = sqrt(cc * cc + ss * ss) * inv;     // global synchronisation 0..1

    // ── per-pixel oscillator entrained by the mean field (Adler readout) ───
    let q = uv / zoom;
    // per-cycle seed offset: a different slice of the crystal each generation
    let s_off = (vec2<f32>(kura_hash(vec2<f32>(seed, 11.0)),
                           kura_hash(vec2<f32>(seed, 23.0))) - 0.5) * 12.0;
    let ct = u.time * speed * 0.05;
    // complex crystal field ψ_c = cf0 + i·cf1 (the proven crystal_field/cf2 twin
    // PHASE uses) → initial phase + a real vortex skeleton (zeros of ψ_c).
    let p   = vec3<f32>((q + s_off) * 2.2, ct);
    let cf0 = crystal_field(p);
    let cf1 = cf2(p);
    let theta_c = atan2(cf1, cf0);
    let amp0    = clamp(length(vec2<f32>(cf0, cf1)) * (0.6 + 0.9 * crys_w), 0.0, 1.0);
    let omega_x = KURA_OSPREAD * (cf0 - cf1) * 0.5;        // crystal detuning landscape

    // class-warped mean phase the locked population pulls toward
    let psi_eff = psi + kclass * kura_class_target(q, cls) * 0.25;

    // Adler locking: a pixel entrains where the drive K·r beats its detuning |ω|.
    let drive = coupling * rmag;
    let lock  = smoothstep(1.15, 0.30, abs(omega_x) / max(drive, 1e-3));
    let sync  = clamp(lock * rmag, 0.0, 1.0);   // per-pixel coherence (0..1)

    // The crystal PHASE portrait is the substrate; synchronisation only *pulls*
    // locked phases toward the mean (capped), so coherent patches and drifting
    // rainbow domains coexist — a chimera — instead of the mean washing it out.
    let dphi    = atan2(sin(psi_eff - theta_c), cos(psi_eff - theta_c));
    let ang     = theta_c + sync * dphi * 0.55;
    let amp     = amp0;

    // ── render in the phase-field language (PHASE / SPIN TEXTURE / NEMATIC) ─
    // PHASE: cyclic domain colour, amplitude-gated.
    let hue = fract(ang / TAU + cshift + cls * 0.13 + u.time * speed * 0.01);
    var col = kura_palette(hue) * smoothstep(0.0, 0.30, amp) * 1.6;

    // PHASE nodal lines where each component vanishes (the domain-wall lattice).
    col = col + smoothstep(0.025, 0.0, abs(cf0)) * vec3<f32>(1.0, 0.9, 0.5) * 0.7;
    col = col + smoothstep(0.025, 0.0, abs(cf1)) * vec3<f32>(0.5, 0.9, 1.0) * 0.7;

    // NEMATIC/PHASE: bright magenta cores at phase vortices (amp → 0) — the
    // unresolved topological defects, brightest where still incoherent.
    let core = 1.0 - smoothstep(0.0, 0.07, amp);
    col = col + core * vec3<f32>(1.20, 0.50, 1.05) * (0.7 + 1.4 * cores_a) * (1.3 - sync);

    let bg = mix(vec3<f32>(0.010, 0.012, 0.028), u.crystal_color.xyz * 0.06, 0.5);
    col = mix(bg, col, smoothstep(0.0, 0.14, amp));

    // SPIN TEXTURE: a lattice of oscillator "hands" pointing along the local
    // phase — dark/scattered while drifting, golden/aligned once locked.
    let dens = (5.0 + hands_a * 13.0) / max(zoom, 0.25);
    let hand = kura_hand(uv, ang, dens, sync);
    let hand_col = mix(vec3<f32>(0.03, 0.04, 0.07), vec3<f32>(1.10, 0.96, 0.55), sync);
    col = mix(col, hand_col, hand * (0.22 + 0.45 * hands_a));

    // synchronisation sheen: coherent patches glow with the crystal accent.
    col = col + u.crystal_color.xyz * sync * smoothstep(0.1, 0.5, amp) * 0.30;

    // chroma boost so domains read as saturated fields, not pastel washes
    let lum = dot(col, vec3<f32>(0.299, 0.587, 0.114));
    col = max(mix(vec3<f32>(lum), col, 1.4), vec3<f32>(0.0));

    // mouse probe glow
    if (u.mouse_down >= 0.5) {
        let m = (u.mouse - vec2<f32>(0.5, 0.5)) * vec2<f32>(2.0 * u.aspect, -2.0);
        col = col + exp(-length(uv - m) * 26.0) * vec3<f32>(1.0, 0.95, 0.8) * 0.5;
    }

    return col;
}
