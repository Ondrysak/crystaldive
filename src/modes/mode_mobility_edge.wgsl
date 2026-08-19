// ── MOBILITY EDGE — transfer-matrix localization spectrum of the crystal ─
//
// Entirely different math from every other mode: no field synthesis, no
// raymarching, no escape iteration. Each pixel solves a 1D tight-binding
// Schrödinger problem
//
//     ψₙ₊₁ = (E − λ·V(n))·ψₙ − ψₙ₋₁
//
// whose quasiperiodic potential V(n) is the loaded crystal sampled along a
// slowly precessing 1D cut: every reciprocal vector Gₖ projects to a scalar
// frequency qₖ = Gₖ·d̂ (rad/site), weighted by its structure-factor amplitude
// and phase. The pixel's x-coordinate is the energy E, y is the coupling λ.
//
// Two spectral observables fall out of one sweep of the transfer-matrix
// cocycle Tₙ = [[E−λVₙ, −1],[1, 0]]:
//   • γ — the Lyapunov exponent (log growth of ‖Tₙ⋯T₁‖) → brightness.
//     γ≈0: extended Bloch-like states (bright bands). γ>0: localized states
//     and spectral gaps (dark). The bright↔dark frontier is the mobility edge.
//   • Sturm node count (sign changes of ψₙ with ψ₀=0, ψ₁=1) = number of
//     eigenvalues below E → integrated density of states → hue. Constant in
//     gaps, sweeping through the rainbow across each band.

fn mobility_edge_palette(h: f32) -> vec3<f32> {
    let c = 0.5 + 0.5 * cos(TAU * (h + vec3<f32>(0.00, 0.31, 0.64)));
    return c * c;
}

fn render_mobility_edge(uv: vec2<f32>) -> vec3<f32> {
    let e_span   = clamp(mp(0u), 1.0, 8.0);
    let drift    = mp(1u);
    let lam_max  = clamp(mp(2u), 0.2, 5.0);
    let n_sites  = clamp(i32(round(mp(3u))), 32, 120);
    let hue      = mp(4u);
    let zoom     = max(mp(5u), 0.25);
    let cut_spin = clamp(mp(6u), 0.0, 2.0);
    let contrast = clamp(mp(7u), 0.2, 4.0);
    let glow_amt = clamp(mp(8u), 0.0, 2.0);
    let t = u.time;

    // ── Warped chart: the screen is a curved, precessing slice through the
    // full spectral manifold (E, λ, phason φ, cut direction d̂) rather than a
    // flat (E, λ) plot. Slots 9..12 are free knobs; generator zeros fall back
    // to defaults (see CONTRACT.md).
    let warp_in = mp(9u);  let warp   = select(0.55, clamp(warp_in, 0.0, 2.0), warp_in > 1e-4);
    let tun_in  = mp(10u); let tunnel = select(0.30, clamp(tun_in, 0.0, 1.0), tun_in > 1e-4);
    let pha_in  = mp(11u); let phason = select(0.80, clamp(pha_in, 0.0, 3.0), pha_in > 1e-4);
    let fan_in  = mp(12u); let fan    = select(0.50, clamp(fan_in, 0.0, 2.0), fan_in > 1e-4);

    var p = uv / zoom;
    let r0  = length(p) + 1e-5;
    let th0 = atan2(p.y, p.x);

    // Swirl: angular shear breathing with radius and time, plus a two-petal
    // term so the shear is not radially symmetric.
    let sw = warp * (0.8 * sin(1.7 * r0 - t * 0.33)
                   + 0.45 * r0 * sin(th0 * 2.0 + t * 0.21));
    p = vec2<f32>(cos(th0 + sw), sin(th0 + sw)) * r0;

    // Project through the crystal: the two strongest G-vectors displace the
    // spectral plane itself, so the chart carries the lattice.
    if u.num_g >= 2u {
        let ga0 = g_block.gamp[0];
        let ga1 = g_block.gamp[1];
        p += warp * 0.22 * vec2<f32>(
            sin(dot(ga0.xy, p) * 1.9 + t * 0.27),
            sin(dot(ga1.xy, p) * 1.9 - t * 0.19),
        );
    }

    // Tunnel: blend the Cartesian chart into a mirrored log-polar chart —
    // energy becomes azimuth, coupling becomes wrapped log-radius, so the
    // band tree closes into concentric ring spectra diving toward the center.
    let ang  = abs(atan2(p.y, p.x)) / 3.14159265; // 0..1 mirrored → no seam
    let lr   = log(length(p) + 0.12);
    let ring = 1.0 - 2.0 * abs(fract(lr * 0.8 - t * 0.02 * (1.0 + drift)) - 0.5);
    let ex = mix(p.x, (ang * 2.0 - 1.0) * 1.35, tunnel);
    let ey = mix(p.y, ring * 2.0 - 1.0, tunnel);

    // Mouse drags the energy window and biases the coupling.
    var e_center = 0.0;
    var lam_bias = 0.0;
    if u.mouse_down >= 0.5 {
        e_center = (u.mouse.x * 2.0 - 1.0) * e_span * 0.6;
        lam_bias = (0.5 - u.mouse.y) * lam_max * 0.8;
    }
    let energy = e_center + ex * e_span;
    let ny     = clamp(0.5 + 0.5 * ey, 0.0, 1.0);
    let lam    = clamp(mix(0.0, lam_max, ny * ny) + lam_bias, 0.0, 6.0);

    // Phason shear: each pixel starts its chain at a different point on the
    // phason torus (a tilted cut through the hull dimensions), so filaments
    // bend and shimmer instead of standing straight.
    let psn = phason * dot(p, vec2<f32>(cos(t * 0.07), sin(t * 0.07)));

    // Fan of cuts: the sampling direction precesses in time AND swings with
    // screen position — each pixel slices reciprocal space differently.
    let a   = t * cut_spin * 0.2 + 0.7
            + fan * (th0 * 0.35 + 0.3 * sin(r0 * 1.3 - t * 0.17));
    let dir = normalize(vec3<f32>(cos(a), sin(a) * 0.9, 0.42 * sin(a * 0.37 + 1.3)));

    // Project up to 6 G-vectors onto the cut → chain frequencies/amps/phases.
    var freq: array<f32, 6>;
    var amp:  array<f32, 6>;
    var pha:  array<f32, 6>;
    var nk = 0;
    var amp_sum = 0.0;
    let ng = i32(u.num_g);
    for (var i = 0; i < 6; i++) {
        if i >= ng { break; }
        let ga = g_block.gamp[i];
        freq[nk] = dot(ga.xyz, dir) * 1.35;
        amp[nk]  = max(abs(ga.w), 1e-4);
        pha[nk]  = g_block.phases[i].x + (t * drift + psn) * (0.31 + 0.17 * f32(i));
        amp_sum += amp[nk];
        nk++;
    }
    if nk == 0 {
        // No crystal loaded → pure Aubry–André chain (golden-mean cosine).
        freq[0] = TAU * 0.6180339887;
        amp[0]  = 1.0;
        pha[0]  = t * drift * 0.31 + psn;
        amp_sum = 1.0;
        nk = 1;
    }
    let inv_amp = 1.0 / amp_sum;

    // Oscillator bank: cos(qₖn+φₖ) advances by complex rotation each site —
    // zero transcendentals inside the sweep.
    var osc: array<vec2<f32>, 6>;
    var rot: array<vec2<f32>, 6>;
    for (var k = 0; k < 6; k++) {
        if k >= nk { break; }
        osc[k] = vec2<f32>(cos(pha[k]), sin(pha[k]));
        rot[k] = vec2<f32>(cos(freq[k]), sin(freq[k]));
    }

    // Transfer-matrix sweep, Dirichlet boundary (ψ₀ = 0, ψ₁ = 1).
    var psi_prev = 0.0;
    var psi      = 1.0;
    var log_acc  = 0.0;
    var nodes    = 0.0;
    for (var n = 0; n < 120; n++) {
        if n >= n_sites { break; }
        var v = 0.0;
        for (var k = 0; k < 6; k++) {
            if k >= nk { break; }
            v += amp[k] * osc[k].x;
            osc[k] = vec2<f32>(
                osc[k].x * rot[k].x - osc[k].y * rot[k].y,
                osc[k].x * rot[k].y + osc[k].y * rot[k].x,
            );
        }
        let psi_next = (energy - lam * v * inv_amp) * psi - psi_prev;
        nodes += select(0.0, 1.0, psi_next * psi < 0.0);
        psi_prev = psi;
        psi = psi_next;
        // Renormalize the cocycle before f32 overflow; bank the log-norm.
        if abs(psi) > 1e8 {
            psi *= 1e-8;
            psi_prev *= 1e-8;
            log_acc += 18.420681;
        }
    }
    let nsf   = f32(n_sites);
    let nrm   = max(sqrt(psi * psi + psi_prev * psi_prev), 1e-20);
    let gamma = max((log_acc + log(nrm)) / nsf, 0.0);

    // γ → brightness (extended vs localized), node count → IDOS → hue.
    let ids    = nodes / nsf;
    let g2     = gamma * contrast * 13.0;
    let bright = exp(-g2 * g2);
    let edge   = 4.0 * bright * (1.0 - bright); // peaks on the mobility edge

    let pal = mobility_edge_palette(fract(hue + ids * 0.85 + t * 0.005));
    var col = vec3<f32>(0.004, 0.003, 0.014);
    col += pal * bright * (0.5 + 0.5 * ids);
    col += mix(u.crystal_color.xyz, pal, 0.35) * edge * glow_amt * 0.35;
    col += pal * exp(-gamma * 3.0) * 0.04; // faint haze deep in the gaps
    return col;
}
