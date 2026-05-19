//! Per-mode metadata: display name, tagline, equations (unicode-styled math),
//! and slider/usage notes. One source of truth for the 46 visualizer modes.
//!
//! Both `crate::main`'s `MODE_NAMES` array and `crate::bench`'s name slice are
//! derived from `MODES` so they stay in sync automatically.
//!
//! `equations` strings are rendered with a monospace font in the egui mode-info
//! window, so they may use unicode subscripts (ᵢⱼₖ), superscripts (²³), Greek
//! letters (αβγθΣ), and math glyphs (·×∇∂∝→≈) directly.

pub struct ModeInfo {
    pub idx: usize,
    pub name: &'static str,
    pub tagline: &'static str,
    pub equations: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModeArea {
    Crystal,
    Reciprocal,
    Electronic,
    Materials,
    Classical,
    Fields,
}

impl ModeArea {
    pub const ALL: [ModeArea; 6] = [
        ModeArea::Crystal,
        ModeArea::Reciprocal,
        ModeArea::Electronic,
        ModeArea::Materials,
        ModeArea::Classical,
        ModeArea::Fields,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            ModeArea::Crystal => "Crystal",
            ModeArea::Reciprocal => "Reciprocal",
            ModeArea::Electronic => "Electronic",
            ModeArea::Materials => "Materials",
            ModeArea::Classical => "Classical",
            ModeArea::Fields => "Fields",
        }
    }

    pub const fn title(self) -> &'static str {
        match self {
            ModeArea::Crystal => "Crystal Fields",
            ModeArea::Reciprocal => "Reciprocal & Diffraction",
            ModeArea::Electronic => "Electronic & Topological",
            ModeArea::Materials => "Materials & Order",
            ModeArea::Classical => "Classical Continuum",
            ModeArea::Fields => "Fields & Relativity",
        }
    }
}

impl ModeInfo {
    pub const fn area(&self) -> ModeArea {
        match self.idx {
            0..=8 | 11 | 13 | 24 | 37 => ModeArea::Crystal,
            9 | 10 | 14 | 18 => ModeArea::Reciprocal,
            15 | 17 | 20..=22 | 28..=34 | 36 | 39 | 40 => ModeArea::Electronic,
            12 | 16 | 19 | 23 | 25 | 26 | 35 | 38 => ModeArea::Materials,
            27 | 42 | 43 => ModeArea::Classical,
            41 | 44 | 45 => ModeArea::Fields,
            _ => ModeArea::Crystal,
        }
    }
}

pub const MODES: [ModeInfo; 46] = [
    ModeInfo {
        idx: 0, name: "3D ISO",
        tagline: "Ray-marched isosurface of the crystal-field scalar.",
        equations: &[
            "f(r) = Σᵢ aᵢ sin(Gᵢ·r + φᵢ)",
            "surface: f(r) = c",
            "shading: n̂ = −∇f / |∇f|",
        ],
        notes: "Camera orbits, mouse-drag to aim. iso_level shifts the level set; field_mix blends the primary field with the offset twin cf2; w_lattice / w_motif / w_band weight the three sin variants summed into f.",
    },
    ModeInfo {
        idx: 1, name: "BZ SLICE",
        tagline: "Brillouin-zone slice of f(r) at z = t·speed.",
        equations: &[
            "p(uv,t) = (3·uv, 0.3 t)",
            "draw  f(p),  zero set  |f|≈0,",
            "and iso line  |f − ½·iso| ≈ 0",
        ],
        notes: "A flat 2D cut through the periodic crystal field that scrolls in z. iso_level picks the highlighted contour; color_shift hue-rotates the palette.",
    },
    ModeInfo {
        idx: 2, name: "FERMI",
        tagline: "Sweeping z-slice as a Fermi-surface scan.",
        equations: &[
            "z(t) = 0.8 sin(0.5 t)",
            "draw  iso(f, ½·iso), iso(f₂, −0.4·iso)",
        ],
        notes: "Same field as BZ SLICE, but z oscillates so you see one full traversal of the lattice's reciprocal slab. iso_level controls which energy surface is highlighted.",
    },
    ModeInfo {
        idx: 3, name: "DENSITY",
        tagline: "Projected charge density ρ ∝ ⟨ψ²⟩.",
        equations: &[
            "ρ(uv) ≈ (1/N) Σ_z f(uv, z)²",
            "(6 z-samples)",
        ],
        notes: "Quick proxy for |ψ|² integrated along z. Hue cycles with ρ; brightness pow(ρ, 0.4) gives a perceptual rolloff.",
    },
    ModeInfo {
        idx: 4, name: "NODAL",
        tagline: "Zero sets of f and f₂ — nodal lines and their crossings.",
        equations: &[
            "lines: |f|≈0  ∪  |f₂|≈0",
            "crossings: |f|+|f₂| ≈ 0",
        ],
        notes: "Highlights where the wavefunction changes sign. Bright yellow at crossings (both fields vanish). Useful intuition for nodal surfaces in band structures.",
    },
    ModeInfo {
        idx: 5, name: "PHASE",
        tagline: "Phase portrait of the complex field ψ = f + i·f₂.",
        equations: &[
            "amp = √(f² + f₂²)",
            "arg = atan2(f₂, f)",
            "hue ← arg/2π,   bright ← amp",
        ],
        notes: "Classic complex-function visualisation. Vortices appear where amp→0 and arg wraps. iso_level / color_shift recolour.",
    },
    ModeInfo {
        idx: 6, name: "STRIPES",
        tagline: "Quantised DOS-like stripes via fract(f·N).",
        equations: &[
            "N = 6 + 14·iso",
            "stripe = fract(mix(f, f₂)·N)",
            "draw thin lines near stripe = 0",
        ],
        notes: "Each stripe is an equi-energy contour at evenly-spaced levels — like a quick-and-dirty density of states. iso_level adds more contours.",
    },
    ModeInfo {
        idx: 7, name: "WARP",
        tagline: "Gradient-lensed crystal field.",
        equations: &[
            "g(p) = ∇f(p) · 2ε",
            "p_warped = p + field_mix · g",
            "render f(p_warped)",
        ],
        notes: "The field bends rays toward its own gradient. field_mix controls the bend strength; w_band sets shimmer. Looks like the lattice through a heat haze.",
    },
    ModeInfo {
        idx: 8, name: "LINKS",
        tagline: "Hollow sphere chained by crystal-field-noised links (SDF).",
        equations: &[
            "sphere: ||r| − 0.65| − 0.04",
            "link:  sd_link(r̂, …)",
            "scene = smin(sphere, link, 0.18)",
            "    + 0.10·f(p) near surface",
        ],
        notes: "GLKITTY-style sphere-trace. Crystal field noise perturbs the surface only near hit. Two-pass SDF: cheap geometric far, full noise near.",
    },
    ModeInfo {
        idx: 9, name: "XRD",
        tagline: "Simulated X-ray diffraction pattern (spots + powder rings).",
        equations: &[
            "Bragg: 2d sin θ = nλ",
            "spot(uv,G) ∝ |F(G)|² · e^(−ε·z²)",
            "        · e^(−|uv−s(G)|²/σ²)",
            "powder:  ∑ exp(−(|uv|−|G|)²·k)",
        ],
        notes: "Spots ← projected G-vectors, Ewald-weighted. Rings ← powder average. iso_level sharpens spots (smaller σ).",
    },
    ModeInfo {
        idx: 10, name: "RECIP",
        tagline: "3D reciprocal-lattice point cloud.",
        equations: &[
            "G points: {n₁b₁ + n₂b₂ + n₃b₃}",
            "radius ∝ |F(G)|, raycast spheres",
        ],
        notes: "Sphere-traced rendering of the loaded crystal's actual G-vectors. Spokes connect each G to Γ. Mouse-orbit; iso_level scales sphere size.",
    },
    ModeInfo {
        idx: 11, name: "NONEUC",
        tagline: "Kleinian circle-inversion kaleidoscope.",
        equations: &[
            "inv(z; c, r) = c + r²(z−c)/|z−c|²",
            "iterate 52× through 4 G-derived circles",
            "tint each step by f(z), decay e^(−i/20)",
        ],
        notes: "Hyperbolic (non-Euclidean) tiling — circles inverted into themselves. Centres come from the first four G-vectors so different materials give different fundamental groups.",
    },
    ModeInfo {
        idx: 12, name: "PHONON",
        tagline: "Animated 5×5×5 atom lattice on one phonon eigenmode.",
        equations: &[
            "R'(t) = R + ê·A·cos(q·R − ω t)",
            "ω(q) ≈ √(|q|+0.1) · speed · 1.5",
            "ê = mix(transverse, longitudinal, field_mix)",
        ],
        notes: "Picks q from g_tex[iso_level·num_g]. field_mix slides between transverse (T) and longitudinal (L) polarisations. Acoustic-like dispersion only — not a real eigensolver.",
    },
    ModeInfo {
        idx: 13, name: "MOIRE",
        tagline: "Twisted-bilayer interference of two crystal fields.",
        equations: &[
            "θ = field_mix · 30°",
            "draw  f(p) · f(Rθ p)   (product)",
            "+ contour at  f + f_rot = 0",
        ],
        notes: "Two copies of the lattice, one rotated by θ. Magic-angle moiré patterns and pinwheel ripples appear at small θ. iso_level sharpens contours.",
    },
    ModeInfo {
        idx: 14, name: "EWALD",
        tagline: "Ewald sphere sweeping the 3D reciprocal lattice.",
        equations: &[
            "|kᵢ + G| = |kᵢ|   (Laue condition)",
            "shell centred at −kᵢ, radius |kᵢ|=1/λ",
            "G flashes when on the shell",
        ],
        notes: "Geometric construction of Bragg diffraction. Mouse rotates the incident beam; iso_level scales |kᵢ| so you can pick which G's satisfy Laue.",
    },
    ModeInfo {
        idx: 15, name: "WANNIER",
        tagline: "Localised Wannier orbital |ψ|² isosurface.",
        equations: &[
            "ψ(r) = e^(−0.6 r²) Σᵢ aᵢ e^(i Gᵢ·r + iφᵢ)",
            "isosurface: |ψ|² = ½·iso",
            "lobes ← sign of Re ψ",
        ],
        notes: "A Gaussian envelope localises the inverse-FT into one orbital. field_mix translates the centre; lobes are coloured by sign(Re ψ) — classic chemistry-textbook two-tone.",
    },
    ModeInfo {
        idx: 16, name: "MAGNETIC",
        tagline: "Skyrmion-lattice spin texture with oriented streaks.",
        equations: &[
            "n̂(r) = (nx, ny, nz)/|n|,",
            "nx = f, ny = f₂, nz = ∂z f",
            "Bloch ↔ Néel: rot by atan2(w_motif, w_lat)",
        ],
        notes: "Real-space 3D spin field. Red ↑ → blue ↓ polar dome; streaks aligned with in-plane direction; bright cores at |nz|→1. Type controlled by w_motif/w_lattice ratio.",
    },
    ModeInfo {
        idx: 17, name: "DISPERSION",
        tagline: "Band-structure plot ε(k) along Γ–X–M–Γ.",
        equations: &[
            "ε(K) = Σᵢ aᵢ cos(Gᵢ·K + φᵢ·u)",
            "two bands: u = 0 and u = 1",
            "x ↦ path param,  y ↦ energy",
        ],
        notes: "Tight-binding-style cosine sum on a piecewise-linear k-path. Sweeping highlight scans the path; tick marks at Γ/X/M/Γ. iso_level thins the lines.",
    },
    ModeInfo {
        idx: 18, name: "KIKUCHI",
        tagline: "EBSD-style Kikuchi band pattern from G-vectors.",
        equations: &[
            "for each G: pair of lines",
            "    d_par(uv) = ±|G|·k_scale",
            "    smeared by  exp(−d²/w²)",
        ],
        notes: "Each G produces two parallel Kikuchi lines (excess/deficiency cone intersections with the detector). Crystal orientation maps onto pattern symmetry. iso_level sharpens.",
    },
    ModeInfo {
        idx: 19, name: "DEFECT",
        tagline: "Vacancy probe + Friedel oscillations.",
        equations: &[
            "f(r) = f₀ − A·e^(−r²/σ²)",
            "       + cos(2 k_F r)·e^(−r)/r",
            "k_F = kscale·(3 + 6·field_mix)",
        ],
        notes: "Drag with the mouse — the cursor is a vacancy. Lattice depresses at the core; Friedel rings ring outward at 2k_F. Real, if highly stylised, condensed-matter physics.",
    },
    ModeInfo {
        idx: 20, name: "BAND SURFACE",
        tagline: "Two-band energy landscape with avoided crossings.",
        equations: &[
            "ε±(k) = ½(εₐ+ε_b) ± ½√((εₐ−ε_b)²+Δ²)",
            "Δ = 0.055 + 0.22·field_mix + …",
            "contours every  spacing·iso",
        ],
        notes: "Two bands with a controlled gap Δ. Saddle points, avoided crossings, and Fermi-level contours all visible. Cool→warm divergent palette around ε=0.",
    },
    ModeInfo {
        idx: 21, name: "SPIN TEXTURE",
        tagline: "Reciprocal-space spin field with Rashba winding.",
        equations: &[
            "winding n = 1 + ⌊4·field_mix⌋",
            "θ(k) = n·atan2(kᵧ, kₓ) + 0.75 f(k)",
            "draw n̂(k) as oriented dashes",
        ],
        notes: "Each cell carries a director-dash aligned with the local in-plane spin. ±polarity coloured up/down (warm/cool). field_mix sets the topological winding number.",
    },
    ModeInfo {
        idx: 22, name: "BZ PATH",
        tagline: "Animated traversal of a high-symmetry k-path.",
        equations: &[
            "path: Γ → X → M → Γ  (k from g_tex)",
            "probe at s(t) = fract(t · 0.155)",
            "response = Σᵢ aᵢ cos(Gᵢ·k + φᵢ + t)",
        ],
        notes: "A moving probe walks the path; trailing afterimage, twin band plot beneath, comet+sparkle decoration. Educational + decorative.",
    },
    ModeInfo {
        idx: 23, name: "CDW",
        tagline: "Charge-density wave with domains, slips, and vortices.",
        equations: &[
            "ρ(r) ∝ Σⱼ cos(kⱼ·r + Φ(r))",
            "Φ = domain ± vortex(r,c,±1)",
            "    + slip lines + Friedel noise",
        ],
        notes: "Periodic stripes that lock onto reciprocal-lattice directions, with domain walls, phase slips, and ±1 vortex pairs. iso_level sharpens nodes; field_mix tilts the cross-stripe weight.",
    },
    ModeInfo {
        idx: 24, name: "QUASICRYSTAL",
        tagline: "Five/tenfold reciprocal interference (Penrose-like).",
        equations: &[
            "10 plane waves at angles 2π·i/10",
            "amplitudes alternate 1 and 1/φ",
            "(φ = golden ratio)",
            "diffraction: spots at (k, φk, k/φ)",
        ],
        notes: "Non-periodic but quasiperiodic — true tenfold symmetry forbidden in crystals. Inner field shows Penrose-tile edges; outer panel shows the famous decagonal diffraction stars.",
    },
    ModeInfo {
        idx: 25, name: "THERMAL",
        tagline: "Debye–Waller blurred lattice peaks + heat haze.",
        equations: &[
            "I(G) ∝ |F(G)|² · e^(−⟨u²⟩|G|²)",
            "⟨u²⟩ ≈ T · 0.065,  T = iso_level",
            "peak width grows with T",
        ],
        notes: "Increases iso_level → hotter sample → peaks broaden and dim per Debye–Waller, plus a vector noise haze. Cool→warm tint follows T.",
    },
    ModeInfo {
        idx: 26, name: "DOMAIN WALL",
        tagline: "Ferroic domains separated by moving phase walls.",
        equations: &[
            "wall(r,t) = sin(p·k₁+t) + ½ sin(p·k₂−t)",
            "domain = step(wall, 0)",
            "wall mask = smoothstep(0, w, |wall|)",
        ],
        notes: "Two-state order parameter, walls advect across the field. Inside-domain stripes track the local symmetry; bright walls flash on transit. iso_level thickens the walls.",
    },
    ModeInfo {
        idx: 27, name: "FRACTURE",
        tagline: "Strained lattice around an advancing crack tip.",
        equations: &[
            "strain  u ∝ (1/√r)·(cos θ/2, −sin θ/2)",
            "    (mode-I K-field, leading order)",
            "lattice = f(p + u(p − r_tip))",
        ],
        notes: "Classical linear-elastic K-field around a crack: 1/√r stress singularity, mode-I displacement angles. field_mix sweeps the tip position. Crack opens behind the tip; lattice stretches ahead.",
    },
    ModeInfo {
        idx: 28, name: "BERRY",
        tagline: "Berry curvature Ω(k) heatmap from a two-band model.",
        equations: &[
            "h(k) = (f, f₂, m + 0.3·f(1.5k))",
            "d̂ = h/|h|,    m = 2·field_mix − 1",
            "Ω(k) = ½ d̂ · (∂ₓd̂ × ∂_y d̂)",
        ],
        notes: "Diverging palette (blue/orange) around Ω=0. Dirac points light up where |h|→0; m = field_mix − ½ tunes the topological transition. Streamlines trace the in-plane (h₁,h₂) flow.",
    },
    ModeInfo {
        idx: 29, name: "HOFSTADTER",
        tagline: "Self-similar butterfly via continued-fraction recursion.",
        equations: &[
            "Harper-like: ε ≈ 2cos(k) + 2cos(k+2π p n)",
            "p ← fract(1/p)   (continued fraction)",
            "summed over 7 levels,  amp ← 0.72^level",
        ],
        notes: "Not a real Harper solver — a fractal-shaped proxy. x ↦ φ (flux per plaquette), y ↦ energy. Vertical lines mark φ = ½, ⅓, ¼, ⅕. field_mix re-scales the energy axis.",
    },
    ModeInfo {
        idx: 30, name: "STM",
        tagline: "Top-down STM topograph with QPI Friedel rings.",
        equations: &[
            "ρ(r) = |ψ|² + Σᵢ cos(2k_F|r−rᵢ|+φᵢ)·e^(−|r−rᵢ|)/|r−rᵢ|",
            "z_tip ∝ ρ^0.6  (constant-current)",
        ],
        notes: "Five fixed impurities scatter quasiparticles into Friedel-ring QPI patterns. k_F = kscale·(3 + 4·field_mix); iso_level boosts QPI amplitude. Lambertian-shaded height map.",
    },
    ModeInfo {
        idx: 31, name: "SPECTRAL",
        tagline: "ARPES-style A(ω, k) Lorentzian-broadened bands.",
        equations: &[
            "A(ω, k) = (1/π) Σ_b  Σ_b / ((ω−ε_b)² + Σ_b²)",
            "Σ(ω) = 0.04 + 0.30 ω²   (Fermi-liquid)",
            "Σ ← Σ · (1 − 0.7·iso_level)",
        ],
        notes: "Three poles per k-point with weights (1.0, 0.7, 0.4). ω² scattering rate ⇒ sharper quasiparticles near E_F. iso_level → coherent quasiparticles. Navy→blue→warm-white ARPES palette.",
    },
    ModeInfo {
        idx: 32, name: "VORTEX KNOT",
        tagline: "Luminous tube around the nodal line of a complex ψ(r).",
        equations: &[
            "ψ(r) = Σᵢ aᵢ e^(i Gᵢ·r + iφᵢ + i t·i)",
            "nodal set: ψ(r) = 0  (codim 2 in 3D)",
            "SDF: |ψ(r)| − R_tube",
        ],
        notes: "Zeros of a complex scalar in 3D form 1D lines — knotted in general. Hue around the tube ← arg(ψ); volumetric halo follows |ψ|²→0. iso_level thickens the tube.",
    },
    ModeInfo {
        idx: 33, name: "BLOCH WAVE",
        tagline: "Semiclassical wavepacket undergoing Bloch oscillations.",
        equations: &[
            "ε(k) = −2 cos(k)",
            "x_c(t) = sin(F·t) / F",
            "T_B = 2π / F   (period)",
            "F = 0.4 + 1.2·field_mix",
        ],
        notes: "Space (x) along the horizontal, time downward — the packet oscillates instead of drifting because of the lattice. Gaussian envelope with phase fringes. field_mix sets the DC field, hence the period.",
    },
    ModeInfo {
        idx: 34, name: "PLASMON",
        tagline: "Lindhard particle–hole continuum + plasmon line.",
        equations: &[
            "continuum: q² − q ≤ ω ≤ q + q²",
            "ω_p(q) = √(ω_p0² + α q²)",
            "Lorentzian: γ/((ω−ω_p)² + γ²)",
        ],
        notes: "(q, ω) phase diagram. The bright yellow line is the collective plasmon; the cloud is single-particle excitations (Landau damping kicks in where they overlap). field_mix → ω_p0; iso_level → α dispersion.",
    },
    ModeInfo {
        idx: 35, name: "NEMATIC",
        tagline: "Liquid-crystal director field with ±½ disclinations.",
        equations: &[
            "2θ = atan2(f₂, f)",
            "θ ∈ (−π/2, π/2]   (n̂ ≡ −n̂)",
            "defect charge = sign det J",
        ],
        notes: "Working with 2θ keeps the head–tail symmetry of a director. Cores at |order|→0 light up magenta; sign of the analytic Jacobian distinguishes +½ (warm) vs −½ (cool) defects.",
    },
    ModeInfo {
        idx: 36, name: "ABRIKOSOV",
        tagline: "Type-II superconductor vortex lattice (crystal-derived).",
        equations: &[
            "|ψ|²(r) = 1 − Σ_K f(K) cos(K·r)",
            "a₀ ∝ 1/√B,   ξ ∝ 1/√(1−t)",
            "λ ∝ 1/√(1−t⁴),   t = T/Tc",
            "CdGM star ∝ cos(6·(θ − θ_axis))",
        ],
        notes: "Lattice SYMMETRY inherits from the loaded crystal (cubic→square, hex→triangular). iso_level → reduced temperature; field_mix → applied B; w_lattice → CdGM 6-fold-star strength; w_motif → phase-winding rainbow ring; w_band → material-fingerprint modulation.",
    },
    ModeInfo {
        idx: 37, name: "CRYSTAL DIVE",
        tagline: "Endless self-similar zoom via the lattice point group.",
        equations: &[
            "zoom_log = t · 0.16,",
            "phase = fract(zoom_log)",
            "view_A = sample(uv, 2^⌊zoom_log⌋,  φ_A)",
            "view_B = sample(uv, 2·view_A,  φ_A + 2π/n)",
            "n ∈ {2, 4, 6}  from |cos∠(G₁,G₂)|",
        ],
        notes: "Crystal periodicity + point-group rotation = seamless infinite zoom. Cubic → 4-fold, hex → 6-fold, oblique → 2-fold. w_motif overlays a counter-spiral; w_lattice tunnels the rim; w_band sparkles the centre.",
    },
    ModeInfo {
        idx: 38, name: "NANO PHASE",
        tagline: "Size-dependent melting of a nanocrystal with twin-plane coexistence.",
        equations: &[
            "Tm(R) = Tm_bulk · (1 − 2σ_sl / (ρ L R))",
            "δ(T)  ∝ smoothstep(Tm − ΔT, Tm, T)   (liquid skin)",
            "core: |ψ|² = Σᵢ aᵢ cos(Gᵢ·r + φᵢ),  twin: FCC ⇌ BCT",
            "Debye–Waller:  ⟨I⟩ ∝ exp(−θ² · B_T)",
        ],
        notes: "iso_level is T/Tm_bulk — the particle melts from the outside in via a premelted surface shell, and smaller kscale (smaller R) drops Tm so tiny particles liquefy first. field_mix biases the sweeping martensitic twin boundary inside the solid core (austenite vs BCT-strained martensite); w_motif sharpens the habit plane, w_band sets liquid-shell turbulence, w_lattice the atomic-row contrast. Drag with the mouse to reposition the particle; near T → Tm watch the surface wobble (capillary waves) and shed vapour drops.",
    },
    ModeInfo {
        idx: 39, name: "MAGNON",
        tagline: "Spin-wave dispersion vs. Stoner particle–hole continuum.",
        equations: &[
            "ω_FM(q) = √(Δ² + (D q²)²)",
            "ω_AFM(q) = √(Δ² + (c q)²)",
            "Stoner: ω±(q) = D_S q² + Δ_ex ± v_F q",
            "Δ = 0.05 + 0.7·iso_level",
        ],
        notes: "iso_level sets the anisotropy gap Δ at q=0; field_mix crossfades FM (small-q quadratic) into AFM (linear); kscale stiffens the dispersion / Fermi velocity. w_motif dials the violet Stoner particle–hole cloud; w_band controls Landau damping (line broadens inside the continuum and above 2Δ where magnon decay opens). color_shift hue-rotates the magnon line; crystal_color tints it.",
    },
    ModeInfo {
        idx: 40, name: "WAVEPKT",
        tagline: "Free Gaussian wave packets: spreading, drifting, and interfering.",
        equations: &[
            "ψ(x,t) = (2π(σ₀² + iℏt/2m))^(−½) · exp[ik₀·(x−x₀) − iℏ|k₀|²t/2m]",
            "          · exp[−(x − x₀ − v_g t)² / (4(σ₀² + iℏt/2m))]",
            "σ(t) = σ₀ √(1 + (ℏt / 2mσ₀²)²)",
            "iℏ ∂ψ/∂t = −(ℏ²/2m) ∇²ψ,    v_g = ℏk₀/m",
        ],
        notes: "field_mix controls mean momentum |k₀| (slow drift → fast streaks); iso_level sets the initial width σ₀ — narrow packets spread visibly within seconds, the textbook minimum-uncertainty quench. color_shift adds a global phase and rotates k₀'s propagation axis; zoom rescales the visible window in ψ-units. Watch the bright Re(ψ)² fringe comb during the head-on collision (period ≈ 14 s) — those stripes are genuine quantum interference, not a texture; hold the mouse to relocate the launch points and throw packets at each other.",
    },
    ModeInfo {
        idx: 41, name: "DIPOLE",
        tagline: "Hertzian dipole radiation: near-field loops + far-field wavefronts with retarded phase.",
        equations: &[
            "E_θ ∝ sinθ · ω²/(c²r) · cos(ω(t−r/c))         (far,  1/r)",
            "      + ω/(cr²) · sin(ω(t−r/c))                (ind., 1/r²)",
            "      − 1/r³ · cos(ω(t−r/c))                   (quasi-stat, 1/r³)",
            "S = (1/μ₀c) · |E|² ∝ sin²θ / r²",
        ],
        notes: "field_mix sets the radiation angular frequency ω (controls wavelength on screen). iso_level fades in the near-field 1/r² and 1/r³ terms — at 0 you see clean far-field wavefronts, at 1 the swirling quasi-static loops near the source become visible. Watch the sin²θ donut envelope around r ≈ 2λ and the outgoing spherical phase fronts; the green arrow at origin is the instantaneous dipole moment p(t) = p₀ cos(ωt) ẑ.",
    },
    ModeInfo {
        idx: 42, name: "KARMAN",
        tagline: "Counter-rotating vortices shed from a bluff body in the Kármán wake regime.",
        equations: &[
            "ω(r,t) = Γ / (π r_c²) · exp(−r² / r_c²)",
            "r_c²(t) = r_c0² + 4 ν t     (Lamb–Oseen viscous core growth)",
            "f_s = St · U∞ / D,   St ≈ 0.21    (Strouhal shedding)",
        ],
        notes: "iso_level sets Reynolds number (60..300) — higher Re yields tighter cores and faster shedding; field_mix grows the initial vortex core radius (low Re → fat fuzzy vortices). speed acts as freestream U∞ (vortex drift rate downstream), zoom rescales the wake. Watch the staggered ±Γ pattern advect off the cylinder, the boundary-layer rim glow on its leading edge, and the streamline ribbons sweep around each core.",
    },
    ModeInfo {
        idx: 43, name: "LORENZ",
        tagline: "Lorenz attractor with per-pixel RK4 + Lyapunov coloring.",
        equations: &[
            "ẋ = σ(y − x)",
            "ẏ = x(ρ − z) − y,    ż = xy − βz",
            "λ = lim (1/T) · ln ‖δ(T)‖ / ‖δ(0)‖",
        ],
        notes: "field_mix sweeps ρ across the Hopf bifurcation (~24.74) — low values give regular spirals, high values give the full butterfly. iso_level controls the RK4 step length (shorter = finer trails, longer = wilder). zoom sets the phase-space window; hold the mouse for a probe spot. Watch the wings: bright regions = high local Lyapunov λ, sensitive to initial conditions; y₀ is perturbed by crystal_field so each loaded crystal stamps its signature onto the chaos.",
    },
    ModeInfo {
        idx: 44, name: "LENSING",
        tagline: "Schwarzschild black hole — shadow, photon ring, Doppler-beamed accretion disk.",
        equations: &[
            "ds² = −(1 − rₛ/r) c² dt² + (1 − rₛ/r)⁻¹ dr² + r² dΩ²",
            "α(b) = 4GM/(c² b),    b_c = (3√3/2) rₛ",
            "D = 1 / [γ(1 − β·n̂)],    I_obs = D⁴ · √(1 − rₛ/r) · I_em",
        ],
        notes: "field_mix raises the camera elevation (0 = edge-on disk, 1 = top-down). iso_level controls disk emissivity, color_shift rotates the disk hue, zoom is the camera FOV. Hold the mouse to orbit — sweep horizontally for azimuth, vertically for elevation. Watch the disk near-side pass in front of the black hole while the secondary image arcs over the top of the photon ring; the photon sphere sits at b = (3√3/2) rₛ.",
    },
    ModeInfo {
        idx: 45, name: "GRAV WAVE",
        tagline: "Binary-inspiral gravitational waves with interferometer fringes.",
        equations: &[
            "h_+(r,t) = A/r * cos(2(phi - theta) - omega r)",
            "h_x(r,t) = A/r * sin(2(phi - theta) - omega r)",
            "ds^2 = -dt^2 + (1+h_+)dx^2 + (1-h_+)dy^2 + 2h_x dxdy",
            "Delta L/L ~= h_+/2",
        ],
        notes: "A compact binary chirps from wide orbit to merger, sending quadrupole strain rings through a distorted test grid. The cross-shaped Michelson arms turn h+/hx into shifting interference fringes. field_mix raises chirp frequency, iso_level increases strain amplitude and ring sharpness, color_shift rotates the polarization basis, and zoom sets the detector window.",
    },
];

/// Static array of mode display names, derived from `MODES`. Kept as a `[&str;
/// N]` so call sites can use both indexing and `.len()` ergonomically.
pub const MODE_NAMES: [&str; MODES.len()] = {
    let mut names = [""; MODES.len()];
    let mut i = 0;
    while i < MODES.len() {
        names[i] = MODES[i].name;
        i += 1;
    }
    names
};

/// Slice-typed alias for crates that prefer `&[&str]` (e.g. `bench.rs`).
pub const MODE_NAMES_SLICE: &[&str] = &MODE_NAMES;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_are_indexed_in_order() {
        for (i, m) in MODES.iter().enumerate() {
            assert_eq!(m.idx, i, "MODES[{}] has idx {}", i, m.idx);
        }
    }

    #[test]
    fn names_match_modes() {
        for (i, m) in MODES.iter().enumerate() {
            assert_eq!(MODE_NAMES[i], m.name);
        }
    }
}
