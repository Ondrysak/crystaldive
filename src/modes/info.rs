//! Per-mode metadata: display name, tagline, equations (unicode-styled math),
//! and slider/usage notes. One source of truth for the visualizer modes.
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

// ── Per-mode parameter bank metadata ─────────────────────────────────────
//
// Each mode declares which of the 16 generic `mp` slots it exposes, with a
// human label, range, and default. The egui panel renders one named slider per
// declared `ParamDesc` of the active mode (instead of a fixed global list), and
// `slot_range` feeds the LFO/mic modulator the right span per slot. Slot indices
// mirror the WGSL `MP_*` consts in `prelude.wgsl`.

/// One slider in a mode's tailored parameter panel.
#[derive(Clone, Copy, Debug)]
pub struct ParamDesc {
    pub slot:    usize,
    pub name:    &'static str,
    pub min:     f32,
    pub max:     f32,
    pub default: f32,
}

impl ParamDesc {
    pub const fn new(slot: usize, name: &'static str, min: f32, max: f32, default: f32) -> Self {
        Self { slot, name, min, max, default }
    }
}

// Canonical crystal-field generator slots (0..8). Modes that drive
// `crystal_field`/`sdf` reuse these so the generator keeps working; they may be
// relabelled per mode but must stay on their slot. Ranges/defaults match the
// pre-bank global params exactly (behaviour-preserving).
pub const P_KSCALE:      ParamDesc = ParamDesc::new(0, "kscale",    0.1, 5.0, 1.4);
pub const P_SPEED:       ParamDesc = ParamDesc::new(1, "speed",     0.0, 2.0, 0.3);
pub const P_FIELD_MIX:   ParamDesc = ParamDesc::new(2, "field_mix", 0.0, 1.0, 0.55);
pub const P_ISO_LEVEL:   ParamDesc = ParamDesc::new(3, "iso_level", 0.0, 1.0, 0.5);
pub const P_COLOR_SHIFT: ParamDesc = ParamDesc::new(4, "color_sft", 0.0, 1.0, 0.0);
pub const P_ZOOM:        ParamDesc = ParamDesc::new(5, "zoom",      0.2, 5.0, 1.0);
pub const P_W_LATTICE:   ParamDesc = ParamDesc::new(6, "w_lattice", 0.0, 2.0, 1.0);
pub const P_W_MOTIF:     ParamDesc = ParamDesc::new(7, "w_motif",   0.0, 2.0, 0.6);
pub const P_W_BAND:      ParamDesc = ParamDesc::new(8, "w_band",    0.0, 2.0, 0.4);

/// The full canonical 9-slot set — the default panel for any mode that hasn't
/// been given a tailored list yet.
pub const CANONICAL: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_FIELD_MIX, P_ISO_LEVEL, P_COLOR_SHIFT,
    P_ZOOM, P_W_LATTICE, P_W_MOTIF, P_W_BAND,
];

/// Tailored parameter panel for `mode`. Modes with a bespoke list (see the
/// `match`) get mode-specific labels/ranges and extra free slots (9..15);
/// everything else falls back to the canonical 9.
// 28 LORENZ — chaotic attractor; sigma/rho/beta/dt are now direct knobs
// (free slots 9..12) instead of hardcoded constants. window = zoom slot.
const LORENZ_PARAMS: &[ParamDesc] = &[
    P_SPEED,
    ParamDesc::new(5,  "window", 0.2,  5.0,  1.0),
    P_COLOR_SHIFT,
    ParamDesc::new(9,  "sigma",  4.0,  18.0, 10.0),
    ParamDesc::new(10, "rho",    14.0, 40.0, 28.0),
    ParamDesc::new(11, "beta",   1.0,  5.0,  2.6667),
    ParamDesc::new(12, "dt",     0.002, 0.012, 0.006),
];

// Per-mode panels. Slots a mode samples through the crystal-field generator
// (kscale / speed / w_lattice / w_motif / w_band — and field_mix / iso_level
// via `sdf`) keep their canonical names since they genuinely shape the field;
// each mode's *headline* control is relabelled to what it actually does (drawn
// from the mode's `notes`). Defaults match canonical so a mode switch stays
// well-behaved. Physics modes 25..30 use bespoke ranges; LORENZ also free slots.

// Common relabel helpers.
const fn p(slot: usize, name: &'static str, def: f32) -> ParamDesc {
    // canonical-range relabel: pick the right range by slot family.
    match slot {
        0 => ParamDesc::new(0, name, 0.1, 5.0, def),  // kscale-like
        5 => ParamDesc::new(5, name, 0.2, 5.0, def),  // zoom-like
        6 | 7 | 8 => ParamDesc::new(slot, name, 0.0, 2.0, def), // weight-like
        _ => ParamDesc::new(slot, name, 0.0, 1.0, def), // unit sliders (1..4)
    }
}

// ── crystal-field core (0..9) ────────────────────────────────────────────
const ISO_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM,
    p(2, "cf2_mix", 0.55), p(3, "iso_lvl", 0.5),
    P_W_LATTICE, P_W_MOTIF, P_W_BAND,
];
const BZ_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, p(3, "contour", 0.5), p(4, "hue", 0.0),
    P_W_LATTICE, P_W_MOTIF, P_W_BAND,
];
const FERMI_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM, p(3, "energy", 0.5),
    P_W_LATTICE, P_W_MOTIF, P_W_BAND,
];
const DENSITY_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM, p(4, "hue", 0.0),
    P_W_LATTICE, P_W_MOTIF, P_W_BAND,
];
const NODAL_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM, P_W_LATTICE, P_W_MOTIF, P_W_BAND,
];
const PHASE_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM, p(4, "hue", 0.0),
    P_W_LATTICE, P_W_MOTIF, P_W_BAND,
];
const STRIPES_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM,
    p(3, "contours", 0.5), p(2, "tilt", 0.55), p(4, "hue", 0.0),
    P_W_LATTICE, P_W_MOTIF, P_W_BAND,
];
const WARP_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM,
    p(2, "bend", 0.55), p(8, "shimmer", 0.4), p(4, "hue", 0.0),
    P_W_LATTICE, P_W_MOTIF,
];
const NONEUC_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, p(3, "tiling", 0.5), p(4, "hue", 0.0),
];
const MOIRE_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM,
    p(2, "twist", 0.55), p(3, "sharpen", 0.5), p(4, "hue", 0.0),
    P_W_LATTICE, P_W_MOTIF, P_W_BAND,
];

// ── reciprocal / electronic / materials (10..24) ──────────────────────────
const WANNIER_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM,
    p(2, "orbital", 0.55), p(3, "envelope", 0.5),
    P_W_LATTICE, P_W_MOTIF, P_W_BAND,
];
const MAGNETIC_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM, p(3, "tilt", 0.5),
    p(6, "type_lat", 1.0), p(7, "type_mot", 0.6),
];
const KIKUCHI_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM, p(3, "sharpen", 0.5),
];
const DEFECT_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM, p(2, "depth", 0.55), p(3, "friedel", 0.5),
];
const BAND_SURFACE_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM,
    p(2, "gap", 0.55), p(3, "fermi", 0.5), p(4, "hue", 0.0),
    P_W_LATTICE, P_W_MOTIF, P_W_BAND,
];
const SPIN_TEXTURE_PARAMS: &[ParamDesc] = &[
    P_SPEED, P_ZOOM,
    p(2, "winding", 0.55), p(3, "scale", 0.5), p(4, "hue", 0.0),
    P_W_LATTICE, P_W_MOTIF, P_W_BAND,
];
const CDW_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM,
    p(2, "tilt", 0.55), p(3, "sharpen", 0.5), p(4, "hue", 0.0),
];
const QUASICRYSTAL_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM, p(2, "phason", 0.55), p(3, "edges", 0.5), p(4, "hue", 0.0),
];
const THERMAL_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM, p(3, "temp", 0.5), p(4, "hue", 0.0),
];
const DOMAIN_WALL_PARAMS: &[ParamDesc] = &[
    P_SPEED, P_ZOOM, p(2, "order", 0.55), p(3, "wall_w", 0.5), p(4, "hue", 0.0),
];
const BERRY_PARAMS: &[ParamDesc] = &[
    P_SPEED, P_ZOOM, p(2, "mass_m", 0.55), p(3, "scale", 0.5), p(4, "hue", 0.0),
];
const STM_PARAMS: &[ParamDesc] = &[
    p(0, "k_F", 1.4), P_ZOOM, p(2, "qpi_k", 0.55), p(3, "qpi_amp", 0.5), p(4, "hue", 0.0),
];
const VORTEX_KNOT_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM, p(2, "twist", 0.55), p(3, "tube_r", 0.5), p(4, "hue", 0.0),
];
const NEMATIC_PARAMS: &[ParamDesc] = &[
    P_SPEED, P_ZOOM, p(2, "winding", 0.55), p(3, "scale", 0.5),
];
const ABRIKOSOV_PARAMS: &[ParamDesc] = &[
    P_SPEED, P_ZOOM, p(3, "temp", 0.5), p(2, "B_field", 0.55),
    p(6, "CdGM", 1.0), p(7, "phase_ring", 0.6), p(8, "material", 0.4), p(4, "hue", 0.0),
];

// ── physics continuum / fields (25..30) ───────────────────────────────────
// 25 WAVEPACKET — Gaussian wavepacket dispersion.
const WAVEPACKET_PARAMS: &[ParamDesc] = &[
    P_SPEED, ParamDesc::new(5, "zoom", 0.2, 5.0, 1.0),
    ParamDesc::new(2, "k_mag", 0.0, 1.0, 0.55),
    ParamDesc::new(3, "spread", 0.0, 1.0, 0.5),
    ParamDesc::new(4, "k_angle", 0.0, 1.0, 0.0),
];
// 26 DIPOLE_RAD — oscillating dipole radiation.
const DIPOLE_PARAMS: &[ParamDesc] = &[
    P_SPEED, ParamDesc::new(5, "zoom", 0.2, 5.0, 1.0),
    ParamDesc::new(2, "omega", 0.0, 1.0, 0.55),
    ParamDesc::new(3, "near_fld", 0.0, 1.0, 0.5),
    ParamDesc::new(4, "hue", 0.0, 1.0, 0.0),
];
// 27 KARMAN — Kármán vortex street.
const KARMAN_PARAMS: &[ParamDesc] = &[
    ParamDesc::new(1, "U_inf", 0.0, 2.0, 0.3),
    ParamDesc::new(2, "core_rad", 0.0, 1.0, 0.55),
    ParamDesc::new(3, "Reynolds", 0.0, 1.0, 0.5),
    ParamDesc::new(5, "scale", 0.2, 5.0, 1.0),
    ParamDesc::new(4, "palette", 0.0, 1.0, 0.0),
];
// 29 LENSING — gravitationally lensed accretion disk.
const LENSING_PARAMS: &[ParamDesc] = &[
    ParamDesc::new(1, "orbit", 0.0, 2.0, 0.3),
    ParamDesc::new(2, "cam_elev", 0.0, 1.0, 0.55),
    ParamDesc::new(3, "disk_emis", 0.0, 1.0, 0.5),
    ParamDesc::new(5, "magnify", 0.2, 5.0, 1.0),
    ParamDesc::new(4, "disk_hue", 0.0, 1.0, 0.0),
];
// 30 GRAV_WAVE — inspiral strain pattern.
const GRAV_WAVE_PARAMS: &[ParamDesc] = &[
    P_SPEED, ParamDesc::new(5, "zoom", 0.2, 5.0, 1.0),
    ParamDesc::new(2, "freq", 0.0, 1.0, 0.55),
    ParamDesc::new(3, "amp", 0.0, 1.0, 0.5),
    ParamDesc::new(4, "polariz", 0.0, 1.0, 0.0),
];

// ── lattice walk + elastic / strain / spin (31..35) ───────────────────────
const LATTICE_WALK_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM,
    p(2, "morph", 0.55), p(3, "bore", 0.5), p(4, "hue", 0.0),
    P_W_LATTICE, P_W_MOTIF, P_W_BAND,
];
const ELASTIC_ANISO_PARAMS: &[ParamDesc] = &[
    P_SPEED, P_ZOOM, p(2, "amplitude", 0.55), p(3, "isotropy", 0.5),
];
const ELASTIC_WAVE_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM, p(2, "qT_mode", 0.55), p(3, "anisotropy", 0.5), p(4, "hue", 0.0),
];
const STRAIN_FIELD_PARAMS: &[ParamDesc] = &[
    P_SPEED, P_ZOOM, p(2, "lock_strain", 0.55), p(3, "amount", 0.5), p(4, "hue", 0.0),
];
const SPIN_DENSITY_PARAMS: &[ParamDesc] = &[
    P_KSCALE, P_SPEED, P_ZOOM, p(2, "order", 0.55), p(3, "moment", 0.5), p(4, "hue", 0.0),
];

pub fn mode_params(mode: u32) -> &'static [ParamDesc] {
    match mode {
        0  => ISO_PARAMS,
        1  => BZ_PARAMS,
        2  => FERMI_PARAMS,
        3  => DENSITY_PARAMS,
        4  => NODAL_PARAMS,
        5  => PHASE_PARAMS,
        6  => STRIPES_PARAMS,
        7  => WARP_PARAMS,
        8  => NONEUC_PARAMS,
        9  => MOIRE_PARAMS,
        10 => WANNIER_PARAMS,
        11 => MAGNETIC_PARAMS,
        12 => KIKUCHI_PARAMS,
        13 => DEFECT_PARAMS,
        14 => BAND_SURFACE_PARAMS,
        15 => SPIN_TEXTURE_PARAMS,
        16 => CDW_PARAMS,
        17 => QUASICRYSTAL_PARAMS,
        18 => THERMAL_PARAMS,
        19 => DOMAIN_WALL_PARAMS,
        20 => BERRY_PARAMS,
        21 => STM_PARAMS,
        22 => VORTEX_KNOT_PARAMS,
        23 => NEMATIC_PARAMS,
        24 => ABRIKOSOV_PARAMS,
        25 => WAVEPACKET_PARAMS,
        26 => DIPOLE_PARAMS,
        27 => KARMAN_PARAMS,
        28 => LORENZ_PARAMS,
        29 => LENSING_PARAMS,
        30 => GRAV_WAVE_PARAMS,
        31 => LATTICE_WALK_PARAMS,
        32 => ELASTIC_ANISO_PARAMS,
        33 => ELASTIC_WAVE_PARAMS,
        34 => STRAIN_FIELD_PARAMS,
        35 => SPIN_DENSITY_PARAMS,
        _  => CANONICAL,
    }
}

/// Clamp range for `slot` under `mode`: the declared range if the mode exposes
/// that slot, else the canonical range for slots 0..8, else a wide default.
pub fn slot_range(mode: u32, slot: usize) -> (f32, f32) {
    if let Some(p) = mode_params(mode).iter().find(|p| p.slot == slot) {
        return (p.min, p.max);
    }
    if slot < CANONICAL.len() {
        let c = CANONICAL[slot];
        return (c.min, c.max);
    }
    (0.0, 5.0)
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
            0..=9 | 17 | 31 => ModeArea::Crystal,
            12 => ModeArea::Reciprocal,
            10 | 14 | 15 | 20 | 21 | 22 | 24 | 25 => ModeArea::Electronic,
            11 | 13 | 16 | 18 | 19 | 23 | 32 | 33 | 34 | 35 => ModeArea::Materials,
            27 | 28 => ModeArea::Classical,
            26 | 29 | 30 => ModeArea::Fields,
            _ => ModeArea::Crystal,
        }
    }
}

pub const MODES: [ModeInfo; 36] = [
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
        idx: 8, name: "NONEUC",
        tagline: "Kleinian circle-inversion kaleidoscope.",
        equations: &[
            "inv(z; c, r) = c + r²(z−c)/|z−c|²",
            "iterate 52× through 4 G-derived circles",
            "tint each step by f(z), decay e^(−i/20)",
        ],
        notes: "Hyperbolic (non-Euclidean) tiling — circles inverted into themselves. Centres come from the first four G-vectors so different materials give different fundamental groups.",
    },
    ModeInfo {
        idx: 9, name: "MOIRE",
        tagline: "Twisted-bilayer interference of two crystal fields.",
        equations: &[
            "θ = field_mix · 30°",
            "draw  f(p) · f(Rθ p)   (product)",
            "+ contour at  f + f_rot = 0",
        ],
        notes: "Two copies of the lattice, one rotated by θ. Magic-angle moiré patterns and pinwheel ripples appear at small θ. iso_level sharpens contours.",
    },
    ModeInfo {
        idx: 10, name: "WANNIER",
        tagline: "Localised Wannier orbital |ψ|² isosurface.",
        equations: &[
            "ψ(r) = e^(−0.6 r²) Σᵢ aᵢ e^(i Gᵢ·r + iφᵢ)",
            "isosurface: |ψ|² = ½·iso",
            "lobes ← sign of Re ψ",
        ],
        notes: "A Gaussian envelope localises the inverse-FT into one orbital. field_mix translates the centre; lobes are coloured by sign(Re ψ) — classic chemistry-textbook two-tone.",
    },
    ModeInfo {
        idx: 11, name: "MAGNETIC",
        tagline: "Skyrmion-lattice spin texture with oriented streaks.",
        equations: &[
            "n̂(r) = (nx, ny, nz)/|n|,",
            "nx = f, ny = f₂, nz = ∂z f",
            "Bloch ↔ Néel: rot by atan2(w_motif, w_lat)",
        ],
        notes: "Real-space 3D spin field. Red ↑ → blue ↓ polar dome; streaks aligned with in-plane direction; bright cores at |nz|→1. Type controlled by w_motif/w_lattice ratio.",
    },
    ModeInfo {
        idx: 12, name: "KIKUCHI",
        tagline: "EBSD-style Kikuchi band pattern from G-vectors.",
        equations: &[
            "for each G: pair of lines",
            "    d_par(uv) = ±|G|·k_scale",
            "    smeared by  exp(−d²/w²)",
        ],
        notes: "Each G produces two parallel Kikuchi lines (excess/deficiency cone intersections with the detector). Crystal orientation maps onto pattern symmetry. iso_level sharpens.",
    },
    ModeInfo {
        idx: 13, name: "DEFECT",
        tagline: "Vacancy probe + Friedel oscillations.",
        equations: &[
            "f(r) = f₀ − A·e^(−r²/σ²)",
            "       + cos(2 k_F r)·e^(−r)/r",
            "k_F = kscale·(3 + 6·field_mix)",
        ],
        notes: "Drag with the mouse — the cursor is a vacancy. Lattice depresses at the core; Friedel rings ring outward at 2k_F. Real, if highly stylised, condensed-matter physics.",
    },
    ModeInfo {
        idx: 14, name: "BAND SURFACE",
        tagline: "Two-band energy landscape with avoided crossings.",
        equations: &[
            "ε±(k) = ½(εₐ+ε_b) ± ½√((εₐ−ε_b)²+Δ²)",
            "Δ = 0.055 + 0.22·field_mix + …",
            "contours every  spacing·iso",
        ],
        notes: "Two bands with a controlled gap Δ. Saddle points, avoided crossings, and Fermi-level contours all visible. Cool→warm divergent palette around ε=0.",
    },
    ModeInfo {
        idx: 15, name: "SPIN TEXTURE",
        tagline: "Reciprocal-space spin field with Rashba winding.",
        equations: &[
            "winding n = 1 + ⌊4·field_mix⌋",
            "θ(k) = n·atan2(kᵧ, kₓ) + 0.75 f(k)",
            "draw n̂(k) as oriented dashes",
        ],
        notes: "Each cell carries a director-dash aligned with the local in-plane spin. ±polarity coloured up/down (warm/cool). field_mix sets the topological winding number.",
    },
    ModeInfo {
        idx: 16, name: "CDW",
        tagline: "Charge-density wave with domains, slips, and vortices.",
        equations: &[
            "ρ(r) ∝ Σⱼ cos(kⱼ·r + Φ(r))",
            "Φ = domain ± vortex(r,c,±1)",
            "    + slip lines + Friedel noise",
        ],
        notes: "Periodic stripes that lock onto reciprocal-lattice directions, with domain walls, phase slips, and ±1 vortex pairs. iso_level sharpens nodes; field_mix tilts the cross-stripe weight.",
    },
    ModeInfo {
        idx: 17, name: "QUASICRYSTAL",
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
        idx: 18, name: "THERMAL",
        tagline: "Debye–Waller blurred lattice peaks + heat haze.",
        equations: &[
            "I(G) ∝ |F(G)|² · e^(−⟨u²⟩|G|²)",
            "⟨u²⟩ ≈ T · 0.065,  T = iso_level",
            "peak width grows with T",
        ],
        notes: "Increases iso_level → hotter sample → peaks broaden and dim per Debye–Waller, plus a vector noise haze. Cool→warm tint follows T.",
    },
    ModeInfo {
        idx: 19, name: "DOMAIN WALL",
        tagline: "Ferroic domains separated by moving phase walls.",
        equations: &[
            "wall(r,t) = sin(p·k₁+t) + ½ sin(p·k₂−t)",
            "domain = step(wall, 0)",
            "wall mask = smoothstep(0, w, |wall|)",
        ],
        notes: "Two-state order parameter, walls advect across the field. Inside-domain stripes track the local symmetry; bright walls flash on transit. iso_level thickens the walls.",
    },
    ModeInfo {
        idx: 20, name: "BERRY",
        tagline: "Berry curvature Ω(k) heatmap from a two-band model.",
        equations: &[
            "h(k) = (f, f₂, m + 0.3·f(1.5k))",
            "d̂ = h/|h|,    m = 2·field_mix − 1",
            "Ω(k) = ½ d̂ · (∂ₓd̂ × ∂_y d̂)",
        ],
        notes: "Diverging palette (blue/orange) around Ω=0. Dirac points light up where |h|→0; m = field_mix − ½ tunes the topological transition. Streamlines trace the in-plane (h₁,h₂) flow.",
    },
    ModeInfo {
        idx: 21, name: "STM",
        tagline: "Top-down STM topograph with QPI Friedel rings.",
        equations: &[
            "ρ(r) = |ψ|² + Σᵢ cos(2k_F|r−rᵢ|+φᵢ)·e^(−|r−rᵢ|)/|r−rᵢ|",
            "z_tip ∝ ρ^0.6  (constant-current)",
        ],
        notes: "Five fixed impurities scatter quasiparticles into Friedel-ring QPI patterns. k_F = kscale·(3 + 4·field_mix); iso_level boosts QPI amplitude. Lambertian-shaded height map.",
    },
    ModeInfo {
        idx: 22, name: "VORTEX KNOT",
        tagline: "Luminous tube around the nodal line of a complex ψ(r).",
        equations: &[
            "ψ(r) = Σᵢ aᵢ e^(i Gᵢ·r + iφᵢ + i t·i)",
            "nodal set: ψ(r) = 0  (codim 2 in 3D)",
            "SDF: |ψ(r)| − R_tube",
        ],
        notes: "Zeros of a complex scalar in 3D form 1D lines — knotted in general. Hue around the tube ← arg(ψ); volumetric halo follows |ψ|²→0. iso_level thickens the tube.",
    },
    ModeInfo {
        idx: 23, name: "NEMATIC",
        tagline: "Liquid-crystal director field with ±½ disclinations.",
        equations: &[
            "2θ = atan2(f₂, f)",
            "θ ∈ (−π/2, π/2]   (n̂ ≡ −n̂)",
            "defect charge = sign det J",
        ],
        notes: "Working with 2θ keeps the head–tail symmetry of a director. Cores at |order|→0 light up magenta; sign of the analytic Jacobian distinguishes +½ (warm) vs −½ (cool) defects.",
    },
    ModeInfo {
        idx: 24, name: "ABRIKOSOV",
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
        idx: 25, name: "WAVEPKT",
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
        idx: 26, name: "DIPOLE",
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
        idx: 27, name: "KARMAN",
        tagline: "Counter-rotating vortices shed from a bluff body in the Kármán wake regime.",
        equations: &[
            "ω(r,t) = Γ / (π r_c²) · exp(−r² / r_c²)",
            "r_c²(t) = r_c0² + 4 ν t     (Lamb–Oseen viscous core growth)",
            "f_s = St · U∞ / D,   St ≈ 0.21    (Strouhal shedding)",
        ],
        notes: "iso_level sets Reynolds number (60..300) — higher Re yields tighter cores and faster shedding; field_mix grows the initial vortex core radius (low Re → fat fuzzy vortices). speed acts as freestream U∞ (vortex drift rate downstream), zoom rescales the wake. Watch the staggered ±Γ pattern advect off the cylinder, the boundary-layer rim glow on its leading edge, and the streamline ribbons sweep around each core.",
    },
    ModeInfo {
        idx: 28, name: "LORENZ",
        tagline: "Lorenz attractor with per-pixel RK4 + Lyapunov coloring.",
        equations: &[
            "ẋ = σ(y − x)",
            "ẏ = x(ρ − z) − y,    ż = xy − βz",
            "λ = lim (1/T) · ln ‖δ(T)‖ / ‖δ(0)‖",
        ],
        notes: "field_mix sweeps ρ across the Hopf bifurcation (~24.74) — low values give regular spirals, high values give the full butterfly. iso_level controls the RK4 step length (shorter = finer trails, longer = wilder). zoom sets the phase-space window; hold the mouse for a probe spot. Watch the wings: bright regions = high local Lyapunov λ, sensitive to initial conditions; y₀ is perturbed by crystal_field so each loaded crystal stamps its signature onto the chaos.",
    },
    ModeInfo {
        idx: 29, name: "LENSING",
        tagline: "Schwarzschild black hole — shadow, photon ring, Doppler-beamed accretion disk.",
        equations: &[
            "ds² = −(1 − rₛ/r) c² dt² + (1 − rₛ/r)⁻¹ dr² + r² dΩ²",
            "α(b) = 4GM/(c² b),    b_c = (3√3/2) rₛ",
            "D = 1 / [γ(1 − β·n̂)],    I_obs = D⁴ · √(1 − rₛ/r) · I_em",
        ],
        notes: "field_mix raises the camera elevation (0 = edge-on disk, 1 = top-down). iso_level controls disk emissivity, color_shift rotates the disk hue, zoom is the camera FOV. Hold the mouse to orbit — sweep horizontally for azimuth, vertically for elevation. Watch the disk near-side pass in front of the black hole while the secondary image arcs over the top of the photon ring; the photon sphere sits at b = (3√3/2) rₛ.",
    },
    ModeInfo {
        idx: 30, name: "GRAV WAVE",
        tagline: "Binary-inspiral gravitational waves with interferometer fringes.",
        equations: &[
            "h_+(r,t) = A/r * cos(2(phi - theta) - omega r)",
            "h_x(r,t) = A/r * sin(2(phi - theta) - omega r)",
            "ds^2 = -dt^2 + (1+h_+)dx^2 + (1-h_+)dy^2 + 2h_x dxdy",
            "Delta L/L ~= h_+/2",
        ],
        notes: "A compact binary chirps from wide orbit to merger, sending quadrupole strain rings through a distorted test grid. The cross-shaped Michelson arms turn h+/hx into shifting interference fringes. field_mix raises chirp frequency, iso_level increases strain amplitude and ring sharpness, color_shift rotates the polarization basis, and zoom sets the detector window.",
    },
    ModeInfo {
        idx: 31, name: "LATTICE WALK",
        tagline: "First-person endless flight through the crystal isosurface tunnel.",
        equations: &[
            "pos(t) = (A sin(ω₁t), B cos(ω₂t), v t)   (helix path)",
            "surface: |mix(f, f₂, blend)| = iso · 0.3",
            "colour ← hue(f(hit) · 0.45 + color_shift)",
        ],
        notes: "The camera flies forward through the periodic crystal field, which repeats infinitely — same tunnel geometry, endlessly. iso_level opens/closes the tunnel bore (low = wide corridors, high = tight tubes). field_mix blends between the two field twins for tunnel topology variety. zoom is FOV (higher = telephoto, narrower). Mouse steers the gaze direction while the camera still flies straight.",
    },
    ModeInfo {
        idx: 32, name: "ELASTIC ANISO",
        tagline: "Directional Young's modulus polar surface derived from crystal G-vectors.",
        equations: &[
            "C(n̂) = Σᵢ aᵢ² (n̂·Ĝᵢ)⁴   (4th-rank projection)",
            "surface: r(n̂) = C(n̂) · scale",
            "Zener ring: highlight where C(n̂) ≈ C_iso",
        ],
        notes: "Each loaded crystal produces a distinct anisotropy shape: cubic → near-sphere with cubic-symmetric lobes; hexagonal → prolate/oblate blob. Mouse orbits. field_mix scales the surface amplitude; iso_level highlights the isotropic contour (Zener A = 1 ring). kscale modulates velocity range.",
    },
    ModeInfo {
        idx: 33, name: "ELASTIC WAVE",
        tagline: "Anisotropic acoustic wavefronts from a repeating impulse — qL and qT branches.",
        equations: &[
            "v_L(θ) ∝ √(Σ aᵢ² (n̂·Ĝᵢ)²)",
            "v_T(θ) ∝ √(1 − Σ aᵢ² (n̂·Ĝᵢ)²)",
            "focus ∝ |∂v/∂θ|²   (phonon-focusing caustic)",
        ],
        notes: "Warm rings = quasi-longitudinal (qL); cool rings = quasi-transverse (qT, toggled by field_mix). Crystal anisotropy deforms the circular wavefront into a crinkled shape; high |∂v/∂θ| directions produce bright caustic cusps (phonon focusing). The inner inset diagram shows the slowness surface (1/v vs θ). iso_level tints sector directions; kscale stiffens both branches.",
    },
    ModeInfo {
        idx: 34, name: "STRAIN FIELD",
        tagline: "Crystal under biaxial strain — deformed lattice + piezoelectric polarization charge.",
        equations: &[
            "ε_xx = ε_yy = ε,  ε_zz = −2ν ε / (1−ν)   (Poisson, ν ≈ 0.28)",
            "r → (r_x(1+ε), r_y(1+ε), r_z(1+ε_zz))",
            "ρ_P ∝ ε · ∇² f(r_strained)",
        ],
        notes: "Left panel shows the unstrained reference crystal field; right shows the deformed version. The interface glows with the piezoelectric polarization charge density (div P). Strain cycles sinusoidally unless field_mix is raised to lock it at maximum. iso_level sets the strain amplitude. A gauge bar at the bottom tracks the instantaneous strain.",
    },
    ModeInfo {
        idx: 35, name: "SPIN DENSITY",
        tagline: "Real-space element-resolved magnetic moment map across a 2D supercell.",
        equations: &[
            "ρ_s(r) = Σ_i mᵢ · exp(−|r − Rᵢ|² / σ²)",
            "mᵢ = m₀·(1 + δ sin(t))   (spin precession)",
            "ordering: FM ↔ AFM ↔ ferrimagnetic  (field_mix)",
        ],
        notes: "Red blobs = spin-up moments; blue = spin-down; dark = nonmagnetic sites. field_mix crossfades between ferromagnetic (all red), antiferromagnetic (checkerboard), and ferrimagnetic (alternating magnitudes). iso_level scales the moment magnitude. Mouse pans the supercell. Sublattice assignment uses the parity of the lattice site index so every loaded crystal has a natural two-sublattice decomposition.",
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

    /// Every mode's declared params must fit the 16-slot bank, have sane ranges,
    /// and not declare the same slot twice.
    #[test]
    fn mode_params_are_well_formed() {
        for m in MODES.iter() {
            let ps = mode_params(m.idx as u32);
            let mut seen = [false; 16];
            for p in ps {
                assert!(p.slot < 16, "{}: slot {} >= 16", m.name, p.slot);
                assert!(p.min <= p.max, "{}: {} min>max", m.name, p.name);
                assert!(
                    p.default >= p.min && p.default <= p.max,
                    "{}: {} default {} out of [{}, {}]",
                    m.name, p.name, p.default, p.min, p.max
                );
                assert!(!seen[p.slot], "{}: slot {} declared twice", m.name, p.slot);
                seen[p.slot] = true;
            }
        }
    }
}
