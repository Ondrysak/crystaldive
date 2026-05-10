//! Built-in crystal library: 34 hand-coded structures (all 7 systems)
//! plus 78 structures auto-fetched from the Materials Project API (11 thematic groups).

pub mod mp;  // auto-generated: src/crystals_mp.rs → mod mp

use crate::poscar::{frac_to_cart, Crystal, Atom};
use crate::symmetry::CrystalSystem;

#[derive(Clone)]
pub struct CrystalDef {
    pub name:        &'static str,
    pub space_group: &'static str,
    pub sp_number:   u16,
    pub system:      CrystalSystem,
    pub lattice_type:&'static str,  // P, I, F, R, A, C
    pub point_group: &'static str,
    pub color:       [f32; 3],
    /// Lattice constant a in Å (used to scale the dimensionless lattice matrix below)
    a: f32,
    /// Dimensionless 3×3 row matrix (multiply by `a` to get Å)
    lat_raw: [[f32; 3]; 3],
    /// Atom fractional coordinates
    fracs: &'static [[f32; 3]],
    /// Atomic numbers matching fracs
    z_vals: &'static [u8],
    /// Element symbols matching fracs
    symbols: &'static [&'static str],
}

impl CrystalDef {
    /// Build a Crystal (with Cartesian atom positions in Å) from this definition.
    pub fn to_crystal(&self) -> Crystal {
        let lat = [
            [self.lat_raw[0][0]*self.a, self.lat_raw[0][1]*self.a, self.lat_raw[0][2]*self.a],
            [self.lat_raw[1][0]*self.a, self.lat_raw[1][1]*self.a, self.lat_raw[1][2]*self.a],
            [self.lat_raw[2][0]*self.a, self.lat_raw[2][1]*self.a, self.lat_raw[2][2]*self.a],
        ];
        let atoms = self.fracs.iter().zip(self.symbols.iter()).map(|(f, s)| {
            Atom { species: s.to_string(), pos_cart: frac_to_cart(*f, &lat) }
        }).collect();
        Crystal { lattice: lat, atoms }
    }
}

pub struct CrystalGroup {
    pub label:    &'static str,
    pub color:    egui::Color32,
    pub crystals: &'static [CrystalDef],
}

// ── helpers to make table entries compact ────────────────────────────────

macro_rules! fracs {
    ($( [$a:expr,$b:expr,$c:expr] ),* $(,)?) => {
        &[ $( [$a as f32, $b as f32, $c as f32] ),* ]
    }
}
macro_rules! zvals { ($($z:expr),* $(,)?) => { &[$($z as u8),*] } }
macro_rules! syms  { ($($s:literal),* $(,)?) => { &[$($s),*] } }
// Make macros available to the mp submodule
pub(crate) use fracs;
pub(crate) use zvals;
pub(crate) use syms;

// ── CUBIC ─────────────────────────────────────────────────────────────────

static CUBIC: &[CrystalDef] = &[
    CrystalDef {
        name:"BCC Iron", space_group:"Im-3m (#229)", sp_number:229,
        system:CrystalSystem::Cubic, lattice_type:"I", point_group:"m-3m",
        color:[1.0,0.55,0.18], a:2.87,
        lat_raw:[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        fracs: fracs![[0,0,0],[0.5,0.5,0.5]],
        z_vals: zvals![26,26], symbols: syms!["Fe","Fe"],
    },
    CrystalDef {
        name:"FCC Gold", space_group:"Fm-3m (#225)", sp_number:225,
        system:CrystalSystem::Cubic, lattice_type:"F", point_group:"m-3m",
        color:[1.0,0.85,0.1], a:4.08,
        lat_raw:[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        fracs: fracs![[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]],
        z_vals: zvals![79,79,79,79], symbols: syms!["Au","Au","Au","Au"],
    },
    CrystalDef {
        name:"Diamond", space_group:"Fd-3m (#227)", sp_number:227,
        system:CrystalSystem::Cubic, lattice_type:"F", point_group:"m-3m",
        color:[0.55,0.92,1.0], a:3.567,
        lat_raw:[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        fracs: fracs![[0,0,0],[0.25,0.25,0.25],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0],[0.25,0.75,0.75],[0.75,0.25,0.75],[0.75,0.75,0.25]],
        z_vals: zvals![6,6,6,6,6,6,6,6], symbols: syms!["C","C","C","C","C","C","C","C"],
    },
    CrystalDef {
        name:"NaCl Rock salt", space_group:"Fm-3m (#225)", sp_number:225,
        system:CrystalSystem::Cubic, lattice_type:"F", point_group:"m-3m",
        color:[0.4,0.7,1.0], a:5.64,
        lat_raw:[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        fracs: fracs![[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.5,0,0],[0,0.5,0],[0,0,0.5],[0.5,0.5,0.5]],
        z_vals: zvals![11,11,11,11,17,17,17,17], symbols: syms!["Na","Na","Na","Na","Cl","Cl","Cl","Cl"],
    },
    CrystalDef {
        name:"Perovskite BaTiO3", space_group:"Pm-3m (#221)", sp_number:221,
        system:CrystalSystem::Cubic, lattice_type:"P", point_group:"m-3m",
        color:[1.0,0.3,0.5], a:4.01,
        lat_raw:[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        fracs: fracs![[0,0,0],[0.5,0.5,0.5],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]],
        z_vals: zvals![56,22,8,8,8], symbols: syms!["Ba","Ti","O","O","O"],
    },
    CrystalDef {
        name:"Spinel MgAl2O4", space_group:"Fd-3m (#227)", sp_number:227,
        system:CrystalSystem::Cubic, lattice_type:"F", point_group:"m-3m",
        color:[0.9,0.5,0.2], a:8.08,
        lat_raw:[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        fracs: fracs![[0,0,0],[0.625,0.625,0.625],[0.375,0.375,0.375],[0.25,0.25,0.25],[0.5,0.5,0.5]],
        z_vals: zvals![12,12,13,13,8], symbols: syms!["Mg","Mg","Al","Al","O"],
    },
    CrystalDef {
        name:"Pyrite FeS2", space_group:"Pa-3 (#205)", sp_number:205,
        system:CrystalSystem::Cubic, lattice_type:"P", point_group:"m-3",
        color:[1.0,0.7,0.1], a:5.42,
        lat_raw:[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        fracs: fracs![[0,0,0],[0.5,0,0.5],[0,0.5,0.5],[0.5,0.5,0],[0.385,0.385,0.385],[0.615,0.615,0.615],[0.115,0.885,0.615],[0.885,0.115,0.385]],
        z_vals: zvals![26,26,26,26,16,16,16,16], symbols: syms!["Fe","Fe","Fe","Fe","S","S","S","S"],
    },
    CrystalDef {
        name:"Zincblende GaAs", space_group:"F-43m (#216)", sp_number:216,
        system:CrystalSystem::Cubic, lattice_type:"F", point_group:"-43m",
        color:[0.7,0.4,1.0], a:5.65,
        lat_raw:[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        fracs: fracs![[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.25,0.25,0.25],[0.75,0.75,0.25],[0.75,0.25,0.75],[0.25,0.75,0.75]],
        z_vals: zvals![31,31,31,31,33,33,33,33], symbols: syms!["Ga","Ga","Ga","Ga","As","As","As","As"],
    },
];

// ── HEXAGONAL ─────────────────────────────────────────────────────────────

static HEXAGONAL: &[CrystalDef] = &[
    CrystalDef {
        name:"Graphite", space_group:"P63/mmc (#194)", sp_number:194,
        system:CrystalSystem::Hexagonal, lattice_type:"P", point_group:"6/mmm",
        color:[0.3,0.9,0.7], a:2.46,
        lat_raw:[[1.,0.,0.],[-0.5,0.866,0.],[0.,0.,2.726]],
        fracs: fracs![[0,0,0],[0.333,0.667,0],[0,0,0.5],[0.667,0.333,0.5]],
        z_vals: zvals![6,6,6,6], symbols: syms!["C","C","C","C"],
    },
    CrystalDef {
        name:"beta-Quartz SiO2", space_group:"P6222 (#180)", sp_number:180,
        system:CrystalSystem::Hexagonal, lattice_type:"P", point_group:"622",
        color:[0.6,0.8,0.4], a:4.91,
        lat_raw:[[1.,0.,0.],[-0.5,0.866,0.],[0.,0.,1.086]],
        fracs: fracs![[0.5,0,0],[0,0.5,0.333],[0.5,0.5,0.667],[0.415,0.27,0.214],[0.27,0.415,0.547],[0.585,0.73,0.881]],
        z_vals: zvals![14,14,14,8,8,8], symbols: syms!["Si","Si","Si","O","O","O"],
    },
    CrystalDef {
        name:"HCP Magnesium", space_group:"P63/mmc (#194)", sp_number:194,
        system:CrystalSystem::Hexagonal, lattice_type:"P", point_group:"6/mmm",
        color:[0.7,1.0,0.5], a:3.21,
        lat_raw:[[1.,0.,0.],[-0.5,0.866,0.],[0.,0.,1.624]],
        fracs: fracs![[0,0,0],[0.333,0.667,0.5]],
        z_vals: zvals![12,12], symbols: syms!["Mg","Mg"],
    },
    CrystalDef {
        name:"Wurtzite ZnS", space_group:"P63mc (#186)", sp_number:186,
        system:CrystalSystem::Hexagonal, lattice_type:"P", point_group:"6mm",
        color:[1.0,0.9,0.3], a:3.82,
        lat_raw:[[1.,0.,0.],[-0.5,0.866,0.],[0.,0.,1.636]],
        fracs: fracs![[0.333,0.667,0],[0.667,0.333,0.5],[0.333,0.667,0.375],[0.667,0.333,0.875]],
        z_vals: zvals![30,30,16,16], symbols: syms!["Zn","Zn","S","S"],
    },
    CrystalDef {
        name:"Corundum Al2O3", space_group:"R-3c (#167)", sp_number:167,
        system:CrystalSystem::Hexagonal, lattice_type:"R", point_group:"-3m",
        color:[0.9,0.6,0.9], a:4.76,
        lat_raw:[[1.,0.,0.],[-0.5,0.866,0.],[0.,0.,2.73]],
        fracs: fracs![[0,0,0.352],[0,0,0.148],[0.306,0,0.25],[0,0.306,0.25]],
        z_vals: zvals![13,13,8,8], symbols: syms!["Al","Al","O","O"],
    },
    CrystalDef {
        name:"Beryl Be3Al2Si6O18", space_group:"P6/mcc (#192)", sp_number:192,
        system:CrystalSystem::Hexagonal, lattice_type:"P", point_group:"6/mmm",
        color:[0.4,0.95,0.8], a:9.21,
        lat_raw:[[1.,0.,0.],[-0.5,0.866,0.],[0.,0.,0.968]],
        fracs: fracs![[0.5,0,0.25],[0,0.5,0.25],[0.667,0.333,0.25],[0.333,0.667,0.25],[0.386,0.077,0.25],[0.077,0.309,0.25]],
        z_vals: zvals![4,4,13,13,14,14], symbols: syms!["Be","Be","Al","Al","Si","Si"],
    },
];

// ── TRIGONAL ──────────────────────────────────────────────────────────────

static TRIGONAL: &[CrystalDef] = &[
    CrystalDef {
        name:"alpha-Quartz SiO2", space_group:"P3121 (#152)", sp_number:152,
        system:CrystalSystem::Trigonal, lattice_type:"P", point_group:"32",
        color:[0.95,0.85,0.4], a:4.913,
        lat_raw:[[1.,0.,0.],[-0.5,0.866,0.],[0.,0.,1.101]],
        fracs: fracs![[0.47,0,0],[0,0.47,0.667],[0.53,0.53,0.333],[0.415,0.272,0.12],[0.272,0.143,0.453],[0.857,0.585,0.787]],
        z_vals: zvals![14,14,14,8,8,8], symbols: syms!["Si","Si","Si","O","O","O"],
    },
    CrystalDef {
        name:"Calcite CaCO3", space_group:"R-3c (#167)", sp_number:167,
        system:CrystalSystem::Trigonal, lattice_type:"R", point_group:"-3m",
        color:[0.8,0.9,1.0], a:4.99,
        lat_raw:[[1.,0.,0.],[-0.5,0.866,0.],[0.,0.,1.701]],
        fracs: fracs![[0,0,0],[0,0,0.5],[0.257,0,0.25],[0.743,0,0.25],[0,0.257,0.25]],
        z_vals: zvals![20,20,6,6,8], symbols: syms!["Ca","Ca","C","C","O"],
    },
    CrystalDef {
        name:"Bismuth", space_group:"R-3m (#166)", sp_number:166,
        system:CrystalSystem::Trigonal, lattice_type:"R", point_group:"-3m",
        color:[0.6,0.5,1.0], a:4.546,
        lat_raw:[[1.,0.,0.],[-0.5,0.866,0.],[0.,0.,2.603]],
        fracs: fracs![[0,0,0.234],[0,0,0.766]],
        z_vals: zvals![83,83], symbols: syms!["Bi","Bi"],
    },
    CrystalDef {
        name:"Tourmaline", space_group:"R3m (#160)", sp_number:160,
        system:CrystalSystem::Trigonal, lattice_type:"R", point_group:"3m",
        color:[0.4,0.7,0.9], a:15.98,
        lat_raw:[[1.,0.,0.],[-0.5,0.866,0.],[0.,0.,0.448]],
        fracs: fracs![[0,0,0],[0.333,0,0],[0.167,0,0.5],[0.267,0.133,0.25],[0.1,0.267,0.75]],
        z_vals: zvals![11,13,14,5,8], symbols: syms!["Na","Al","Si","B","O"],
    },
];

// ── TETRAGONAL ────────────────────────────────────────────────────────────

static TETRAGONAL: &[CrystalDef] = &[
    CrystalDef {
        name:"Rutile TiO2", space_group:"P42/mnm (#136)", sp_number:136,
        system:CrystalSystem::Tetragonal, lattice_type:"P", point_group:"4/mmm",
        color:[0.8,0.4,1.0], a:4.594,
        lat_raw:[[1.,0.,0.],[0.,1.,0.],[0.,0.,0.644]],
        fracs: fracs![[0,0,0],[0.5,0.5,0.5],[0.305,0.305,0],[0.695,0.695,0],[0.805,0.195,0.5],[0.195,0.805,0.5]],
        z_vals: zvals![22,22,8,8,8,8], symbols: syms!["Ti","Ti","O","O","O","O"],
    },
    CrystalDef {
        name:"Indium", space_group:"I4/mmm (#139)", sp_number:139,
        system:CrystalSystem::Tetragonal, lattice_type:"I", point_group:"4/mmm",
        color:[0.7,0.85,0.95], a:3.252,
        lat_raw:[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.079]],
        fracs: fracs![[0,0,0],[0.5,0.5,0.5]],
        z_vals: zvals![49,49], symbols: syms!["In","In"],
    },
    CrystalDef {
        name:"Zircon ZrSiO4", space_group:"I41/amd (#141)", sp_number:141,
        system:CrystalSystem::Tetragonal, lattice_type:"I", point_group:"4/mmm",
        color:[0.6,0.5,0.9], a:6.607,
        lat_raw:[[1.,0.,0.],[0.,1.,0.],[0.,0.,0.902]],
        fracs: fracs![[0,0.75,0.125],[0,0.25,0.375],[0,0,0.5],[0.2,0.45,0.25]],
        z_vals: zvals![40,40,14,8], symbols: syms!["Zr","Zr","Si","O"],
    },
    CrystalDef {
        name:"Cassiterite SnO2", space_group:"P42/mnm (#136)", sp_number:136,
        system:CrystalSystem::Tetragonal, lattice_type:"P", point_group:"4/mmm",
        color:[0.9,0.7,0.4], a:4.737,
        lat_raw:[[1.,0.,0.],[0.,1.,0.],[0.,0.,0.674]],
        fracs: fracs![[0,0,0],[0.5,0.5,0.5],[0.307,0.307,0],[0.693,0.693,0],[0.807,0.193,0.5],[0.193,0.807,0.5]],
        z_vals: zvals![50,50,8,8,8,8], symbols: syms!["Sn","Sn","O","O","O","O"],
    },
];

// ── ORTHORHOMBIC ──────────────────────────────────────────────────────────

static ORTHORHOMBIC: &[CrystalDef] = &[
    CrystalDef {
        name:"Aragonite CaCO3", space_group:"Pmcn (#62)", sp_number:62,
        system:CrystalSystem::Orthorhombic, lattice_type:"P", point_group:"mmm",
        color:[0.7,0.9,1.0], a:4.96,
        lat_raw:[[1.,0.,0.],[0.,0.803,0.],[0.,0.,1.509]],
        fracs: fracs![[0.25,0.415,0.758],[0.75,0.585,0.242],[0.25,0.762,0.092],[0.75,0.238,0.908],[0.25,0.922,0.696],[0.75,0.078,0.304]],
        z_vals: zvals![20,20,6,6,8,8], symbols: syms!["Ca","Ca","C","C","O","O"],
    },
    CrystalDef {
        name:"Forsterite Mg2SiO4", space_group:"Pbnm (#62)", sp_number:62,
        system:CrystalSystem::Orthorhombic, lattice_type:"P", point_group:"mmm",
        color:[0.4,0.85,0.5], a:4.758,
        lat_raw:[[1.,0.,0.],[0.,1.232,0.],[0.,0.,1.471]],
        fracs: fracs![[0,0,0],[0.5,0,0.5],[0.277,0.25,0.099],[0.278,0.25,0.651],[0.992,0.25,0.274]],
        z_vals: zvals![12,12,14,8,8], symbols: syms!["Mg","Mg","Si","O","O"],
    },
    CrystalDef {
        name:"Sulphur S8", space_group:"Fddd (#70)", sp_number:70,
        system:CrystalSystem::Orthorhombic, lattice_type:"F", point_group:"mmm",
        color:[1.0,0.95,0.2], a:10.46,
        lat_raw:[[1.,0.,0.],[0.,0.962,0.],[0.,0.,0.747]],
        fracs: fracs![[0.375,0.2,0.073],[0.375,0.08,0.2],[0.375,0.12,0.35],[0.125,0.3,0.073]],
        z_vals: zvals![16,16,16,16], symbols: syms!["S","S","S","S"],
    },
    CrystalDef {
        name:"Topaz Al2SiO4F2", space_group:"Pbnm (#62)", sp_number:62,
        system:CrystalSystem::Orthorhombic, lattice_type:"P", point_group:"mmm",
        color:[0.7,0.85,0.95], a:4.65,
        lat_raw:[[1.,0.,0.],[0.,2.317,0.],[0.,0.,1.869]],
        fracs: fracs![[0.5,0.25,0.202],[0.75,0.065,0.135],[0.75,0.333,0.05],[0.5,0.083,0.0]],
        z_vals: zvals![13,8,8,9], symbols: syms!["Al","O","O","F"],
    },
];

// ── MONOCLINIC ────────────────────────────────────────────────────────────

static MONOCLINIC: &[CrystalDef] = &[
    CrystalDef {
        name:"Gypsum CaSO4·2H2O", space_group:"A2/a (#15)", sp_number:15,
        system:CrystalSystem::Monoclinic, lattice_type:"A", point_group:"2/m",
        color:[1.0,0.95,0.75], a:5.68,
        lat_raw:[[1.,0.,0.],[0.,0.576,0.],[0.255,0.,0.706]],
        fracs: fracs![[0,0.5,0],[0.25,0.682,0.5],[0.25,0.182,0.5],[0.25,0.932,0.25],[0.25,0.432,0.25]],
        z_vals: zvals![20,16,8,8,8], symbols: syms!["Ca","S","O","O","O"],
    },
    CrystalDef {
        name:"Orthoclase KAlSi3O8", space_group:"C2/m (#12)", sp_number:12,
        system:CrystalSystem::Monoclinic, lattice_type:"C", point_group:"2/m",
        color:[1.0,0.7,0.5], a:8.56,
        lat_raw:[[1.,0.,0.],[0.,0.684,0.],[-0.159,0.,0.779]],
        fracs: fracs![[0,0,0],[0.5,0.5,0],[0.292,0,0.146],[0.708,0.5,0.854],[0.345,0.326,0.281],[0.655,0.674,0.719]],
        z_vals: zvals![19,13,14,14,8,8], symbols: syms!["K","Al","Si","Si","O","O"],
    },
    CrystalDef {
        name:"Malachite Cu2CO3(OH)2", space_group:"P21/a (#14)", sp_number:14,
        system:CrystalSystem::Monoclinic, lattice_type:"P", point_group:"2/m",
        color:[0.2,0.8,0.5], a:9.48,
        lat_raw:[[1.,0.,0.],[0.,0.334,0.],[-0.149,0.,0.703]],
        fracs: fracs![[0,0,0],[0.5,0.5,0.5],[0.25,0.25,0.25],[0.75,0.75,0.75],[0.12,0.38,0.12]],
        z_vals: zvals![29,29,6,6,8], symbols: syms!["Cu","Cu","C","C","O"],
    },
    CrystalDef {
        name:"Clinoenstatite MgSiO3", space_group:"P21/c (#14)", sp_number:14,
        system:CrystalSystem::Monoclinic, lattice_type:"P", point_group:"2/m",
        color:[0.7,0.6,0.9], a:9.62,
        lat_raw:[[1.,0.,0.],[0.,0.531,0.],[-0.095,0.,1.067]],
        fracs: fracs![[0,0,0],[0.373,0.25,0.072],[0.622,0.25,0.578],[0.12,0.25,0.36],[0.38,0.25,0.86]],
        z_vals: zvals![12,14,14,8,8], symbols: syms!["Mg","Si","Si","O","O"],
    },
];

// ── TRICLINIC ─────────────────────────────────────────────────────────────

static TRICLINIC: &[CrystalDef] = &[
    CrystalDef {
        name:"Albite NaAlSi3O8", space_group:"P-1 (#2)", sp_number:2,
        system:CrystalSystem::Triclinic, lattice_type:"P", point_group:"-1",
        color:[0.9,0.7,0.7], a:8.14,
        lat_raw:[[1.,0.,0.],[0.118,0.994,0.],[-0.037,0.098,0.945]],
        fracs: fracs![[0,0,0],[0.508,0.338,0.271],[0.02,0.19,0.22],[0.52,0.04,0.28],[0.19,0.11,0.38]],
        z_vals: zvals![11,13,14,14,8], symbols: syms!["Na","Al","Si","Si","O"],
    },
    CrystalDef {
        name:"Kyanite Al2SiO5", space_group:"P-1 (#2)", sp_number:2,
        system:CrystalSystem::Triclinic, lattice_type:"P", point_group:"-1",
        color:[0.5,0.7,1.0], a:7.12,
        lat_raw:[[1.,0.,0.],[0.127,0.992,0.],[-0.086,0.044,0.963]],
        fracs: fracs![[0,0,0],[0.5,0.5,0.5],[0.32,0.70,0.45],[0.68,0.30,0.55],[0.22,0.39,0.28]],
        z_vals: zvals![13,13,14,14,8], symbols: syms!["Al","Al","Si","Si","O"],
    },
    CrystalDef {
        name:"Wollastonite CaSiO3", space_group:"P-1 (#2)", sp_number:2,
        system:CrystalSystem::Triclinic, lattice_type:"P", point_group:"-1",
        color:[0.95,0.85,0.6], a:7.94,
        lat_raw:[[1.,0.,0.],[0.087,0.996,0.],[-0.162,0.072,0.963]],
        fracs: fracs![[0,0,0],[0.5,0.5,0],[0.31,0.08,0.32],[0.31,0.42,0.19],[0.04,0.82,0.31]],
        z_vals: zvals![20,20,14,8,8], symbols: syms!["Ca","Ca","Si","O","O"],
    },
    CrystalDef {
        name:"Microcline KAlSi3O8", space_group:"P-1 (#2)", sp_number:2,
        system:CrystalSystem::Triclinic, lattice_type:"P", point_group:"-1",
        color:[0.9,0.5,0.6], a:8.58,
        lat_raw:[[1.,0.,0.],[0.085,0.996,0.],[-0.108,0.028,0.908]],
        fracs: fracs![[0,0,0],[0.5,0.5,0.5],[0.29,0,0.14],[0.71,0.5,0.86],[0.34,0.32,0.28]],
        z_vals: zvals![19,13,14,14,8], symbols: syms!["K","Al","Si","Si","O"],
    },
];

// ── Public groups list ────────────────────────────────────────────────────

/// Classic 7-system groups (hand-coded, 34 crystals).
pub fn classic_groups() -> Vec<(&'static str, egui::Color32, &'static [CrystalDef])> {
    vec![
        ("CUBIC",        egui::Color32::from_rgb(255,160, 80), CUBIC),
        ("HEXAGONAL",    egui::Color32::from_rgb( 80,220,180), HEXAGONAL),
        ("TRIGONAL",     egui::Color32::from_rgb(160,110,255), TRIGONAL),
        ("TETRAGONAL",   egui::Color32::from_rgb(255,100,180), TETRAGONAL),
        ("ORTHORHOMBIC", egui::Color32::from_rgb( 80,200,255), ORTHORHOMBIC),
        ("MONOCLINIC",   egui::Color32::from_rgb(255,180, 60), MONOCLINIC),
        ("TRICLINIC",    egui::Color32::from_rgb(255,120,120), TRICLINIC),
    ]
}

/// All groups: classic 7 systems + Materials-Project categories (~52 crystals).
pub fn all_groups() -> Vec<(&'static str, egui::Color32, &'static [CrystalDef])> {
    let mut v = classic_groups();
    v.extend(mp::mp_groups());
    v
}

/// Flat list of every crystal from all groups.
pub fn all_crystals() -> Vec<&'static CrystalDef> {
    all_groups().into_iter().flat_map(|(_, _, defs)| defs.iter()).collect()
}
