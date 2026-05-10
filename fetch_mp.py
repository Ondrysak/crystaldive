#!/usr/bin/env python3
"""
Fetch crystal structures from the Materials Project API and emit Rust CrystalDef entries.

Usage: python3 fetch_mp.py > crystals_mp_generated.rs
"""

import requests, json, math, sys, time

API_KEY = "JopaiBgR2QYpgoiW2UEbY9vQpp4hJ5DZ"
BASE    = "https://api.materialsproject.org"
HEADERS = {"X-API-KEY": API_KEY}

# ── Crystal system map ────────────────────────────────────────────────────

SYS_RS = {
    "cubic":        "CrystalSystem::Cubic",
    "hexagonal":    "CrystalSystem::Hexagonal",
    "trigonal":     "CrystalSystem::Trigonal",
    "tetragonal":   "CrystalSystem::Tetragonal",
    "orthorhombic": "CrystalSystem::Orthorhombic",
    "monoclinic":   "CrystalSystem::Monoclinic",
    "triclinic":    "CrystalSystem::Triclinic",
}

# ── Element data ──────────────────────────────────────────────────────────

ELEMENTS = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
    "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
    "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
    "Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,
    "Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,
    "Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,
    "Sb":51,"Te":52,"I":53,"Xe":54,"Cs":55,"Ba":56,"La":57,"Ce":58,
    "Pr":59,"Nd":60,"Pm":61,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,
    "Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71,"Hf":72,"Ta":73,"W":74,
    "Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,"Tl":81,"Pb":82,
    "Bi":83,"Po":84,"At":85,"Rn":86,"Fr":87,"Ra":88,"Ac":89,"Th":90,
    "Pa":91,"U":92,"Np":93,"Pu":94,
}

# ── Colour palette (hue by element group) ────────────────────────────────

def elem_color(sym: str) -> tuple:
    z = ELEMENTS.get(sym, 6)
    if   z <= 2:   return (0.9, 0.9, 1.0)   # noble / H
    elif z <= 10:  return (0.9, 0.5, 0.2)   # p-block light
    elif z <= 18:  return (0.5, 0.8, 0.3)   # Na-Ar
    elif z <= 36:  return (0.3, 0.6, 0.9)   # K-Kr (3d transition)
    elif z <= 54:  return (0.8, 0.3, 0.7)   # Rb-Xe (4d transition)
    elif z <= 86:  return (0.7, 0.9, 0.4)   # Cs-Rn (5d+lanthanoids)
    else:          return (1.0, 0.5, 0.5)   # actinoids

def mix_colors(symbols):
    if not symbols: return (0.6, 0.6, 0.8)
    r, g, b = 0.0, 0.0, 0.0
    for s in set(symbols):
        cr, cg, cb = elem_color(s)
        r += cr; g += cg; b += cb
    n = len(set(symbols))
    return (r/n, g/n, b/n)

# ── Geometry helpers ──────────────────────────────────────────────────────

def norm(v):
    return math.sqrt(sum(x*x for x in v))

def to_lat_raw(matrix):
    """Divide every row by |row0| to get dimensionless lat_raw."""
    a = norm(matrix[0])
    return [[x/a for x in row] for row in matrix], a

# ── API helpers ───────────────────────────────────────────────────────────

def get_json(url, params):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2)

def fetch_by_ids(mp_ids: list):
    """Fetch full structure + symmetry for a list of mp-ids."""
    ids_str = ",".join(mp_ids)
    data = get_json(
        f"{BASE}/materials/summary/",
        {
            "_fields": "material_id,formula_pretty,symmetry,structure",
            "material_ids": ids_str,
            "_limit": len(mp_ids),
        }
    )
    return data.get("data", [])

def fetch_by_system(crystal_system: str, n: int = 8, nsites_max: int = 10):
    """Fetch stable, simple structures for a given crystal system."""
    data = get_json(
        f"{BASE}/materials/summary/",
        {
            "_fields": "material_id,formula_pretty,symmetry,structure",
            "crystal_system": crystal_system,
            "is_stable": True,
            "nsites_max": nsites_max,
            "_limit": n,
            "_sort_fields": "nsites",
        }
    )
    return data.get("data", [])

# ── Curated list of interesting materials ─────────────────────────────────
# (material_id, short label for name override, group label)

CURATED = [
    # ── Topological / Quantum ─────────────────────────────────────────
    ("mp-541837",  "Bi2Se3 (Topological Insulator)",    "TOPOLOGICAL"),
    ("mp-34202",   "Bi2Te3 (Thermoelectric TI)",        "TOPOLOGICAL"),
    ("mp-2286",    "SnTe (Topological Crystalline)",    "TOPOLOGICAL"),
    ("mp-20305",   "Pb2SnTe2 (TCI)",                   "TOPOLOGICAL"),
    ("mp-22598",   "Sb2Te3 (Topological Insulator)",   "TOPOLOGICAL"),
    ("mp-569572",  "TaAs (Weyl Semimetal)",             "TOPOLOGICAL"),
    ("mp-10182",   "WTe2 (Type-II Weyl Semimetal)",    "TOPOLOGICAL"),
    ("mp-672",     "Bi (Semimetal)",                    "TOPOLOGICAL"),

    # ── Superconductors ───────────────────────────────────────────────
    ("mp-763",     "MgB2 (Superconductor 39K)",         "SUPERCONDUCTORS"),
    ("mp-11415",   "NbN (Superconductor 16K)",          "SUPERCONDUCTORS"),
    ("mp-23",      "Nb (BCC Superconductor 9K)",        "SUPERCONDUCTORS"),
    ("mp-7643",    "V3Si (A15 Superconductor)",         "SUPERCONDUCTORS"),
    ("mp-8636",    "Nb3Sn (A15 Superconductor 18K)",   "SUPERCONDUCTORS"),
    ("mp-19326",   "NbSe2 (CDW Superconductor)",       "SUPERCONDUCTORS"),
    ("mp-20561",   "FeSe (Iron Selenide SC)",           "SUPERCONDUCTORS"),
    ("mp-1639",    "TiN (Hard Superconductor)",        "SUPERCONDUCTORS"),

    # ── 2-D / Layered ─────────────────────────────────────────────────
    ("mp-2815",    "MoS2 (2D Semiconductor)",           "LAYERED_2D"),
    ("mp-224",     "WS2 (2D TMD)",                      "LAYERED_2D"),
    ("mp-1821",    "WSe2 (2D TMD)",                     "LAYERED_2D"),
    ("mp-1634",    "BN (Hexagonal Boron Nitride)",      "LAYERED_2D"),
    ("mp-1020057", "MoSe2 (2D TMD)",                   "LAYERED_2D"),
    ("mp-27711",   "SnS2 (2D Semiconductor)",           "LAYERED_2D"),
    ("mp-1018809", "MoTe2 (Topological TMD)",          "LAYERED_2D"),
    ("mp-602",     "Bi2S3 (Layered Semiconductor)",    "LAYERED_2D"),

    # ── Magnets ───────────────────────────────────────────────────────
    ("mp-19306",   "Fe3O4 (Magnetite)",                 "MAGNETIC"),
    ("mp-3536",    "CoFe2O4 (Cobalt Ferrite)",          "MAGNETIC"),
    ("mp-390",     "NiO (Antiferromagnet)",             "MAGNETIC"),
    ("mp-568",     "MnO (Antiferromagnet)",             "MAGNETIC"),
    ("mp-18759",   "CrO2 (Half-metal Magnet)",          "MAGNETIC"),
    ("mp-19079",   "MnBi (Permanent Magnet)",           "MAGNETIC"),
    ("mp-545483",  "BiFeO3 (Room-T Multiferroic)",     "MAGNETIC"),
    ("mp-25017",   "Cr2O3 (Chromia AFM)",              "MAGNETIC"),

    # ── Battery / Energy ──────────────────────────────────────────────
    ("mp-22526",   "LiCoO2 (Layered Cathode)",          "ENERGY"),
    ("mp-19017",   "LiFePO4 (Olivine Cathode)",         "ENERGY"),
    ("mp-1272804", "LiMn2O4 (Spinel Cathode)",          "ENERGY"),
    ("mp-18921",   "NaCoO2 (Na-ion Cathode)",           "ENERGY"),
    ("mp-25279",   "V2O5 (Intercalation Cathode)",      "ENERGY"),
    ("mp-19770",   "TiS2 (2D Cathode)",                 "ENERGY"),
    ("mp-20194",   "CeO2 (Fuel Cell Oxide)",           "ENERGY"),
    ("mp-23240",   "Li2O2 (Li-Air Cathode)",           "ENERGY"),

    # ── Semiconductors ────────────────────────────────────────────────
    ("mp-149",     "Si (Diamond Cubic)",                "SEMICONDUCTORS"),
    ("mp-32",      "Ge (Group-IV Semiconductor)",       "SEMICONDUCTORS"),
    ("mp-830",     "GaN (Nitride LED)",                 "SEMICONDUCTORS"),
    ("mp-2133",    "ZnO (Wide-gap Semiconductor)",      "SEMICONDUCTORS"),
    ("mp-10695",   "CdTe (Solar Cell)",                 "SEMICONDUCTORS"),
    ("mp-406",     "InP (III-V Semiconductor)",         "SEMICONDUCTORS"),
    ("mp-2534",    "CdS (II-VI Semiconductor)",         "SEMICONDUCTORS"),
    ("mp-8062",    "GaP (III-V Semiconductor)",         "SEMICONDUCTORS"),
    ("mp-2172",    "InAs (III-V Narrow-gap)",           "SEMICONDUCTORS"),
    ("mp-1550",    "GaAs (III-V Semiconductor)",       "SEMICONDUCTORS"),

    # ── Metals & Alloys ───────────────────────────────────────────────
    ("mp-81",      "Cu (FCC)",                          "METALS"),
    ("mp-13",      "Fe (BCC)",                          "METALS"),
    ("mp-146",     "W (BCC Refractory)",                "METALS"),
    ("mp-129",     "Pt (FCC Catalyst)",                 "METALS"),
    ("mp-124",     "Al (FCC Light Metal)",              "METALS"),
    ("mp-35",      "Ti (HCP)",                          "METALS"),
    ("mp-75",      "Ni (FCC Magnetic)",                 "METALS"),
    ("mp-127",     "Ag (FCC)",                          "METALS"),
    ("mp-126",     "Pd (FCC Catalyst)",                 "METALS"),
    ("mp-72",      "Os (HCP)",                          "METALS"),
    ("mp-101",     "Ir (FCC Hard Metal)",               "METALS"),
    ("mp-33",      "Ru (HCP Catalyst)",                 "METALS"),
    ("mp-91",      "Cr (BCC Antiferromagnet)",          "METALS"),
    ("mp-54",      "Co (HCP Magnet)",                   "METALS"),
    ("mp-74",      "Mo (BCC Refractory)",               "METALS"),
    ("mp-216",     "Ta (BCC Refractory)",               "METALS"),

    # ── Piezo / Ferroelectric ─────────────────────────────────────────
    ("mp-5020",    "BaTiO3 (Tetragonal FE)",            "FUNCTIONAL"),
    ("mp-4019",    "PbTiO3 (Ferroelectric)",            "FUNCTIONAL"),
    ("mp-3731",    "LiNbO3 (Electro-optic)",            "FUNCTIONAL"),
    ("mp-661",     "AlN (Piezoelectric Nitride)",       "FUNCTIONAL"),
    ("mp-4651",    "KNbO3 (Ferroelectric)",             "FUNCTIONAL"),
    ("mp-2418",    "SrTiO3 (Quantum Paraelectric)",    "FUNCTIONAL"),
    ("mp-4626",    "LaAlO3 (2DEG Interface)",           "FUNCTIONAL"),

    # ── Thermoelectrics ───────────────────────────────────────────────
    ("mp-19717",   "PbTe (Classic Thermoelectric)",    "THERMOELECTRICS"),
    ("mp-938",     "GeTe (Phase-change TE)",            "THERMOELECTRICS"),
    ("mp-7440",    "CoSb3 (Skutterudite TE)",           "THERMOELECTRICS"),
    ("mp-3813",    "Cu2Se (Liquid-like TE)",            "THERMOELECTRICS"),
    ("mp-2748",    "SnSe (Record TE ZT>2.6)",          "THERMOELECTRICS"),
    ("mp-1450",    "Bi2Te2Se (TI Thermoelectric)",     "THERMOELECTRICS"),
    ("mp-22598",   "Sb2Te3 (TE Compound)",             "THERMOELECTRICS"),  # shared with TOPO

    # ── Halide Perovskites ────────────────────────────────────────────
    ("mp-600089",  "CsPbBr3 (Green LED Perovskite)",  "HALIDE_PEROV"),
    ("mp-1067827", "CsPbI3 (Solar Perovskite)",        "HALIDE_PEROV"),
    ("mp-27357",   "CsSnI3 (Lead-free Perovskite)",   "HALIDE_PEROV"),
    ("mp-567629",  "CsPbCl3 (Blue LED Perovskite)",   "HALIDE_PEROV"),

    # ── Minerals / Geological ─────────────────────────────────────────
    ("mp-5827",    "MgO (Periclase)",                   "MINERALS"),
    ("mp-1265",    "SiO2 (beta-Quartz)",                "MINERALS"),
    ("mp-2920",    "TiO2 (Anatase)",                    "MINERALS"),
    ("mp-19399",   "FeS2 (Marcasite)",                  "MINERALS"),
    ("mp-856",     "ZrO2 (Baddeleyite)",                "MINERALS"),
    ("mp-7048",    "Al2O3 (Corundum)",                  "MINERALS"),
    ("mp-4163",    "CaTiO3 (Perovskite mineral)",       "MINERALS"),
    ("mp-2657",    "MgSiO3 (Enstatite)",               "MINERALS"),
    ("mp-4979",    "CaCO3 (Calcite)",                  "MINERALS"),
    ("mp-3564",    "FeS (Troilite)",                   "MINERALS"),
]

# deduplicate by mp-id
seen_ids = {}
CURATED_DEDUP = []
for mp_id, label, group in CURATED:
    if mp_id not in seen_ids:
        seen_ids[mp_id] = True
        CURATED_DEDUP.append((mp_id, label, group))

# ── Rust code generation ──────────────────────────────────────────────────

def fmt_f32(x: float) -> str:
    s = f"{x:.4f}"
    # Trim trailing zeros but keep at least one decimal
    s = s.rstrip('0').rstrip('.')
    if '.' not in s: s += '.0'
    return s

def mat_entry(mp_id, label, group, mat) -> str | None:
    try:
        struct = mat["structure"]
        sym    = mat["symmetry"]
        matrix = struct["lattice"]["matrix"]   # [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]
        sites  = struct["sites"]

        lat_raw, a = to_lat_raw(matrix)

        fracs   = [s["abc"] for s in sites]
        symbols = [s["species"][0]["element"] for s in sites]
        z_vals  = [ELEMENTS.get(sym_el, 6) for sym_el in symbols]

        if len(fracs) == 0 or len(fracs) > 20:
            return None

        col = mix_colors(symbols)
        sg_sym = sym.get("symbol", "?")
        sg_num = sym.get("number", 0)
        cs     = sym.get("crystal_system", "cubic").lower()
        cs_rs  = SYS_RS.get(cs, "CrystalSystem::Cubic")

        # Build fracs! macro
        fracs_str = ", ".join(
            f"[{fmt_f32(f[0])},{fmt_f32(f[1])},{fmt_f32(f[2])}]" for f in fracs
        )
        z_str = ", ".join(str(z) for z in z_vals)
        sym_str = ", ".join(f'"{s}"' for s in symbols)

        # lat_raw rows
        lr = lat_raw
        lat_str = (
            f"[[{fmt_f32(lr[0][0])},{fmt_f32(lr[0][1])},{fmt_f32(lr[0][2])}],"
            f"[{fmt_f32(lr[1][0])},{fmt_f32(lr[1][1])},{fmt_f32(lr[1][2])}],"
            f"[{fmt_f32(lr[2][0])},{fmt_f32(lr[2][1])},{fmt_f32(lr[2][2])}]]"
        )

        col_str = f"[{fmt_f32(col[0])},{fmt_f32(col[1])},{fmt_f32(col[2])}]"

        return f'''\
    CrystalDef {{
        name:"{label}", space_group:"{sg_sym} (#{sg_num})", sp_number:{sg_num},
        system:{cs_rs}, lattice_type:"P", point_group:"?",
        color:{col_str}, a:{fmt_f32(a)},
        lat_raw:{lat_str},
        fracs: fracs![{fracs_str}],
        z_vals: zvals![{z_str}], symbols: syms![{sym_str}],
    }},'''
    except Exception as e:
        print(f"    // ERROR for {mp_id} ({label}): {e}", file=sys.stderr)
        return None


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    # Group the curated list by group label
    groups: dict[str, list] = {}
    id_to_meta: dict[str, tuple] = {}
    for mp_id, label, group in CURATED_DEDUP:
        groups.setdefault(group, []).append(mp_id)
        id_to_meta[mp_id] = (label, group)

    # Fetch all at once in batches of 20
    all_ids = list(id_to_meta.keys())
    fetched: dict[str, dict] = {}
    batch = 20
    for i in range(0, len(all_ids), batch):
        chunk = all_ids[i:i+batch]
        print(f"// Fetching batch {i//batch+1}: {chunk}", file=sys.stderr)
        try:
            mats = fetch_by_ids(chunk)
            for m in mats:
                fetched[m["material_id"]] = m
        except Exception as e:
            print(f"// Batch error: {e}", file=sys.stderr)
        time.sleep(0.5)

    print("// Auto-generated from Materials Project API – do not edit manually.")
    print("// Run: python3 fetch_mp.py > src/crystals_mp.rs")
    print()
    print("use crate::poscar::{frac_to_cart, Crystal, Atom};")
    print("use crate::symmetry::CrystalSystem;")
    print("use super::{CrystalDef, fracs, zvals, syms};")
    print()

    group_arrays = []

    for group_name, ids in groups.items():
        entries = []
        for mp_id in ids:
            if mp_id not in fetched:
                print(f"    // {mp_id} not fetched", file=sys.stderr)
                continue
            label, _ = id_to_meta[mp_id]
            entry = mat_entry(mp_id, label, group_name, fetched[mp_id])
            if entry:
                entries.append(entry)

        if not entries:
            continue

        var_name = group_name.upper()
        print(f"pub static {var_name}: &[CrystalDef] = &[")
        for e in entries:
            print(e)
        print("];")
        print()
        group_arrays.append((group_name, var_name))

    # Emit the groups() helper
    GROUP_COLORS = {
        "TOPOLOGICAL":    "egui::Color32::from_rgb(120,220,255)",
        "SUPERCONDUCTORS":"egui::Color32::from_rgb(180,120,255)",
        "LAYERED_2D":     "egui::Color32::from_rgb( 80,240,180)",
        "MAGNETIC":       "egui::Color32::from_rgb(255, 80, 80)",
        "ENERGY":         "egui::Color32::from_rgb(255,210, 50)",
        "SEMICONDUCTORS": "egui::Color32::from_rgb(100,200,100)",
        "METALS":         "egui::Color32::from_rgb(200,200,200)",
        "FUNCTIONAL":     "egui::Color32::from_rgb(255,140, 80)",
        "THERMOELECTRICS":"egui::Color32::from_rgb(255,100,160)",
        "HALIDE_PEROV":   "egui::Color32::from_rgb(255,230,100)",
        "MINERALS":       "egui::Color32::from_rgb(180,160,120)",
    }

    print("pub fn mp_groups() -> Vec<(&'static str, egui::Color32, &'static [CrystalDef])> {")
    print("    vec![")
    for name, var in group_arrays:
        color = GROUP_COLORS.get(name, "egui::Color32::from_rgb(180,180,180)")
        print(f'        ("{name}", {color}, {var}),')
    print("    ]")
    print("}")
    print()
    print("pub fn mp_all_crystals() -> Vec<&'static CrystalDef> {")
    print("    mp_groups().into_iter().flat_map(|(_, _, d)| d.iter()).collect()")
    print("}")

if __name__ == "__main__":
    main()
