use std::fs;

#[derive(Debug, Clone)]
pub struct Atom {
    pub species: String,    // raw label from file, e.g. "Na+" or "Na"
    pub pos_cart: [f32; 3], // Cartesian position in Ångströms
}

#[derive(Debug, Clone)]
pub struct Crystal {
    pub lattice: [[f32; 3]; 3], // lattice[i] = i-th basis vector (row) in Å
    pub atoms: Vec<Atom>,
}

impl Crystal {
    pub fn from_file(path: &str) -> Result<Self, String> {
        let src = fs::read_to_string(path)
            .map_err(|e| format!("Cannot read '{path}': {e}"))?;
        Self::parse(&src)
    }

    fn parse(src: &str) -> Result<Self, String> {
        let mut lines = src.lines();

        // 1 — comment
        lines.next().ok_or("EOF at comment line")?;

        // 2 — scale
        let scale: f32 = lines
            .next()
            .ok_or("EOF at scale")?
            .split_whitespace()
            .next()
            .ok_or("empty scale line")?
            .parse()
            .map_err(|e| format!("bad scale: {e}"))?;

        // 3-5 — lattice vectors
        let mut lattice = [[0f32; 3]; 3];
        for (i, row) in lattice.iter_mut().enumerate() {
            let l = lines.next().ok_or(format!("EOF at lattice row {i}"))?;
            let v: Vec<f32> = l
                .split_whitespace()
                .take(3)
                .map(|s| s.parse::<f32>().map_err(|e| format!("lattice: {e}")))
                .collect::<Result<_, _>>()?;
            if v.len() < 3 {
                return Err(format!("lattice row {i} has fewer than 3 values"));
            }
            row[0] = v[0] * scale;
            row[1] = v[1] * scale;
            row[2] = v[2] * scale;
        }

        // 6 — VASP5 species names, or VASP4 counts
        let line6 = lines.next().ok_or("EOF at species/counts")?;
        let first = line6.split_whitespace().next().unwrap_or("");

        let (species, counts_line): (Vec<String>, String) =
            if first.parse::<u32>().is_err() && !first.is_empty() {
                // VASP5: element symbols on this line, counts on the next
                let names = line6.split_whitespace().map(String::from).collect();
                let cl = lines.next().ok_or("EOF at counts")?.to_owned();
                (names, cl)
            } else {
                // VASP4: no species names
                let n = line6.split_whitespace().count();
                ((0..n).map(|i| format!("X{i}")).collect(), line6.to_owned())
            };

        let counts: Vec<usize> = counts_line
            .split_whitespace()
            .map(|s| s.parse::<usize>().map_err(|e| format!("count: {e}")))
            .collect::<Result<_, _>>()?;

        // 7 — optional "Selective dynamics", then Direct/Cartesian
        let mut coord_line = lines.next().ok_or("EOF at coord type")?.to_owned();
        if coord_line.trim().to_ascii_lowercase().starts_with('s') {
            coord_line = lines.next().ok_or("EOF at coord type (after S.D.)")?.to_owned();
        }
        let is_direct = coord_line.trim().to_ascii_lowercase().starts_with('d');

        // 8+ — atom positions
        let total: usize = counts.iter().sum();
        let mut atoms = Vec::with_capacity(total);

        for (sp, &cnt) in species.iter().zip(counts.iter()) {
            for _ in 0..cnt {
                // skip any blank lines
                let l = loop {
                    match lines.next() {
                        Some(s) if s.trim().is_empty() => continue,
                        Some(s) => break s.to_owned(),
                        None => return Err("EOF: not enough atom positions".to_owned()),
                    }
                };

                let mut tok = l.split_whitespace();
                let x: f32 = tok.next().ok_or("atom: missing x")?.parse().map_err(|e| format!("x: {e}"))?;
                let y: f32 = tok.next().ok_or("atom: missing y")?.parse().map_err(|e| format!("y: {e}"))?;
                let z: f32 = tok.next().ok_or("atom: missing z")?.parse().map_err(|e| format!("z: {e}"))?;

                // Inline label ("Na+") overrides the header species
                let inline = tok.next();
                let species_str = inline
                    .filter(|s| s.chars().next().map(|c| c.is_ascii_alphabetic()).unwrap_or(false))
                    .map(String::from)
                    .unwrap_or_else(|| sp.clone());

                let pos_cart = if is_direct {
                    frac_to_cart([x, y, z], &lattice)
                } else {
                    [x, y, z]
                };

                atoms.push(Atom { species: species_str, pos_cart });
            }
        }

        Ok(Crystal { lattice, atoms })
    }

    /// Cartesian mid-point of the cell: 0.5*(a1 + a2 + a3)
    pub fn cell_center(&self) -> [f32; 3] {
        let mut c = [0f32; 3];
        for row in &self.lattice {
            c[0] += 0.5 * row[0];
            c[1] += 0.5 * row[1];
            c[2] += 0.5 * row[2];
        }
        c
    }
}

pub fn frac_to_cart(frac: [f32; 3], lat: &[[f32; 3]; 3]) -> [f32; 3] {
    [
        frac[0] * lat[0][0] + frac[1] * lat[1][0] + frac[2] * lat[2][0],
        frac[0] * lat[0][1] + frac[1] * lat[1][1] + frac[2] * lat[2][1],
        frac[0] * lat[0][2] + frac[1] * lat[1][2] + frac[2] * lat[2][2],
    ]
}

/// CPK-ish colour for an element symbol (strips trailing charge notation).
pub fn element_color(sym: &str) -> [f32; 3] {
    let s: String = sym.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
    match s.as_str() {
        "H"  => [1.00, 1.00, 1.00],
        "Li" => [0.80, 0.50, 1.00],
        "C"  => [0.34, 0.34, 0.34],
        "N"  => [0.19, 0.31, 0.97],
        "O"  => [1.00, 0.05, 0.05],
        "F"  => [0.56, 0.88, 0.31],
        "Na" => [0.67, 0.36, 0.95],
        "Mg" => [0.54, 1.00, 0.00],
        "Al" => [0.75, 0.65, 0.65],
        "Si" => [0.94, 0.78, 0.63],
        "P"  => [1.00, 0.50, 0.00],
        "S"  => [1.00, 1.00, 0.19],
        "Cl" => [0.12, 0.94, 0.12],
        "K"  => [0.56, 0.25, 0.83],
        "Ca" => [0.24, 1.00, 0.00],
        "Ti" => [0.75, 0.76, 0.78],
        "Fe" => [0.88, 0.40, 0.20],
        "Cu" => [0.78, 0.50, 0.20],
        "Zn" => [0.49, 0.50, 0.69],
        "Br" => [0.65, 0.16, 0.16],
        "Ag" => [0.75, 0.75, 0.75],
        "I"  => [0.58, 0.00, 0.58],
        "Au" => [1.00, 0.82, 0.14],
        "Pb" => [0.34, 0.35, 0.38],
        _    => [1.00, 0.08, 0.58], // hot-pink fallback
    }
}

/// Display radius in Ångströms (scaled for visual clarity, not true vdW).
pub fn element_radius(sym: &str) -> f32 {
    let s: String = sym.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
    match s.as_str() {
        "H"  => 0.25,
        "Li" => 0.55,
        "C"  => 0.40,
        "N"  => 0.37,
        "O"  => 0.35,
        "F"  => 0.30,
        "Na" => 0.55,
        "Mg" => 0.50,
        "Al" => 0.50,
        "Si" => 0.50,
        "P"  => 0.45,
        "S"  => 0.45,
        "Cl" => 0.50,
        "K"  => 0.65,
        "Ca" => 0.60,
        "Ti" => 0.55,
        "Fe" => 0.55,
        "Cu" => 0.50,
        "Zn" => 0.50,
        "Br" => 0.55,
        "Ag" => 0.65,
        "I"  => 0.65,
        "Au" => 0.65,
        "Pb" => 0.75,
        _    => 0.50,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NACL_VASP5: &str = "NaCl
1.0
5.64 0.0  0.0
0.0  5.64 0.0
0.0  0.0  5.64
Na Cl
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
";

    const SI_CARTESIAN: &str = "Si diamond
2.0
2.715  0.000  0.000
0.000  2.715  0.000
0.000  0.000  2.715
Si
2
Cartesian
0.0  0.0  0.0
1.3575 1.3575 1.3575
";

    #[test]
    fn parse_vasp5_direct_coords() {
        let c = Crystal::parse(NACL_VASP5).expect("parse");
        assert_eq!(c.atoms.len(), 2);
        assert_eq!(c.atoms[0].species, "Na");
        assert_eq!(c.atoms[1].species, "Cl");
        // Frac (0.5, 0.5, 0.5) → cart (2.82, 2.82, 2.82) for diagonal lattice.
        let p = c.atoms[1].pos_cart;
        for v in p { assert!((v - 2.82).abs() < 1e-3, "expected 2.82, got {v}"); }
    }

    #[test]
    fn parse_vasp5_cartesian_with_scale() {
        let c = Crystal::parse(SI_CARTESIAN).expect("parse");
        assert_eq!(c.atoms.len(), 2);
        // scale=2 multiplies the lattice; cartesian atom coords are NOT scaled by VASP convention,
        // but this parser does not multiply atom coords by scale — verify lattice scaling.
        assert!((c.lattice[0][0] - 5.43).abs() < 1e-3);
    }

    #[test]
    fn cell_center_diagonal() {
        let c = Crystal::parse(NACL_VASP5).expect("parse");
        let cc = c.cell_center();
        for v in cc { assert!((v - 2.82).abs() < 1e-3); }
    }

    #[test]
    fn frac_to_cart_diagonal() {
        let lat = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]];
        let p = frac_to_cart([0.5, 0.25, 0.1], &lat);
        assert!((p[0] - 1.5).abs() < 1e-5);
        assert!((p[1] - 1.0).abs() < 1e-5);
        assert!((p[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn element_color_strips_charge() {
        // "Na+" should resolve to the same colour as "Na".
        assert_eq!(element_color("Na+"), element_color("Na"));
        assert_eq!(element_color("Cl-"), element_color("Cl"));
    }

    #[test]
    fn element_radius_unknown_falls_back() {
        let r = element_radius("Xx");
        assert!(r > 0.0);
    }

    #[test]
    fn parse_rejects_garbage() {
        assert!(Crystal::parse("not a poscar").is_err());
        assert!(Crystal::parse("").is_err());
    }
}
