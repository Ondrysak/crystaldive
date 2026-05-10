/// Geometric crystal-system classification from lattice vectors.
/// No spglib needed — purely from a, b, c and the inter-vector angles.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrystalSystem {
    Cubic,
    Hexagonal,
    Trigonal,
    Tetragonal,
    Orthorhombic,
    Monoclinic,
    Triclinic,
}

impl CrystalSystem {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cubic        => "Cubic",
            Self::Hexagonal    => "Hexagonal",
            Self::Trigonal     => "Trigonal (Rhombohedral)",
            Self::Tetragonal   => "Tetragonal",
            Self::Orthorhombic => "Orthorhombic",
            Self::Monoclinic   => "Monoclinic",
            Self::Triclinic    => "Triclinic",
        }
    }

    /// Accent colour used for the BZ / symmetry decorations in the visualizer.
    pub fn accent(self) -> [f32; 3] {
        match self {
            Self::Cubic        => [0.40, 0.90, 1.00], // icy cyan
            Self::Hexagonal    => [1.00, 0.60, 0.20], // amber
            Self::Trigonal     => [0.80, 0.40, 1.00], // violet
            Self::Tetragonal   => [0.30, 1.00, 0.50], // green
            Self::Orthorhombic => [1.00, 0.90, 0.20], // yellow
            Self::Monoclinic   => [1.00, 0.40, 0.60], // rose
            Self::Triclinic    => [0.70, 0.70, 0.70], // grey
        }
    }
}

/// Classify the crystal system from the 3×3 lattice (rows = vectors, in Å).
pub fn detect(lattice: &[[f32; 3]; 3]) -> CrystalSystem {
    let a = len(lattice[0]);
    let b = len(lattice[1]);
    let c = len(lattice[2]);

    let cos_al = dot(lattice[1], lattice[2]) / (b * c); // angle between b and c
    let cos_be = dot(lattice[0], lattice[2]) / (a * c);
    let cos_ga = dot(lattice[0], lattice[1]) / (a * b);

    let alpha = cos_al.acos().to_degrees();
    let beta  = cos_be.acos().to_degrees();
    let gamma = cos_ga.acos().to_degrees();

    let tol_l = a * 0.002; // 0.2 % of a
    let tol_a = 1.0_f32;   // 1 degree

    let eq = |x: f32, y: f32| (x - y).abs() < tol_l;
    let ang = |x: f32, t: f32| (x - t).abs() < tol_a;

    if eq(a, b) && eq(b, c) && ang(alpha, 90.) && ang(beta, 90.) && ang(gamma, 90.) {
        CrystalSystem::Cubic
    } else if eq(a, b) && ang(alpha, 90.) && ang(beta, 90.) && ang(gamma, 120.) {
        CrystalSystem::Hexagonal
    } else if eq(a, b) && eq(b, c) && ang(alpha, beta) && ang(beta, gamma) && !ang(alpha, 90.) {
        CrystalSystem::Trigonal
    } else if eq(a, b) && ang(alpha, 90.) && ang(beta, 90.) && ang(gamma, 90.) {
        CrystalSystem::Tetragonal
    } else if ang(alpha, 90.) && ang(beta, 90.) && ang(gamma, 90.) {
        CrystalSystem::Orthorhombic
    } else if ang(alpha, 90.) && ang(gamma, 90.) {
        CrystalSystem::Monoclinic
    } else {
        CrystalSystem::Triclinic
    }
}

fn len(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic() {
        let lat = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]];
        assert_eq!(detect(&lat), CrystalSystem::Cubic);
    }

    #[test]
    fn tetragonal() {
        let lat = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]];
        assert_eq!(detect(&lat), CrystalSystem::Tetragonal);
    }

    #[test]
    fn orthorhombic() {
        let lat = [[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 7.0]];
        assert_eq!(detect(&lat), CrystalSystem::Orthorhombic);
    }

    #[test]
    fn hexagonal() {
        // a=b=3, c=5, gamma=120°: a along x, b at 120° from a in xy plane.
        let lat = [
            [3.0, 0.0, 0.0],
            [-1.5, 3.0 * 0.8660254, 0.0],
            [0.0, 0.0, 5.0],
        ];
        assert_eq!(detect(&lat), CrystalSystem::Hexagonal);
    }

    #[test]
    fn monoclinic() {
        // alpha=gamma=90°, beta=110° — second vector tilted in xz.
        let lat = [
            [4.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [6.0 * (110.0_f32.to_radians()).cos(), 0.0, 6.0 * (110.0_f32.to_radians()).sin()],
        ];
        assert_eq!(detect(&lat), CrystalSystem::Monoclinic);
    }

    #[test]
    fn names_unique_and_nonempty() {
        let all = [
            CrystalSystem::Cubic, CrystalSystem::Hexagonal, CrystalSystem::Trigonal,
            CrystalSystem::Tetragonal, CrystalSystem::Orthorhombic,
            CrystalSystem::Monoclinic, CrystalSystem::Triclinic,
        ];
        let mut names: Vec<&str> = all.iter().map(|s| s.name()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 7);
        assert!(names.iter().all(|n| !n.is_empty()));
    }
}
