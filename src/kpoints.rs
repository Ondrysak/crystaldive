//! High-symmetry k-points and BZ path walker, one set per crystal system.

use crate::symmetry::CrystalSystem;

pub struct KPoint {
    pub label:  &'static str,
    pub coords: [f32; 3],
}

pub struct KpathWalker {
    points:    Vec<KPoint>,
    path:      Vec<usize>,  // indices into points
    pub seg:   usize,
    pub t:     f32,         // [0, 1) within current segment
    pub speed: f32,         // segments / second
}

impl KpathWalker {
    pub fn new(points: Vec<KPoint>, path: Vec<usize>) -> Self {
        Self { points, path, seg: 0, t: 0.0, speed: 0.18 }
    }

    /// Advance by `dt` seconds, return (interpolated k-coords, label of nearest k-point).
    pub fn tick(&mut self, dt: f32) -> ([f32; 3], &str) {
        let n = self.path.len().saturating_sub(1);
        if n == 0 { return ([0.0; 3], "Γ"); }

        self.t += dt * self.speed;
        if self.t >= 1.0 {
            self.t -= 1.0;
            self.seg = (self.seg + 1) % n;
        }
        let ia = self.path[self.seg];
        let ib = self.path[(self.seg + 1).min(self.path.len() - 1)];
        let pa = &self.points[ia].coords;
        let pb = &self.points[ib].coords;
        let tt = self.t;
        let k  = [pa[0]+(pb[0]-pa[0])*tt, pa[1]+(pb[1]-pa[1])*tt, pa[2]+(pb[2]-pa[2])*tt];
        let lbl = if tt < 0.5 { self.points[ia].label } else { self.points[ib].label };
        (k, lbl)
    }

    pub fn reset(&mut self) { self.seg = 0; self.t = 0.0; }

    /// Jump directly to k-point at list index `idx`, return its coords.
    pub fn snap_to(&self, idx: usize) -> [f32; 3] {
        self.points[idx % self.points.len()].coords
    }

    pub fn label_at(&self, idx: usize) -> &str {
        self.points.get(idx % self.points.len()).map(|p| p.label).unwrap_or("?")
    }

    pub fn n_points(&self) -> usize { self.points.len() }
}

/// Build a k-path walker for the given crystal system.
pub fn build_kpath(sys: CrystalSystem) -> KpathWalker {
    let (pts, path) = match sys {
        CrystalSystem::Cubic => (vec![
            KPoint { label: "Γ", coords: [0.000, 0.000, 0.000] },
            KPoint { label: "X", coords: [0.500, 0.000, 0.500] },
            KPoint { label: "W", coords: [0.500, 0.250, 0.750] },
            KPoint { label: "K", coords: [0.375, 0.375, 0.750] },
            KPoint { label: "L", coords: [0.500, 0.500, 0.500] },
            KPoint { label: "U", coords: [0.625, 0.250, 0.625] },
            KPoint { label: "M", coords: [0.500, 0.500, 0.000] },
        ], vec![0,1,2,3,0,4,5,2,4,3,0]),

        CrystalSystem::Hexagonal => (vec![
            KPoint { label: "Γ", coords: [0.000, 0.000, 0.000] },
            KPoint { label: "M", coords: [0.500, 0.000, 0.000] },
            KPoint { label: "K", coords: [0.333, 0.333, 0.000] },
            KPoint { label: "A", coords: [0.000, 0.000, 0.500] },
            KPoint { label: "L", coords: [0.500, 0.000, 0.500] },
            KPoint { label: "H", coords: [0.333, 0.333, 0.500] },
        ], vec![0,2,1,0,3,5,4,3,2,5,0]),

        CrystalSystem::Trigonal => (vec![
            KPoint { label: "Γ", coords: [0.000, 0.000, 0.000] },
            KPoint { label: "Z", coords: [0.500, 0.500, 0.500] },
            KPoint { label: "F", coords: [0.500, 0.500, 0.000] },
            KPoint { label: "L", coords: [0.500, 0.000, 0.000] },
            KPoint { label: "Q", coords: [0.333, 0.333, 0.333] },
        ], vec![0,1,2,0,3,1,4,0]),

        CrystalSystem::Tetragonal => (vec![
            KPoint { label: "Γ", coords: [0.000, 0.000, 0.000] },
            KPoint { label: "X", coords: [0.500, 0.000, 0.000] },
            KPoint { label: "M", coords: [0.500, 0.500, 0.000] },
            KPoint { label: "Z", coords: [0.000, 0.000, 0.500] },
            KPoint { label: "R", coords: [0.500, 0.000, 0.500] },
            KPoint { label: "A", coords: [0.500, 0.500, 0.500] },
        ], vec![0,1,2,0,3,4,5,3,1,4,2,5,0]),

        CrystalSystem::Orthorhombic => (vec![
            KPoint { label: "Γ", coords: [0.000, 0.000, 0.000] },
            KPoint { label: "X", coords: [0.500, 0.000, 0.000] },
            KPoint { label: "Y", coords: [0.000, 0.500, 0.000] },
            KPoint { label: "Z", coords: [0.000, 0.000, 0.500] },
            KPoint { label: "S", coords: [0.500, 0.500, 0.000] },
            KPoint { label: "T", coords: [0.000, 0.500, 0.500] },
            KPoint { label: "U", coords: [0.500, 0.000, 0.500] },
            KPoint { label: "R", coords: [0.500, 0.500, 0.500] },
        ], vec![0,1,4,2,0,3,6,7,5,3,1,6,2,5,0]),

        CrystalSystem::Monoclinic => (vec![
            KPoint { label: "Γ", coords: [0.000, 0.000, 0.000] },
            KPoint { label: "Y", coords: [0.500, 0.000, 0.000] },
            KPoint { label: "M", coords: [0.500, 0.500, 0.000] },
            KPoint { label: "Z", coords: [0.000, 0.000, 0.500] },
            KPoint { label: "B", coords: [0.000, 0.500, 0.000] },
            KPoint { label: "D", coords: [0.500, 0.000, 0.500] },
        ], vec![0,1,2,4,0,3,5,0]),

        CrystalSystem::Triclinic => (vec![
            KPoint { label: "Γ", coords: [0.000, 0.000, 0.000] },
            KPoint { label: "X", coords: [0.500, 0.000, 0.000] },
            KPoint { label: "Y", coords: [0.000, 0.500, 0.000] },
            KPoint { label: "Z", coords: [0.000, 0.000, 0.500] },
            KPoint { label: "R", coords: [0.500, 0.500, 0.500] },
            KPoint { label: "V", coords: [0.500, 0.500, 0.000] },
        ], vec![0,1,4,2,0,3,4,5,0]),
    };
    KpathWalker::new(pts, path)
}
