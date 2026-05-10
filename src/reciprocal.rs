//! Reciprocal lattice, G-vector construction, and GPU field packing.

use crate::poscar::Crystal;

pub const MAX_G: usize = 128;

fn lcg_rand(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 8) as f32 / 16777216.0 * std::f32::consts::TAU
}

fn element_z(sym: &str) -> f32 {
    let s: String = sym.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
    match s.as_str() {
        "H"  =>  1.0, "He" =>  2.0, "Li" =>  3.0, "Be" =>  4.0, "B"  =>  5.0,
        "C"  =>  6.0, "N"  =>  7.0, "O"  =>  8.0, "F"  =>  9.0, "Ne" => 10.0,
        "Na" => 11.0, "Mg" => 12.0, "Al" => 13.0, "Si" => 14.0, "P"  => 15.0,
        "S"  => 16.0, "Cl" => 17.0, "K"  => 19.0, "Ca" => 20.0, "Ti" => 22.0,
        "Fe" => 26.0, "Cu" => 29.0, "Zn" => 30.0, "Ga" => 31.0, "Ge" => 32.0,
        "As" => 33.0, "Se" => 34.0, "Br" => 35.0, "Ag" => 47.0, "I"  => 53.0,
        "Au" => 79.0, "Pb" => 82.0, _    =>  6.0,
    }
}

/// Reciprocal lattice: b_i = 2π (a_j × a_k) / V  (rows = b0, b1, b2 in 1/Å).
pub fn reciprocal_lattice(lat: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    }
    fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
    fn scl(v: [f32; 3], s: f32) -> [f32; 3] { [v[0]*s, v[1]*s, v[2]*s] }

    let (a0, a1, a2) = (lat[0], lat[1], lat[2]);
    let vol = dot3(a0, cross(a1, a2));
    let s = std::f32::consts::TAU / vol;
    [scl(cross(a1, a2), s), scl(cross(a2, a0), s), scl(cross(a0, a1), s)]
}

/// G-vectors + structure-factor amplitudes + phases, ready for GPU upload.
pub struct GpuField {
    pub gvecs:  Vec<[f32; 3]>,
    pub amps:   Vec<f32>,
    pub phases: Vec<f32>,
    pub b_mat:  [[f32; 3]; 3],
    pub count:  usize,
}

impl GpuField {
    pub fn from_crystal(crystal: &Crystal, max_shell: i32) -> Self {
        let b = reciprocal_lattice(&crystal.lattice);
        let mut seed = 0xdead_beef_u32;
        let total_z: f32 = crystal.atoms.iter()
            .map(|a| element_z(&a.species)).sum::<f32>().max(1.0);

        let mut entries: Vec<([f32; 3], f32)> = Vec::new();
        let range = max_shell + 1;

        for h in -range..=range {
            for k in -range..=range {
                for l in -range..=range {
                    let shell = h.abs().max(k.abs()).max(l.abs());
                    if shell == 0 || shell > max_shell { continue; }

                    let gx = h as f32*b[0][0] + k as f32*b[1][0] + l as f32*b[2][0];
                    let gy = h as f32*b[0][1] + k as f32*b[1][1] + l as f32*b[2][1];
                    let gz = h as f32*b[0][2] + k as f32*b[1][2] + l as f32*b[2][2];
                    let g2 = gx*gx + gy*gy + gz*gz;

                    let sg: f32 = crystal.atoms.iter().map(|a| {
                        let ph = gx*a.pos_cart[0] + gy*a.pos_cart[1] + gz*a.pos_cart[2];
                        element_z(&a.species) * ph.cos()
                    }).sum::<f32>() / total_z;

                    let amp = (-0.04 * g2).exp() * sg.abs();
                    if amp > 1e-5 { entries.push(([gx, gy, gz], amp)); }
                }
            }
        }

        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(MAX_G);

        let max_amp = entries.iter().map(|(_, a)| *a).fold(0f32, f32::max);
        let nrm = if max_amp > 0.0 { 1.0 / max_amp } else { 1.0 };

        let count = entries.len();
        let mut gvecs  = Vec::with_capacity(count);
        let mut amps   = Vec::with_capacity(count);
        let mut phases = Vec::with_capacity(count);
        for (g, a) in entries {
            gvecs.push(g);
            amps.push(a * nrm);
            phases.push(lcg_rand(&mut seed));
        }

        GpuField { gvecs, amps, phases, b_mat: b, count }
    }

    /// Set phases to Bloch phases at k-point `k_frac` (fractional reciprocal coords).
    /// `blend` = 1.0: instant snap; < 1.0: smooth angular interpolation.
    pub fn seed_kpoint(&mut self, k_frac: [f32; 3], blend: f32) {
        let b = &self.b_mat;
        let kx = k_frac[0]*b[0][0] + k_frac[1]*b[1][0] + k_frac[2]*b[2][0];
        let ky = k_frac[0]*b[0][1] + k_frac[1]*b[1][1] + k_frac[2]*b[2][1];
        let kz = k_frac[0]*b[0][2] + k_frac[1]*b[1][2] + k_frac[2]*b[2][2];
        for i in 0..self.count {
            let bloch  = self.gvecs[i][0]*kx + self.gvecs[i][1]*ky + self.gvecs[i][2]*kz;
            let target = bloch.rem_euclid(std::f32::consts::TAU);
            if blend >= 1.0 {
                self.phases[i] = target;
            } else {
                let mut d = target - self.phases[i];
                if d >  std::f32::consts::PI { d -= std::f32::consts::TAU; }
                if d < -std::f32::consts::PI { d += std::f32::consts::TAU; }
                self.phases[i] += d * blend;
            }
        }
    }

    pub fn randomize(&mut self) {
        let mut seed = 0x1234_5678_u32;
        for p in &mut self.phases { *p = lcg_rand(&mut seed); }
    }

    /// Pack into 2 × MAX_G × 4 floats for a Rgba32Float texture.
    /// Row 0 col i: (Gx, Gy, Gz, amp)   Row 1 col i: (phase, 0, 0, 0)
    pub fn pack(&self) -> Vec<f32> {
        let mut d = vec![0f32; MAX_G * 4 * 2];
        for i in 0..self.count {
            d[i*4+0] = self.gvecs[i][0];
            d[i*4+1] = self.gvecs[i][1];
            d[i*4+2] = self.gvecs[i][2];
            d[i*4+3] = self.amps[i];
            d[MAX_G*4 + i*4] = self.phases[i];
        }
        d
    }
}

/// Crossfade between two G-fields: interleaves top-half from `from` (scaled by 1-t)
/// and top-half from `to` (scaled by t) into a single MAX_G texture pack.
pub fn crossfade_pack(from: &GpuField, to: &GpuField, t: f32) -> Vec<f32> {
    let half = MAX_G / 2;
    let mut d = vec![0f32; MAX_G * 4 * 2];
    let fade_out = 1.0 - t;
    let fade_in  = t;

    // Slots 0..half  ← `from` field, amplitude faded by (1-t)
    for i in 0..half.min(from.count) {
        d[i*4+0] = from.gvecs[i][0];
        d[i*4+1] = from.gvecs[i][1];
        d[i*4+2] = from.gvecs[i][2];
        d[i*4+3] = from.amps[i] * fade_out;
        d[MAX_G*4 + i*4] = from.phases[i];
    }
    // Slots half..MAX_G  ← `to` field, amplitude faded by t
    for i in 0..half.min(to.count) {
        let slot = half + i;
        d[slot*4+0] = to.gvecs[i][0];
        d[slot*4+1] = to.gvecs[i][1];
        d[slot*4+2] = to.gvecs[i][2];
        d[slot*4+3] = to.amps[i] * fade_in;
        d[MAX_G*4 + slot*4] = to.phases[i];
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poscar::{Atom, Crystal};

    fn nacl() -> Crystal {
        Crystal {
            lattice: [[5.64, 0.0, 0.0], [0.0, 5.64, 0.0], [0.0, 0.0, 5.64]],
            atoms: vec![
                Atom { species: "Na".into(), pos_cart: [0.0, 0.0, 0.0] },
                Atom { species: "Cl".into(), pos_cart: [2.82, 2.82, 2.82] },
            ],
        }
    }

    #[test]
    fn reciprocal_cubic() {
        let lat = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let b = reciprocal_lattice(&lat);
        let expected = std::f32::consts::TAU / 2.0;
        assert!((b[0][0] - expected).abs() < 1e-4);
        assert!((b[1][1] - expected).abs() < 1e-4);
        assert!((b[2][2] - expected).abs() < 1e-4);
        assert!(b[0][1].abs() < 1e-4 && b[0][2].abs() < 1e-4);
    }

    #[test]
    fn gpu_field_construction() {
        let f = GpuField::from_crystal(&nacl(), 3);
        assert!(f.count > 0);
        assert!(f.count <= MAX_G);
        assert_eq!(f.gvecs.len(), f.count);
        assert_eq!(f.amps.len(), f.count);
        assert_eq!(f.phases.len(), f.count);
        // Amplitudes are normalized to ≤1.
        for &a in &f.amps { assert!(a >= 0.0 && a <= 1.0 + 1e-4, "amp out of range: {a}"); }
    }

    #[test]
    fn gpu_field_amps_sorted_descending() {
        let f = GpuField::from_crystal(&nacl(), 3);
        for w in f.amps.windows(2) {
            assert!(w[0] >= w[1] - 1e-5, "amps not sorted: {:?}", w);
        }
    }

    #[test]
    fn pack_size_matches_texture() {
        let f = GpuField::from_crystal(&nacl(), 3);
        let p = f.pack();
        assert_eq!(p.len(), MAX_G * 4 * 2);
    }

    #[test]
    fn seed_kpoint_at_gamma_zero_phase() {
        let mut f = GpuField::from_crystal(&nacl(), 3);
        f.seed_kpoint([0.0, 0.0, 0.0], 1.0);
        for &p in &f.phases {
            // rem_euclid(TAU) of 0 is 0.
            assert!(p.abs() < 1e-3, "Γ should give zero phase, got {p}");
        }
    }

    #[test]
    fn crossfade_pack_is_full_size() {
        let mut a = GpuField::from_crystal(&nacl(), 3);
        let mut b = GpuField::from_crystal(&nacl(), 3);
        a.seed_kpoint([0.0; 3], 1.0);
        b.seed_kpoint([0.5, 0.0, 0.5], 1.0);
        let mid = crossfade_pack(&a, &b, 0.5);
        assert_eq!(mid.len(), MAX_G * 4 * 2);
    }
}
