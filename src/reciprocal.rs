//! Reciprocal lattice, G-vector construction, and GPU field packing.

use std::collections::HashMap;

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
#[derive(Clone)]
pub struct GpuField {
    pub hkls:   Vec<[i32; 3]>,
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

        let mut entries: Vec<([i32; 3], [f32; 3], f32)> = Vec::new();
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
                    if amp > 1e-5 { entries.push(([h, k, l], [gx, gy, gz], amp)); }
                }
            }
        }

        entries.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(MAX_G);

        let max_amp = entries.iter().map(|(_, _, a)| *a).fold(0f32, f32::max);
        let nrm = if max_amp > 0.0 { 1.0 / max_amp } else { 1.0 };

        let count = entries.len();
        let mut hkls   = Vec::with_capacity(count);
        let mut gvecs  = Vec::with_capacity(count);
        let mut amps   = Vec::with_capacity(count);
        let mut phases = Vec::with_capacity(count);
        for (hkl, g, a) in entries {
            hkls.push(hkl);
            gvecs.push(g);
            amps.push(a * nrm);
            phases.push(lcg_rand(&mut seed));
        }

        GpuField { hkls, gvecs, amps, phases, b_mat: b, count }
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
        self.pack_into(&mut d);
        d
    }

    pub fn pack_into(&self, out: &mut [f32]) {
        assert_eq!(out.len(), MAX_G * 4 * 2);
        out.fill(0.0);
        for i in 0..self.count {
            out[i * 4] = self.gvecs[i][0];
            out[i * 4 + 1] = self.gvecs[i][1];
            out[i * 4 + 2] = self.gvecs[i][2];
            out[i * 4 + 3] = self.amps[i];
            out[MAX_G * 4 + i * 4] = self.phases[i];
        }
    }
}

#[derive(Clone, Copy)]
struct MorphPoint {
    g: [f32; 3],
    amp: f32,
    phase: f32,
}

#[derive(Clone, Copy)]
struct MorphEntry {
    from: Option<MorphPoint>,
    to: Option<MorphPoint>,
}

/// Stable correspondence between two reciprocal fields.
///
/// Equal Miller indices morph physically: G moves with the reciprocal basis,
/// while structure-factor amplitude and phase interpolate continuously. If the
/// truncated top-128 sets differ, unmatched vectors pair by strength so both
/// endpoints remain exact without exceeding the shader's fixed MAX_G bank.
pub struct GpuFieldMorph {
    entries: Vec<MorphEntry>,
}

impl GpuFieldMorph {
    pub fn new(from: &GpuField, to: &GpuField) -> Self {
        let point = |field: &GpuField, i: usize| MorphPoint {
            g: field.gvecs[i],
            amp: field.amps[i],
            phase: field.phases[i],
        };
        let mut to_by_hkl = HashMap::with_capacity(to.count);
        for i in 0..to.count {
            to_by_hkl.insert(to.hkls[i], i);
        }

        // Preserve the source ordering. Modes that intentionally inspect the
        // first few vectors therefore remain stable at the start endpoint.
        let mut used_to = vec![false; to.count];
        let mut entries = Vec::with_capacity(from.count.max(to.count));
        for i in 0..from.count {
            let matched = to_by_hkl.get(&from.hkls[i]).copied();
            if let Some(j) = matched {
                used_to[j] = true;
            }
            entries.push(MorphEntry {
                from: Some(point(from, i)),
                to: matched.map(|j| point(to, j)),
            });
        }

        // Reuse source-only slots for the strongest unmatched target vectors.
        let mut next_to = 0usize;
        for entry in &mut entries {
            if entry.to.is_some() {
                continue;
            }
            while next_to < to.count && used_to[next_to] {
                next_to += 1;
            }
            if next_to < to.count {
                entry.to = Some(point(to, next_to));
                used_to[next_to] = true;
                next_to += 1;
            }
        }
        for (i, used) in used_to.into_iter().enumerate() {
            if !used && entries.len() < MAX_G {
                entries.push(MorphEntry { from: None, to: Some(point(to, i)) });
            }
        }

        Self { entries }
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Fill a preallocated G-block pack without per-frame allocation.
    pub fn pack_into(&self, t: f32, out: &mut [f32]) {
        assert_eq!(out.len(), MAX_G * 4 * 2);
        out.fill(0.0);
        let t = t.clamp(0.0, 1.0);
        let lerp = |a: f32, b: f32| a + (b - a) * t;

        for (i, entry) in self.entries.iter().enumerate() {
            let (g, amp, phase) = match (entry.from, entry.to) {
                (Some(a), Some(b)) => {
                    let mut phase_delta = b.phase - a.phase;
                    if phase_delta > std::f32::consts::PI {
                        phase_delta -= std::f32::consts::TAU;
                    } else if phase_delta < -std::f32::consts::PI {
                        phase_delta += std::f32::consts::TAU;
                    }
                    (
                        [
                            lerp(a.g[0], b.g[0]),
                            lerp(a.g[1], b.g[1]),
                            lerp(a.g[2], b.g[2]),
                        ],
                        lerp(a.amp, b.amp),
                        a.phase + phase_delta * t,
                    )
                }
                (Some(a), None) => (a.g, a.amp * (1.0 - t), a.phase),
                (None, Some(b)) => (b.g, b.amp * t, b.phase),
                (None, None) => continue,
            };
            out[i * 4] = g[0];
            out[i * 4 + 1] = g[1];
            out[i * 4 + 2] = g[2];
            out[i * 4 + 3] = amp;
            out[MAX_G * 4 + i * 4] = phase;
        }
    }
}

/// Symmetric real-space deformation tensor ε. Reciprocal vectors transform by
/// G' = (I + ε)^(-T) G.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReciprocalDeformation {
    pub xx: f32,
    pub yy: f32,
    pub zz: f32,
    pub xy: f32,
    pub xz: f32,
    pub yz: f32,
}

/// Apply a real-space lattice deformation to an already packed reciprocal field.
/// Returns false when the deformation matrix is singular and leaves the pack
/// unchanged.
pub fn deform_pack_in_place(
    packed: &mut [f32],
    count: usize,
    strain: ReciprocalDeformation,
) -> bool {
    assert_eq!(packed.len(), MAX_G * 4 * 2);
    let f = [
        [1.0 + strain.xx, strain.xy, strain.xz],
        [strain.xy, 1.0 + strain.yy, strain.yz],
        [strain.xz, strain.yz, 1.0 + strain.zz],
    ];
    let det =
        f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1])
        - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0])
        + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0]);
    if det.abs() < 1e-4 {
        return false;
    }
    let inv_det = 1.0 / det;
    let inv = [
        [
            (f[1][1] * f[2][2] - f[1][2] * f[2][1]) * inv_det,
            (f[0][2] * f[2][1] - f[0][1] * f[2][2]) * inv_det,
            (f[0][1] * f[1][2] - f[0][2] * f[1][1]) * inv_det,
        ],
        [
            (f[1][2] * f[2][0] - f[1][0] * f[2][2]) * inv_det,
            (f[0][0] * f[2][2] - f[0][2] * f[2][0]) * inv_det,
            (f[0][2] * f[1][0] - f[0][0] * f[1][2]) * inv_det,
        ],
        [
            (f[1][0] * f[2][1] - f[1][1] * f[2][0]) * inv_det,
            (f[0][1] * f[2][0] - f[0][0] * f[2][1]) * inv_det,
            (f[0][0] * f[1][1] - f[0][1] * f[1][0]) * inv_det,
        ],
    ];

    let active = count.min(MAX_G);
    let mut max_amp = 0.0f32;
    for i in 0..active {
        let base = i * 4;
        let g = [packed[base], packed[base + 1], packed[base + 2]];
        // inv(F)^T · G: columns of inv(F) dot the old vector.
        let moved = [
            inv[0][0] * g[0] + inv[1][0] * g[1] + inv[2][0] * g[2],
            inv[0][1] * g[0] + inv[1][1] * g[1] + inv[2][1] * g[2],
            inv[0][2] * g[0] + inv[1][2] * g[1] + inv[2][2] * g[2],
        ];
        let old_g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        let new_g2 = moved[0] * moved[0] + moved[1] * moved[1] + moved[2] * moved[2];
        packed[base] = moved[0];
        packed[base + 1] = moved[1];
        packed[base + 2] = moved[2];
        packed[base + 3] *= (-0.04 * (new_g2 - old_g2)).exp().clamp(0.25, 4.0);
        max_amp = max_amp.max(packed[base + 3]);
    }
    if max_amp > 1.0 {
        let inv_max = 1.0 / max_amp;
        for i in 0..active {
            packed[i * 4 + 3] *= inv_max;
        }
    }
    true
}

pub const HETERO_ADD: u32 = 0;
pub const HETERO_PRODUCT: u32 = 1;
pub const HETERO_INTERFERENCE: u32 = 2;

#[derive(Clone, Copy, Debug)]
pub struct HeterostructureParams {
    /// Layer-B rotation about z, in radians.
    pub twist: f32,
    /// Relative real-space lattice mismatch. Positive expands B and contracts
    /// its reciprocal vectors.
    pub mismatch: f32,
    /// Layer-B real-space translation in Å; contributes G·Δr to its phase.
    pub shift: [f32; 2],
    /// Relative translation along z, in Å.
    pub separation: f32,
    pub coupling: f32,
    pub combination: u32,
}

pub struct GpuHeterostructure {
    a: GpuField,
    b: GpuField,
}

impl GpuHeterostructure {
    pub fn new(a: GpuField, b: GpuField) -> Self {
        Self { a, b }
    }

    fn layer_b(&self, i: usize, p: HeterostructureParams) -> ([f32; 3], f32) {
        let g = self.b.gvecs[i];
        let c = p.twist.cos();
        let s = p.twist.sin();
        let inv_mismatch = 1.0 / (1.0 + p.mismatch).clamp(0.5, 1.5);
        let moved = [
            (c * g[0] - s * g[1]) * inv_mismatch,
            (s * g[0] + c * g[1]) * inv_mismatch,
            g[2] * inv_mismatch,
        ];
        let phase = self.b.phases[i]
            + moved[0] * p.shift[0]
            + moved[1] * p.shift[1]
            + moved[2] * p.separation;
        (moved, phase)
    }

    fn store(
        out: &mut [f32],
        slot: usize,
        g: [f32; 3],
        amp: f32,
        phase: f32,
    ) {
        out[slot * 4] = g[0];
        out[slot * 4 + 1] = g[1];
        out[slot * 4 + 2] = g[2];
        out[slot * 4 + 3] = amp;
        out[MAX_G * 4 + slot * 4] = phase;
    }

    /// Build a layered Fourier field. Add preserves both peak sets; Product
    /// uses cos(a)cos(b)=½[cos(a+b)+cos(a-b)]; Interference retains both layers
    /// and adds their strongest difference-frequency moiré beats.
    pub fn pack_into(&self, p: HeterostructureParams, out: &mut [f32]) -> usize {
        assert_eq!(out.len(), MAX_G * 4 * 2);
        out.fill(0.0);
        let coupling = p.coupling.clamp(0.0, 2.0);
        let mut slot = 0usize;

        match p.combination % 3 {
            HETERO_PRODUCT => {
                let na = self.a.count.min(8);
                let nb = self.b.count.min(8);
                for i in 0..na {
                    for j in 0..nb {
                        let ga = self.a.gvecs[i];
                        let (gb, phase_b) = self.layer_b(j, p);
                        let amp = 0.5 * self.a.amps[i] * self.b.amps[j] * coupling;
                        Self::store(
                            out,
                            slot,
                            [ga[0] + gb[0], ga[1] + gb[1], ga[2] + gb[2]],
                            amp,
                            self.a.phases[i] + phase_b,
                        );
                        slot += 1;
                        Self::store(
                            out,
                            slot,
                            [ga[0] - gb[0], ga[1] - gb[1], ga[2] - gb[2]],
                            amp,
                            self.a.phases[i] - phase_b,
                        );
                        slot += 1;
                    }
                }
            }
            HETERO_INTERFERENCE => {
                let na = self.a.count.min(48);
                let nb = self.b.count.min(48);
                for i in 0..na {
                    Self::store(
                        out,
                        slot,
                        self.a.gvecs[i],
                        self.a.amps[i],
                        self.a.phases[i],
                    );
                    slot += 1;
                }
                for i in 0..nb {
                    let (g, phase) = self.layer_b(i, p);
                    Self::store(out, slot, g, self.b.amps[i] * coupling, phase);
                    slot += 1;
                }
                let beats = self.a.count.min(self.b.count).min(MAX_G - slot);
                for i in 0..beats {
                    let ga = self.a.gvecs[i];
                    let (gb, phase_b) = self.layer_b(i, p);
                    Self::store(
                        out,
                        slot,
                        [ga[0] - gb[0], ga[1] - gb[1], ga[2] - gb[2]],
                        (self.a.amps[i] * self.b.amps[i]).sqrt() * coupling * 0.65,
                        self.a.phases[i] - phase_b,
                    );
                    slot += 1;
                }
            }
            _ => {
                let na = self.a.count.min(MAX_G / 2);
                let nb = self.b.count.min(MAX_G - na);
                for i in 0..na {
                    Self::store(
                        out,
                        slot,
                        self.a.gvecs[i],
                        self.a.amps[i],
                        self.a.phases[i],
                    );
                    slot += 1;
                }
                for i in 0..nb {
                    let (g, phase) = self.layer_b(i, p);
                    Self::store(out, slot, g, self.b.amps[i] * coupling, phase);
                    slot += 1;
                }
            }
        }
        let max_amp = (0..slot)
            .map(|i| out[i * 4 + 3])
            .fold(0.0f32, f32::max);
        if max_amp > 1.0 {
            let inv_max = 1.0 / max_amp;
            for i in 0..slot {
                out[i * 4 + 3] *= inv_max;
            }
        }
        slot
    }
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
        assert_eq!(f.hkls.len(), f.count);
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
    fn morph_pack_preserves_endpoints_and_moves_matched_g_vectors() {
        let mut expanded = nacl();
        expanded.lattice = [[7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 7.0]];
        let mut a = GpuField::from_crystal(&nacl(), 3);
        let mut b = GpuField::from_crystal(&expanded, 3);
        a.seed_kpoint([0.0; 3], 1.0);
        b.seed_kpoint([0.0; 3], 1.0);

        let plan = GpuFieldMorph::new(&a, &b);
        let mut packed = vec![0.0; MAX_G * 4 * 2];
        plan.pack_into(0.0, &mut packed);
        assert_eq!(plan.count(), a.count.max(b.count));
        assert_eq!(&packed[0..3], &a.gvecs[0]);
        assert!((packed[3] - a.amps[0]).abs() < 1e-6);

        let target_idx = b.hkls.iter().position(|hkl| *hkl == a.hkls[0]).unwrap();
        plan.pack_into(1.0, &mut packed);
        assert_eq!(&packed[0..3], &b.gvecs[target_idx]);
        assert!((packed[3] - b.amps[target_idx]).abs() < 1e-6);

        plan.pack_into(0.5, &mut packed);
        for axis in 0..3 {
            let expected = (a.gvecs[0][axis] + b.gvecs[target_idx][axis]) * 0.5;
            assert!((packed[axis] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn reciprocal_deformation_applies_inverse_transpose_and_rejects_singular() {
        let field = GpuField::from_crystal(&nacl(), 3);
        let mut packed = field.pack();
        let original = packed.clone();
        assert!(deform_pack_in_place(
            &mut packed,
            field.count,
            ReciprocalDeformation {
                xx: 0.25,
                yy: 0.25,
                zz: 0.25,
                ..ReciprocalDeformation::default()
            },
        ));
        for axis in 0..3 {
            assert!((packed[axis] - original[axis] / 1.25).abs() < 1e-5);
        }

        let mut singular = original.clone();
        assert!(!deform_pack_in_place(
            &mut singular,
            field.count,
            ReciprocalDeformation {
                xx: -1.0,
                ..ReciprocalDeformation::default()
            },
        ));
        assert_eq!(singular, original);
    }

    #[test]
    fn heterostructure_modes_pack_bounded_distinct_fourier_fields() {
        let a = GpuField::from_crystal(&nacl(), 3);
        let b = a.clone();
        let b0 = b.gvecs[0];
        let field = GpuHeterostructure::new(a.clone(), b);
        let mut packed = vec![0.0; MAX_G * 4 * 2];
        let mut params = HeterostructureParams {
            twist: std::f32::consts::FRAC_PI_2,
            mismatch: 0.0,
            shift: [0.0; 2],
            separation: 0.0,
            coupling: 1.0,
            combination: HETERO_ADD,
        };

        let add_count = field.pack_into(params, &mut packed);
        let a_count = a.count.min(MAX_G / 2);
        assert_eq!(add_count, a_count + a.count.min(MAX_G - a_count));
        let b_slot = a_count * 4;
        assert!((packed[b_slot] + b0[1]).abs() < 1e-5);
        assert!((packed[b_slot + 1] - b0[0]).abs() < 1e-5);

        params.combination = HETERO_PRODUCT;
        let product_count = field.pack_into(params, &mut packed);
        assert_eq!(product_count, a.count.min(8) * a.count.min(8) * 2);
        assert!(product_count <= MAX_G);

        params.combination = HETERO_INTERFERENCE;
        let interference_count = field.pack_into(params, &mut packed);
        assert!(interference_count <= MAX_G);
        assert!(interference_count > a.count.min(48) * 2);
        for i in 0..interference_count {
            assert!(packed[i * 4 + 3] <= 1.0);
        }
    }
}
