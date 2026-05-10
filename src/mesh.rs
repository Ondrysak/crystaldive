use bytemuck::{Pod, Zeroable};

/// Single vertex: unit-sphere position (= outward normal).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal:   [f32; 3],
}

// ── UV sphere ──────────────────────────────────────────────────────────────

pub fn uv_sphere(stacks: u32, slices: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut idx   = Vec::new();

    for i in 0..=stacks {
        let phi = std::f32::consts::PI * i as f32 / stacks as f32;
        for j in 0..=slices {
            let theta = 2.0 * std::f32::consts::PI * j as f32 / slices as f32;
            let x = phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();
            verts.push(Vertex { position: [x, y, z], normal: [x, y, z] });
        }
    }
    for i in 0..stacks {
        for j in 0..slices {
            let a = i * (slices + 1) + j;
            let b = a + slices + 1;
            idx.extend_from_slice(&[a, b, a + 1, b, b + 1, a + 1]);
        }
    }
    (verts, idx)
}

// ── Octahedron (flat-shaded — each face has its own verts/normals) ─────────

pub fn octahedron() -> (Vec<Vertex>, Vec<u32>) {
    // 6 base positions on axes
    let px = [ 1., 0., 0.];
    let nx = [-1., 0., 0.];
    let py = [ 0., 1., 0.];
    let ny = [ 0.,-1., 0.];
    let pz = [ 0., 0., 1.];
    let nz = [ 0., 0.,-1.];

    // 8 faces: (v0, v1, v2, outward_normal)  — winding verified CCW from outside
    let faces: [([f32;3], [f32;3], [f32;3]); 8] = [
        (px, py, pz),
        (px, pz, ny),
        (px, nz, py),
        (px, ny, nz),
        (nx, pz, py),
        (nx, py, nz),
        (nx, ny, pz),
        (nx, nz, ny),
    ];

    let inv_sqrt3 = 1.0_f32 / 3.0_f32.sqrt();
    let mut verts = Vec::with_capacity(24);
    let mut idx   = Vec::with_capacity(24);

    for (i, (a, b, c)) in faces.iter().enumerate() {
        // Face normal = average of three vertices (all unit-axis vectors)
        let nx = (a[0] + b[0] + c[0]) * inv_sqrt3;
        let ny = (a[1] + b[1] + c[1]) * inv_sqrt3;
        let nz = (a[2] + b[2] + c[2]) * inv_sqrt3;
        let n = [nx, ny, nz];

        let base = (i * 3) as u32;
        verts.push(Vertex { position: *a, normal: n });
        verts.push(Vertex { position: *b, normal: n });
        verts.push(Vertex { position: *c, normal: n });
        idx.extend_from_slice(&[base, base + 1, base + 2]);
    }
    (verts, idx)
}

// ── Cube (flat-shaded, 6 faces × 4 verts) ─────────────────────────────────

pub fn cube_mesh() -> (Vec<Vertex>, Vec<u32>) {
    // Each face: center position normal and 4 CCW vertices
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +X
        ([ 1., 0., 0.], [[ 1., 1., 1.], [ 1.,-1., 1.], [ 1.,-1.,-1.], [ 1., 1.,-1.]]),
        // -X
        ([-1., 0., 0.], [[-1., 1.,-1.], [-1.,-1.,-1.], [-1.,-1., 1.], [-1., 1., 1.]]),
        // +Y
        ([ 0., 1., 0.], [[-1., 1., 1.], [ 1., 1., 1.], [ 1., 1.,-1.], [-1., 1.,-1.]]),
        // -Y
        ([ 0.,-1., 0.], [[-1.,-1.,-1.], [ 1.,-1.,-1.], [ 1.,-1., 1.], [-1.,-1., 1.]]),
        // +Z
        ([ 0., 0., 1.], [[-1., 1., 1.], [-1.,-1., 1.], [ 1.,-1., 1.], [ 1., 1., 1.]]),
        // -Z
        ([ 0., 0.,-1.], [[ 1., 1.,-1.], [ 1.,-1.,-1.], [-1.,-1.,-1.], [-1., 1.,-1.]]),
    ];

    // Normalise cube verts to unit sphere radius (√3 diagonal → 1)
    let s = 1.0_f32 / 3.0_f32.sqrt();

    let mut verts = Vec::with_capacity(24);
    let mut idx   = Vec::with_capacity(36);

    for (n, corners) in &faces {
        let base = verts.len() as u32;
        for c in corners {
            verts.push(Vertex { position: [c[0]*s, c[1]*s, c[2]*s], normal: *n });
        }
        // Two triangles per quad (CCW)
        idx.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }
    (verts, idx)
}
