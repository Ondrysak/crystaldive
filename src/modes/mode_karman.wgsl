// ── Mode KARMAN — Kármán vortex street behind a bluff body ──────────────
//
// Physics: a 2-D incompressible viscous flow past a circular cylinder sheds
// counter-rotating vortices in a staggered (zig-zag) pattern that advect
// downstream at roughly 0.87·U∞. Each shed vortex is modelled here as a
// Lamb–Oseen kernel,
//
//     ω(r,t)  =  Γ / (π r_c²) · exp( -r² / r_c² ),
//
// with core radius r_c growing diffusively as r_c² = r_c0² + 4 ν t, and
// circulation Γ alternating in sign every half-period of the shedding
// frequency. The cylinder sits upstream and is rendered as an SDF disk
// with a thin viscous boundary layer; the freestream U∞ drifts vortices
// in +x while the crystal_field substrate is used as a subtle Reynolds-
// scale turbulence overlay (high-Re wrinkle on the vortex cores).
//
// Sliders:
//   iso_level   → Reynolds number / shedding rate (period spacing).
//   field_mix   → core radius (low Re = fat cores, high Re = tight cores).
//   color_shift → palette rotation around the (1,1,1) axis.
//   zoom        → spatial scale of the whole street.
//   speed       → freestream velocity U∞ (vortex drift rate).

fn karman_hue_rotate(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k  = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

// One Lamb–Oseen vortex contribution to vorticity ω at a point p.
fn karman_lamb_oseen_vort(p: vec2<f32>, c: vec2<f32>, gamma: f32, rc: f32) -> f32 {
    let d   = p - c;
    let r2  = dot(d, d);
    let rc2 = rc * rc;
    return (gamma / (3.14159265 * rc2)) * exp(-r2 / rc2);
}

// Induced velocity of one Lamb–Oseen vortex (Biot–Savart with viscous cutoff):
//   v = (Γ / 2π r) · (1 - exp(-r²/r_c²)) · t̂,
// where t̂ = (-dy, dx)/r is the tangential unit vector (CCW for Γ > 0).
fn karman_lamb_oseen_vel(p: vec2<f32>, c: vec2<f32>, gamma: f32, rc: f32) -> vec2<f32> {
    let d   = p - c;
    let r2  = dot(d, d) + 1e-6;
    let r   = sqrt(r2);
    let rc2 = rc * rc;
    let mag = (gamma / (TAU * r)) * (1.0 - exp(-r2 / rc2));
    return vec2<f32>(-d.y, d.x) * (mag / r);
}

fn render_karman(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed;

    // ── Flow-space coordinates ────────────────────────────────────────
    let scale = 2.4 / max(u.zoom, 0.05);
    let p     = uv * scale;

    // ── Reynolds-number / shedding control ───────────────────────────
    let Re = 60.0 + 240.0 * u.iso_level;
    let St = 0.18 + 0.04 * smoothstep(60.0, 300.0, Re);
    let cyl_diam = 0.55;
    let U_inf    = 1.0;
    let U_drift  = 0.87 * U_inf;
    let T_shed   = cyl_diam / max(St * U_inf, 1e-3);
    let spacing  = U_drift * T_shed;
    let half_h   = 0.28 * spacing;

    let cyl_c = vec2<f32>(-1.6, 0.0);

    // ── Build the vortex street ──────────────────────────────────────
    let rc0        = (0.10 + 0.35 * u.field_mix) * cyl_diam;
    let nu         = 0.0006 + 0.0015 / max(Re * 0.01, 0.5);
    let shed_phase = t / T_shed;

    var omega = 0.0;
    var vel   = vec2<f32>(0.0, 0.0);

    for (var k: i32 = 0; k < 22; k = k + 1) {
        let age_idx = f32(k);
        let n       = shed_phase - age_idx;
        let age_t   = age_idx * T_shed;

        let x = cyl_c.x + cyl_diam * 0.5 + U_drift * age_t;
        if (x > scale * 1.2) { continue; }

        let parity = (i32(floor(n)) & 1);
        let sign_f = select(-1.0, 1.0, parity == 0);
        let y      = sign_f * half_h;

        let rc    = sqrt(rc0 * rc0 + 4.0 * nu * age_t);
        let gamma = sign_f * 1.6 * exp(-age_t * 0.08);

        omega += karman_lamb_oseen_vort(p, vec2<f32>(x, y), gamma, rc);
        vel   += karman_lamb_oseen_vel(p, vec2<f32>(x, y), gamma, rc);
    }

    vel.x += U_inf;

    // ── Cylinder (bluff body) SDF ────────────────────────────────────
    let cyl_r  = cyl_diam * 0.5;
    let to_cyl = p - cyl_c;
    let d_cyl  = length(to_cyl) - cyl_r;
    let bl     = 0.10 / sqrt(max(Re * 0.05, 1.0));
    let in_cyl = smoothstep(0.0, -0.01, d_cyl);
    let bl_glow = exp(-max(d_cyl, 0.0) / bl) * (1.0 - in_cyl);

    // ── Crystal-field substrate: wrinkle vortex cores at high Re ─────
    let cf_p   = vec3<f32>(p.x * 0.8, p.y * 0.8, t * 0.06);
    let cf     = crystal_field(cf_p);
    let cf_mod = 0.85 + 0.30 * cf * smoothstep(0.0, 1.5, Re * 0.005);
    omega *= cf_mod;

    // ── Color: signed vorticity → diverging palette ──────────────────
    let omag = abs(omega);
    let osig = clamp(omega * 0.9, -2.5, 2.5);
    let warm = vec3<f32>(1.05, 0.55, 0.18);
    let cool = vec3<f32>(0.18, 0.55, 1.05);
    var vort_col = mix(cool, warm, smoothstep(-1.5, 1.5, osig));
    vort_col = mix(vort_col, vort_col * u.crystal_color.xyz, 0.15);
    let bright = omag / (omag + 0.6);

    // ── Background ───────────────────────────────────────────────────
    var col = vec3<f32>(0.010, 0.014, 0.030);
    col += u.crystal_color.xyz * 0.04;

    let speed_mag = length(vel);
    col += vec3<f32>(0.06, 0.08, 0.12) * smoothstep(0.0, 2.0, speed_mag);

    // ── Streamline ribbons (LIC-style proxy) ─────────────────────────
    let v_n = vel / max(length(vel), 1e-4);
    let stripe_coord = dot(p, vec2<f32>(v_n.y, -v_n.x)) * 18.0 - t * 4.0 * speed_mag;
    let stripe = 0.5 + 0.5 * sin(stripe_coord);
    let stripe_w = smoothstep(0.55, 1.0, stripe) * (1.0 - in_cyl);
    col += stripe_w * vec3<f32>(0.10, 0.14, 0.22) * 0.55
         * smoothstep(0.5, 2.5, speed_mag);

    // ── Main attraction: vorticity field ─────────────────────────────
    col += vort_col * bright * 1.35 * (1.0 - in_cyl);

    // ── Cylinder rendering ───────────────────────────────────────────
    let upstream_rim = exp(-max(d_cyl, 0.0) / (bl * 0.6))
                     * smoothstep(0.0, -0.4, to_cyl.x) * (1.0 - in_cyl);
    col = mix(col, vec3<f32>(0.05, 0.06, 0.09), in_cyl);
    col += upstream_rim * vec3<f32>(1.0, 0.92, 0.75) * 0.55;
    col += bl_glow     * vec3<f32>(0.95, 0.65, 0.30) * 0.35 * (1.0 - in_cyl);

    // ── Faint street-axis guide line ─────────────────────────────────
    let axis = smoothstep(0.006, 0.0, abs(p.y))
             * smoothstep(cyl_c.x + cyl_r, scale * 1.0, p.x);
    col += axis * vec3<f32>(0.18, 0.20, 0.26) * 0.35;

    col = karman_hue_rotate(col, u.color_shift);

    return col;
}
