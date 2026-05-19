// ── Mode: LENSING — Schwarzschild gravitational lensing of a procedural sky ──────
//
// Physics
//   A non-rotating (Schwarzschild) black hole of mass M deflects light. For an
//   impact parameter b, the weak-field deflection is α = 4GM/(c² b) = 2rₛ/b.
//   We use units rₛ = 1 (so M = 1/2 in geometric units) and a Bozza-style
//   monotonic interpolation that produces a sharp photon-sphere shadow at the
//   critical impact parameter b_c = (3√3/2) rₛ ≈ 2.598 rₛ, plus an Einstein-
//   ring brightening near the caustic.
//
//   On top of the shadow we draw a thin equatorial accretion disk
//   (r ∈ [r_ISCO, r_out], r_ISCO = 6M = 3 rₛ for Schwarzschild) with
//   relativistic Doppler beaming D = 1/(γ(1 − β·n̂)), boost ∝ D⁴ (Iν/ν³
//   invariant + bolometric integration), plus gravitational redshift
//   √(1 − rₛ/r). The disk near-side passes in front of the BH and a secondary
//   image is lensed over the top of the shadow — the Gargantua look.
//
// Uniform mapping
//   field_mix   → camera elevation above the disk plane (0 = edge-on)
//   iso_level   → disk emissivity strength
//   color_shift → hue of the disk emission
//   zoom        → optical magnification (camera FOV)
//   speed       → orbital frequency of the disk + camera drift
//   mouse       → orbit camera (azimuth, elevation) when held down

const LENS_RS:  f32 = 1.0;           // Schwarzschild radius (units)
const LENS_BC:  f32 = 2.59807621;    // (3√3/2)·rₛ — critical impact parameter
const LENS_RIN: f32 = 3.0;           // r_ISCO = 6M = 3 rₛ
const LENS_ROU: f32 = 10.0;          // disk outer radius

fn lensing_hash(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(123.34, 456.21));
    let r = q + dot(q, q + 78.233);
    return fract(r.x * r.y);
}

fn lensing_noise3(p: vec3<f32>) -> f32 {
    let q = p * 3.0;
    let i = floor(q);
    let f = fract(q);
    let w = f * f * (3.0 - 2.0 * f);
    let a = lensing_hash(i.xy + i.z * 17.0);
    let b = lensing_hash(i.xy + vec2<f32>(1.0, 0.0) + i.z * 17.0);
    let c = lensing_hash(i.xy + vec2<f32>(0.0, 1.0) + i.z * 17.0);
    let d = lensing_hash(i.xy + vec2<f32>(1.0, 1.0) + i.z * 17.0);
    let ab = mix(a, b, w.x);
    let cd = mix(c, d, w.x);
    return mix(ab, cd, w.y);
}

fn lensing_sky(dir: vec3<f32>) -> vec3<f32> {
    let neb_n = lensing_noise3(dir * 2.0 + vec3<f32>(0.0, u.time * u.speed * 0.02, 0.0));
    let cf    = 0.5 + 0.5 * crystal_field(dir * 4.0);
    var neb   = vec3<f32>(0.025, 0.020, 0.055) + 0.18 * u.crystal_color.xyz * cf * neb_n;
    neb += vec3<f32>(0.10, 0.04, 0.18) * pow(neb_n, 3.0);

    let theta = atan2(dir.z, dir.x);
    let phi   = asin(clamp(dir.y, -1.0, 1.0));
    let g     = vec2<f32>(theta, phi) * vec2<f32>(80.0, 80.0);
    let gi    = floor(g);
    let gf    = fract(g) - 0.5;
    let h1    = lensing_hash(gi);
    let h2    = lensing_hash(gi + 11.7);
    let starx = (h1 - 0.5) * 0.7;
    let stary = (h2 - 0.5) * 0.7;
    let d     = length(gf - vec2<f32>(starx, stary));
    let mag   = smoothstep(0.06, 0.0, d) * step(0.94, lensing_hash(gi + 5.31));
    let twk   = 0.7 + 0.3 * sin(u.time * 2.0 + h1 * 50.0);
    var star  = mix(vec3<f32>(1.0, 0.95, 0.85), vec3<f32>(0.7, 0.8, 1.0), h2);
    star = star * mag * twk * 2.4;
    return neb + star;
}

fn lensing_alpha(b: f32) -> f32 {
    let bc = LENS_BC;
    let weak   = 2.0 * LENS_RS / b;
    let strong = -log((b - bc) / bc) - 0.4002;
    let blend  = smoothstep(bc, bc * 1.6, b);
    return mix(strong, weak, blend);
}

fn lensing_hue_shift(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k  = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

fn lensing_disk_emiss(r: f32) -> f32 {
    if (r < LENS_RIN || r > LENS_ROU) { return 0.0; }
    let x    = (r - LENS_RIN) / (LENS_ROU - LENS_RIN);
    let peak = exp(-pow((x - 0.08) * 4.0, 2.0));
    let tail = 1.0 / (1.0 + (r - LENS_RIN) * 0.6);
    return peak * 0.7 + tail * 0.5;
}

fn lensing_disk_image(cam_pos: vec3<f32>, ray_dir: vec3<f32>, n_image: i32) -> vec4<f32> {
    var cp = cam_pos;
    var rd = ray_dir;
    if (n_image == 1) {
        cp.y = -cp.y;
        rd.y = -rd.y;
    }
    if (abs(rd.y) < 1e-4) { return vec4<f32>(0.0); }
    let t = -cp.y / rd.y;
    if (t < 0.0) { return vec4<f32>(0.0); }
    let hit = cp + rd * t;
    let r   = length(hit.xz);
    let eps = lensing_disk_emiss(r);
    if (eps <= 0.0) { return vec4<f32>(0.0); }

    let v_phi  = sqrt(0.5 / max(r, 0.01));
    let phi    = atan2(hit.z, hit.x);
    let tang   = vec3<f32>(-sin(phi), 0.0, cos(phi));
    let to_obs = normalize(cp - hit);
    let cos_th = dot(tang, to_obs);
    let beta   = v_phi;
    let gamma  = 1.0 / sqrt(max(1.0 - beta * beta, 1e-4));
    let dop    = 1.0 / (gamma * (1.0 - beta * cos_th));
    let grav   = sqrt(max(1.0 - LENS_RS / r, 1e-4));
    let boost  = pow(dop * grav, 4.0);

    let omega  = sqrt(0.5 / max(r * r * r, 1e-3));
    let t_phys = u.time * u.speed;
    let arm    = 0.5 + 0.5 * cos(phi * 2.0 - omega * t_phys * 8.0 + r * 0.9);
    let turb   = 0.6 + 0.4 * crystal_field(hit * 0.6 + vec3<f32>(0.0, t_phys, 0.0));

    let hot  = mix(vec3<f32>(0.30, 0.55, 1.00), vec3<f32>(1.00, 0.95, 0.75),
                  smoothstep(LENS_RIN, LENS_RIN + 1.5, r));
    let cool = mix(vec3<f32>(1.00, 0.55, 0.20), vec3<f32>(0.55, 0.15, 0.10),
                  smoothstep(LENS_RIN + 1.5, LENS_ROU, r));
    var col  = mix(hot, cool, smoothstep(LENS_RIN + 1.0, LENS_RIN + 3.0, r));
    col = lensing_hue_shift(col, u.color_shift);
    let iso_boost = mix(0.6, 1.8, u.iso_level);
    col *= eps * boost * iso_boost * (0.55 + 0.55 * arm) * turb;
    if (n_image == 1) { col *= 0.55; }

    let edge = smoothstep(LENS_RIN - 0.05, LENS_RIN + 0.15, r)
             * (1.0 - smoothstep(LENS_ROU - 0.6, LENS_ROU, r));
    return vec4<f32>(col, edge);
}

fn render_lensing(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed;

    var az = t * 0.07;
    var el = mix(0.05, 0.55, u.field_mix);
    if (u.mouse_down >= 0.5) {
        az = u.mouse.x * TAU;
        el = mix(0.02, 1.1, clamp(u.mouse.y, 0.0, 1.0));
    }
    let cam_dist = 18.0;
    let cam_pos  = vec3<f32>(cos(az) * cos(el), sin(el), sin(az) * cos(el)) * cam_dist;

    let fwd      = normalize(-cam_pos);
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right    = normalize(cross(fwd, world_up));
    let up       = cross(right, fwd);

    let fov = 1.0 / max(u.zoom, 0.2);
    let ray = normalize(fwd + right * uv.x * fov + up * uv.y * fov);

    let cd  = dot(cam_pos, ray);
    let b2  = max(dot(cam_pos, cam_pos) - cd * cd, 1e-6);
    let b   = sqrt(b2);

    if (b <= LENS_BC) {
        return vec3<f32>(0.0);
    }

    let alpha = lensing_alpha(b);

    let n_ax = normalize(cross(cam_pos, ray));
    let tang = normalize(cross(n_ax, ray));
    let ca   = cos(alpha);
    let sa   = sin(alpha);
    let src_dir = normalize(ray * ca + tang * sa);

    var col = lensing_sky(src_dir);

    let ring_amp = 0.55 / sqrt(max(b - LENS_BC, 0.02));
    let ring_w   = exp(-pow((b - LENS_BC * 1.35) * 4.0, 2.0));
    col += ring_w * ring_amp * vec3<f32>(0.6, 0.75, 1.0) * 0.10;

    let d0 = lensing_disk_image(cam_pos, ray,     0);
    let d1 = lensing_disk_image(cam_pos, src_dir, 1);
    col = mix(col, d1.rgb, clamp(d1.a, 0.0, 1.0));
    col = mix(col, d0.rgb, clamp(d0.a, 0.0, 1.0));

    let pr_b = LENS_BC * 1.03;
    let pr_w = 0.025;
    let pr   = exp(-pow((b - pr_b) / pr_w, 2.0));
    col += pr * vec3<f32>(1.0, 0.85, 0.55) * 1.4;

    let glow = exp(-pow((b - LENS_BC) * 1.2, 2.0));
    col += glow * vec3<f32>(0.45, 0.30, 0.18) * 0.35;

    col += u.crystal_color.xyz * 0.015;

    return col;
}
