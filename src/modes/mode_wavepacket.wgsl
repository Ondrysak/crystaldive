// Mode: WAVEPACKET -- Free 2D Gaussian wave packet evolution (Schroedinger, hbar = m = 1)
//
// Closed-form solution for a free Gaussian wave packet centred at x0 with mean
// momentum k0 and initial width sigma0:
//
//   psi(x,t) = 1/sqrt(2*pi*(sigma0^2 + i*t/2))
//            * exp( i k0 . (p - x0) - i |k0|^2 t / 2 )
//            * exp( - (p - x0 - k0*t)^2 / ( 4 * (sigma0^2 + i*t/2) ) )
//
// |psi|^2 broadens as sigma(t) = sigma0 * sqrt(1 + (t/(2 sigma0^2))^2) while
// drifting at v_g = k0. Two coherent counter-propagating packets are super-
// posed -> spatial interference fringes appear where their envelopes overlap.

fn wavepacket_cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn wavepacket_cinv(z: vec2<f32>) -> vec2<f32> {
    let d = max(z.x * z.x + z.y * z.y, 1e-12);
    return vec2<f32>(z.x, -z.y) / d;
}

fn wavepacket_cexp(z: vec2<f32>) -> vec2<f32> {
    let r = exp(z.x);
    return vec2<f32>(r * cos(z.y), r * sin(z.y));
}

// Principal complex sqrt: w with w*w = z, Re(w) >= 0.
fn wavepacket_csqrt(z: vec2<f32>) -> vec2<f32> {
    let r  = sqrt(max(z.x * z.x + z.y * z.y, 1e-20));
    let re = sqrt(max((r + z.x) * 0.5, 0.0));
    var im = sqrt(max((r - z.x) * 0.5, 0.0));
    if (z.y < 0.0) { im = -im; }
    return vec2<f32>(re, im);
}

fn wavepacket_psi(
    p: vec2<f32>, t: f32,
    x0: vec2<f32>, k0: vec2<f32>, sigma0: f32
) -> vec2<f32> {
    // s2 = sigma0^2 + i t / 2     (hbar = m = 1)
    let s2 = vec2<f32>(sigma0 * sigma0, 0.5 * t);
    let inv_sqrt_s2 = wavepacket_cinv(wavepacket_csqrt(s2));

    // Displacement from drifting centroid x0 + k0 t.
    let d  = p - x0 - k0 * t;
    let d2 = dot(d, d);

    // Envelope arg: -d^2 / (4 s2).
    let denom4  = vec2<f32>(4.0 * s2.x, 4.0 * s2.y);
    let arg_env = wavepacket_cmul(vec2<f32>(-d2, 0.0), wavepacket_cinv(denom4));

    // Plane-wave phase: i ( k0 . (p - x0) - |k0|^2 t / 2 ).
    let phase     = dot(k0, p - x0) - 0.5 * dot(k0, k0) * t;
    let arg_phase = vec2<f32>(0.0, phase);

    let env = wavepacket_cexp(arg_env + arg_phase);
    return wavepacket_cmul(inv_sqrt_s2, env);
}

fn wavepacket_hue_shift(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k  = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

fn wavepacket_phase_color(phi: f32) -> vec3<f32> {
    let h = fract(phi / TAU + 1.0);
    return 0.5 + 0.5 * cos(TAU * (h + vec3<f32>(0.0, 0.333, 0.667)));
}

fn render_wavepacket(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed;

    // Spatial window in psi-units.
    let zoom = max(u.zoom, 0.05);
    let p    = uv * (4.0 / zoom);

    // Initial width sigma0 -- small => narrow packet, broadens quickly.
    let sigma0 = mix(0.35, 1.4, clamp(u.iso_level, 0.0, 1.0));

    // Mean momentum k0 -- magnitude from field_mix, direction precesses in time.
    let kmag   = mix(0.5, 3.8, clamp(u.field_mix, 0.0, 1.0));
    let kangle = t * 0.15 + u.color_shift * TAU;
    let k0     = vec2<f32>(cos(kangle), sin(kangle)) * kmag;

    // Cycle effective propagation time so packets recur; triangle in t, period ~14 s.
    let tcycle = (fract(t * 0.07) - 0.5) * 14.0;

    // Two symmetric starting positions. Mouse-held overrides for live throwing.
    var x0a = vec2<f32>(-2.0, -0.6);
    var x0b = vec2<f32>( 2.0,  0.6);
    if (u.mouse_down >= 0.5) {
        let mx = (u.mouse.x - 0.5) * 6.0 * max(u.aspect, 0.0001);
        let my = (0.5 - u.mouse.y) * 4.0;
        x0a = vec2<f32>(mx, my);
        x0b = -x0a;
    }

    // Counter-propagating momenta => collision + interference.
    let psi_a = wavepacket_psi(p, tcycle, x0a,  k0, sigma0);
    let psi_b = wavepacket_psi(p, tcycle, x0b, -k0, sigma0);

    let psi   = psi_a + psi_b;
    let prob  = dot(psi, psi);          // |psi|^2
    let phase = atan2(psi.y, psi.x);    // arg(psi)

    var col = vec3<f32>(0.005, 0.008, 0.020);
    col += u.crystal_color.xyz * 0.04;

    var pcol = wavepacket_phase_color(phase);
    pcol = mix(pcol, pcol * u.crystal_color.xyz, 0.20);
    pcol = wavepacket_hue_shift(pcol, u.color_shift);

    col += pcol * (prob * 2.5);

    // Real-part fringes -- emphasise interference where envelopes overlap.
    let re_psi  = psi.x;
    let fringes = re_psi * re_psi;
    col += vec3<f32>(0.95, 0.85, 1.05) * fringes * 0.45;

    // Node wisps -- Re(psi) zero-crossings inside the support of |psi|^2.
    let nodew = 0.04 + 0.10 * sigma0;
    let node  = smoothstep(nodew, 0.0, abs(re_psi)) * smoothstep(0.02, 0.4, prob);
    col += vec3<f32>(0.55, 0.65, 0.85) * node * 0.25;

    // Faint origin crosshair to orient the viewer in the psi-plane.
    let cx = smoothstep(0.012, 0.0, abs(p.x));
    let cy = smoothstep(0.012, 0.0, abs(p.y));
    let edge_a = smoothstep(2.6, 1.2, abs(p.y));
    let edge_b = smoothstep(2.6, 1.2, abs(p.x));
    col += (cx * edge_a + cy * edge_b) * vec3<f32>(0.10, 0.13, 0.18);

    // Soft outer halo keeps the frame alive between packet collisions.
    let halo = exp(-dot(p, p) * 0.05) * 0.05;
    col += halo * u.crystal_color.xyz;

    return col;
}
