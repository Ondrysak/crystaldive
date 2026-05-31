// Mode: LORENZ -- per-pixel Lorenz attractor with Lyapunov coloring.
//
// Physics: each fragment integrates the Lorenz 1963 ODE
//     dx/dt = sigma (y - x)
//     dy/dt = x (rho - z) - y
//     dz/dt = x y - beta z
// from an initial condition encoded by uv (x0, z0) and a y0 perturbed by the
// crystal_field. Alongside the orbit we co-integrate a normalized tangent
// vector to obtain a finite-time Lyapunov exponent lambda (sensitivity to
// initial conditions) -- the signature of deterministic chaos. Color comes
// from lobe membership (sign of x), the local divergence rate, and an
// accumulated trail of the trajectory in the (x, z) plane. Parameters drift
// slowly with u.time so the attractor breathes between regular and chaotic
// regimes (Hopf bifurcation at rho ~= 24.74).

// Per-mode free slots (see modes::mode_params for LORENZ). A slot left at 0 by
// the preset/random generators falls back to the canonical constant.
const LZ_SIGMA: u32 = 9u;
const LZ_RHO:   u32 = 10u;
const LZ_BETA:  u32 = 11u;
const LZ_DT:    u32 = 12u;
fn lz_slot(s: u32, def: f32) -> f32 { let v = mp(s); return select(def, v, v > 1e-4); }

fn lorenz_rhs(p: vec3<f32>, sigma: f32, rho: f32, beta: f32) -> vec3<f32> {
    return vec3<f32>(
        sigma * (p.y - p.x),
        p.x * (rho - p.z) - p.y,
        p.x * p.y - beta * p.z
    );
}

// J(p) . d for the variational equation,
//   J = [[-sigma,  sigma,  0   ],
//        [rho - z, -1,    -x  ],
//        [y,       x,     -beta]]
fn lorenz_jac_tan(p: vec3<f32>, d: vec3<f32>, sigma: f32, rho: f32, beta: f32) -> vec3<f32> {
    return vec3<f32>(
        sigma * (d.y - d.x),
        (rho - p.z) * d.x - d.y - p.x * d.z,
        p.y * d.x + p.x * d.y - beta * d.z
    );
}

// Cosine palette (Inigo Quilez style).
fn lorenz_palette(h: f32) -> vec3<f32> {
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.00, 0.10, 0.20);
    return a + b * cos(TAU * (c * h + d));
}

fn render_lorenz(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * mp(MP_SPEED);

    // Parameters: sigma, rho, beta drift between regimes.
    // rho ~= 28 is the canonical chaotic Lorenz; sweep across the Hopf
    // bifurcation at rho ~= 24.74 so tori-like orbits give way to the butterfly.
    let sigma = lz_slot(LZ_SIGMA, 10.0)   + 1.5 * sin(t * 0.07);
    let rho   = lz_slot(LZ_RHO,   28.0)   + 6.0 * sin(t * 0.05);
    let beta  = lz_slot(LZ_BETA,  2.6667) + 0.4 * sin(t * 0.09);

    // Phase-space window centred on attractor centroid (0, 0, rho).
    // Display the (x, z) projection: uv.x -> x, uv.y -> z (flipped so up = +z).
    let win  = 30.0 / max(mp(MP_ZOOM), 0.08);
    let x0   = uv.x / max(u.aspect, 0.0001) * 0.5 * win;
    let z0   = (-uv.y) * 0.5 * win + rho;
    // y0 from crystal_field: this is where the crystal imprints on the IC.
    let cf   = crystal_field(vec3<f32>(uv * 1.6, t * 0.2));
    let y0   = 0.35 * win * 0.6 * cf;

    var p   = vec3<f32>(x0, y0, z0);
    var dlt = normalize(vec3<f32>(1.0, 0.0, 0.0));
    var log_growth = 0.0;
    var renorms    = 0.0;

    let dt = lz_slot(LZ_DT, 0.006);

    var prev_sign  = sign(p.x);
    var lobe_swaps = 0.0;
    var trail      = vec3<f32>(0.0);
    let inv_steps  = 1.0 / 120.0;

    // RK4 + co-integrated tangent. 120 steps per pixel.
    for (var step = 0i; step < 120i; step = step + 1i) {
        let k1 = lorenz_rhs(p,                 sigma, rho, beta);
        let k2 = lorenz_rhs(p + 0.5 * dt * k1, sigma, rho, beta);
        let k3 = lorenz_rhs(p + 0.5 * dt * k2, sigma, rho, beta);
        let k4 = lorenz_rhs(p +       dt * k3, sigma, rho, beta);
        let pn = p + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

        let mid = 0.5 * (p + pn);
        let j1  = lorenz_jac_tan(p,   dlt,                 sigma, rho, beta);
        let j2  = lorenz_jac_tan(mid, dlt + 0.5 * dt * j1, sigma, rho, beta);
        let j3  = lorenz_jac_tan(mid, dlt + 0.5 * dt * j2, sigma, rho, beta);
        let j4  = lorenz_jac_tan(pn,  dlt +       dt * j3, sigma, rho, beta);
        var dn  = dlt + (dt / 6.0) * (j1 + 2.0 * j2 + 2.0 * j3 + j4);

        let dn_len = length(dn);
        if (dn_len > 1e-8) {
            log_growth = log_growth + log(dn_len);
            dn         = dn / dn_len;
            renorms    = renorms + 1.0;
        }
        dlt = dn;
        p   = pn;

        let s = sign(p.x);
        if (s * prev_sign < 0.0) {
            lobe_swaps = lobe_swaps + 1.0;
        }
        prev_sign = s;

        // Trail shimmer: hue follows lobe + height, weight grows with recency.
        let recency = f32(step) * inv_steps;
        let speed_n = length(k1) * 0.02;
        let hue     = fract(0.62 + 0.18 * s + 0.05 * p.z + mp(MP_COLOR_SHIFT) + t * 0.03);
        let shim    = 0.55 + 0.45 * sin(p.x * 0.6 + p.y * 0.45 + t * 1.7);
        trail = trail + lorenz_palette(hue) * (recency * recency) * shim
                * (0.35 + speed_n) * inv_steps;
    }

    // Finite-time Lyapunov exponent. Classical Lorenz: lambda_max ~= 0.906.
    let total_time = dt * max(renorms, 1.0);
    let lyap  = log_growth / max(total_time, 1e-4);
    let chaos = clamp(lyap * 0.6 + 0.2, 0.0, 1.6);

    // Compose.
    var col = vec3<f32>(0.010, 0.011, 0.022) + u.crystal_color.xyz * 0.04;

    let lobe_hue = fract(0.55 + 0.20 * prev_sign + mp(MP_COLOR_SHIFT));
    let lobe_col = lorenz_palette(lobe_hue);
    let swap_amt = clamp(lobe_swaps * 0.06, 0.0, 1.0);
    let body_col = mix(lobe_col, u.crystal_color.xyz, 0.30 + 0.35 * swap_amt);

    // Near a lobe centre -> bright. Lobe centres sit near z ~ rho, x ~ +- sqrt(beta(rho-1)).
    let lobe_off  = vec2<f32>(p.x, p.z - rho);
    let near_lobe = exp(-length(lobe_off) * 0.04);
    col = col + body_col * chaos * (0.40 + 1.30 * near_lobe);

    // Trail: feathered orbit signature.
    col = col + trail * 1.8;

    // Sensitivity halo where adjacent ICs already diverged strongly.
    let radial = length(p) * 0.02;
    col = col + u.crystal_color.xyz * smoothstep(0.5, 2.5, radial) * 0.25 * chaos;

    // Fixed points C+- = (+- sqrt(beta(rho-1)), +- sqrt(beta(rho-1)), rho-1).
    let xfp          = sqrt(max(beta * (rho - 1.0), 0.0));
    let zfp          = rho - 1.0;
    let inv_half_win = 1.0 / (0.5 * win);
    let uv_fp_z      = -(zfp - rho) * inv_half_win;
    let uv_fp_x      =  xfp * inv_half_win * max(u.aspect, 0.0001);
    let d1 = length(vec2<f32>(uv.x - uv_fp_x, uv.y - uv_fp_z));
    let d2 = length(vec2<f32>(uv.x + uv_fp_x, uv.y - uv_fp_z));
    let fp_glow = exp(-d1 * 60.0) + exp(-d2 * 60.0);
    col = col + fp_glow * vec3<f32>(1.0, 0.85, 0.50) * 0.55;

    // Content-aware soft inner-bright / outer-dim so the butterfly reads.
    let r = length(uv);
    col = col * (0.85 + 0.30 * (1.0 - smoothstep(0.4, 1.6, r)));

    // Mouse: held -> bright readout dot anchored at cursor.
    if (u.mouse_down >= 0.5) {
        let mp = (u.mouse - vec2<f32>(0.5, 0.5)) * vec2<f32>(2.0 * u.aspect, -2.0);
        let dm = length(uv - mp);
        col = col + exp(-dm * 30.0) * vec3<f32>(1.0, 0.95, 0.80) * 0.6;
    }

    return col;
}
