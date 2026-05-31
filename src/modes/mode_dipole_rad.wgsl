// Mode: DIPOLE - Hertzian electric-dipole radiation with retarded phase.
//
// Physics: oscillating point dipole p(t)=p0*cos(omega*t) zhat at origin
// emits E_theta with three terms in retarded time tau = t - r/c:
//
//   E_theta = sinTheta * [ (omega^2/(c^2 r)) * cos(omega*tau)   far field 1/r
//                        + (omega/(c r^2))   * sin(omega*tau)   induction 1/r^2
//                        - (1/r^3)           * cos(omega*tau) ] quasi-static 1/r^3
//
// Far field gives the classic sin^2(theta) donut and spherical wavefronts;
// near-field 1/r^2 and 1/r^3 dominate inside lambda/(2*pi) and are 90 deg
// out of phase, producing the swirling loops near the source.
//
// Visual: xz-plane cross-section. Red = +E_theta, blue = -E_theta.
// White isophase contours = outgoing wavefronts. Pulsing green arrow = p(t).

fn dipole_rad_sign_color(e: f32) -> vec3<f32> {
    let a = clamp(abs(e), 0.0, 6.0);
    let warm = vec3<f32>(1.00, 0.32, 0.18);
    let cool = vec3<f32>(0.18, 0.45, 1.10);
    let tint = select(cool, warm, e >= 0.0);
    return tint * (1.0 - exp(-a * 0.55));
}

fn dipole_rad_hue_shift(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k  = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h * TAU);
    let sn = sin(h * TAU);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

// Returns vec3(E_far, E_ind, E_near). Natural units c=1, omega=w.
fn dipole_rad_E_terms(r: f32, sin_theta: f32, w: f32, tau: f32) -> vec3<f32> {
    let r_safe = max(r, 0.02);
    let kr     = w * tau;
    let c_kr   = cos(kr);
    let s_kr   = sin(kr);
    let inv_r  = 1.0 / r_safe;
    let inv_r2 = inv_r * inv_r;
    let inv_r3 = inv_r2 * inv_r;
    let pref   = sin_theta;
    let e_far  =  pref * (w * w) * inv_r  * c_kr;
    let e_ind  =  pref * w       * inv_r2 * s_kr;
    let e_near = -pref           * inv_r3 * c_kr;
    return vec3<f32>(e_far, e_ind, e_near);
}

fn render_dipole_rad(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * mp(MP_SPEED);

    // xz-plane cross-section; dipole axis is vertical (p.y).
    let z = 1.0 / max(mp(MP_ZOOM) * 0.32, 0.05);
    let p = uv * z;

    let r          = length(p);
    let sin_signed = p.x / max(r, 0.001);
    let sin_abs    = abs(sin_signed);

    // field_mix slider -> angular frequency.
    let w   = 1.6 + 4.4 * mp(MP_FIELD_MIX);
    let tau = t - r;

    let terms  = dipole_rad_E_terms(r, sin_signed, w, tau);
    let e_far  = terms.x;
    let e_ind  = terms.y;
    let e_near = terms.z;

    // iso_level: 0 -> pure far-field, 1 -> include near-field loops.
    let near_w  = mp(MP_ISO_LEVEL);
    let e_total = e_far + near_w * (e_ind + e_near);

    // Background.
    var col = vec3<f32>(0.005, 0.008, 0.018);
    col += u.crystal_color.xyz * 0.04;

    // Half-wavelength radial guide rings, equatorial belt only.
    let lambda = TAU / w;
    let r_mod  = abs(fract(r / lambda - 0.5) - 0.5) * 2.0;
    let circle_glow = exp(-r_mod * 18.0) * 0.10 * smoothstep(0.4, 0.6, sin_abs);
    col += vec3<f32>(0.45, 0.55, 0.75) * circle_glow;

    // Bipolar field color.
    let e_scaled  = e_total * 0.45;
    var field_col = dipole_rad_sign_color(e_scaled);
    field_col     = dipole_rad_hue_shift(field_col, mp(MP_COLOR_SHIFT));
    let dist_fade = 1.0 / (1.0 + r * 0.25);
    col += field_col * dist_fade * 1.4;

    // Poynting flux highlight (far-field term only).
    let s_flux   = e_far * e_far;
    let flux_col = mix(vec3<f32>(1.0, 0.85, 0.40), u.crystal_color.xyz, 0.25);
    col += flux_col * s_flux * 0.18;

    // Isophase wavefront contours where cos(omega*tau) crosses zero.
    let phase      = w * tau;
    let phase_wrap = abs(fract(phase / TAU + 0.25) - 0.5) * 2.0;
    let wave_mask  = smoothstep(lambda * 0.25, lambda * 0.5, r)
                   * smoothstep(0.0, 0.35, sin_abs)
                   * (1.0 / (1.0 + r * 0.18));
    let wave_line  = smoothstep(0.04, 0.0, phase_wrap);
    col += vec3<f32>(0.85, 0.92, 1.00) * wave_line * wave_mask * 0.30;

    // sin^2(theta) donut envelope at ~2 wavelengths.
    let envelope = sin_abs * sin_abs;
    let env_lobe = exp(-pow(r - 2.0 * lambda, 2.0) * 1.5) * envelope * 0.25;
    col += vec3<f32>(0.95, 0.75, 0.30) * env_lobe;

    // Faint dipole axis line.
    let axis_d = abs(p.x);
    let axis   = exp(-axis_d * 28.0) * smoothstep(2.5, 0.5, abs(p.y)) * 0.08;
    col += vec3<f32>(0.30, 0.40, 0.55) * axis;

    // Pulsing green dipole-moment arrow along z.
    let p_amp     = cos(w * t);
    let arrow_len = 0.20 * p_amp;
    let lo        = min(0.0, arrow_len);
    let hi        = max(0.0, arrow_len);
    let seg_y     = clamp(p.y, lo, hi);
    let seg_d     = length(vec2<f32>(p.x, p.y - seg_y));
    let arrow     = smoothstep(0.045, 0.0, seg_d)
                  * smoothstep(0.03, 0.10, abs(arrow_len));
    col += vec3<f32>(0.45, 1.05, 0.55) * arrow * 1.6;

    // Source nucleus.
    let core = exp(-r * 38.0);
    col += vec3<f32>(1.00, 0.95, 0.70) * core * (0.6 + 0.4 * abs(p_amp)) * 1.2;

    // Faint crystal-field whisper in the background.
    let cf = crystal_field(vec3<f32>(p.x * 0.4, p.y * 0.4, t * 0.05));
    col += vec3<f32>(0.05, 0.07, 0.12) * cf * 0.5;

    return col;
}
