// ── Fragment-shader dispatch ─────────────────────────────────────────────
// Switches on `u.mode` and calls the matching `render_<name>(uv)` function.
// Adding a mode = add a new file + a case here + bump MODE_NAMES in main.rs.

@fragment fn fs_field(f: VOut) -> @location(0) vec4<f32> {
    let uv = (f.uv*2.0 - 1.0) * vec2<f32>(u.aspect, 1.0);
    var col: vec3<f32>;
    switch u.mode {
        // ── core modes (0..11) ────────────────────────────────────────────
        case 0u  { col = render_rm(uv); }
        case 1u  { col = render_bz(uv); }
        case 2u  { col = render_fermi(uv); }
        case 3u  { col = render_density(uv); }
        case 4u  { col = render_nodal(uv); }
        case 5u  { col = render_phase(uv); }
        case 6u  { col = render_stripes(uv); }
        case 7u  { col = render_warp(uv); }
        case 8u  { col = render_links(uv); }
        case 9u  { col = render_xrd(uv); }
        case 10u { col = render_recip3d(uv); }
        case 11u { col = render_noneuclidean(uv); }
        // ── derived modes (12..27) ────────────────────────────────────────
        case 12u { col = render_phonon(uv); }
        case 13u { col = render_moire(uv); }
        case 14u { col = render_ewald(uv); }
        case 15u { col = render_wannier(uv); }
        case 16u { col = render_magnetic(uv); }
        case 17u { col = render_dispersion(uv); }
        case 18u { col = render_kikuchi(uv); }
        case 19u { col = render_defect(uv); }
        case 20u { col = render_band_surface(uv); }
        case 21u { col = render_spin_texture(uv); }
        case 22u { col = render_bz_path(uv); }
        case 23u { col = render_cdw(uv); }
        case 24u { col = render_quasicrystal(uv); }
        case 25u { col = render_thermal(uv); }
        case 26u { col = render_domain_wall(uv); }
        case 27u { col = render_fracture(uv); }
        // ── newer batch (28..35) ──────────────────────────────────────────
        case 28u { col = render_berry(uv); }
        case 29u { col = render_hofstadter(uv); }
        case 30u { col = render_stm(uv); }
        case 31u { col = render_spectral(uv); }
        case 32u { col = render_vortex_knot(uv); }
        case 33u { col = render_bloch_wave(uv); }
        case 34u { col = render_plasmon(uv); }
        case 35u { col = render_nematic(uv); }
        case 36u { col = render_abrikosov(uv); }
        case 37u { col = render_crystal_dive(uv); }
        case 38u { col = render_nano_phase(uv); }
        case 39u { col = render_magnon(uv); }
        // ── physics-experts batch (40..44) ────────────────────────────────
        case 40u { col = render_wavepacket(uv); }
        case 41u { col = render_dipole_rad(uv); }
        case 42u { col = render_karman(uv); }
        case 43u { col = render_lorenz(uv); }
        case 44u { col = render_lensing(uv); }
        case 45u { col = render_grav_wave(uv); }
        default  { col = render_crystal_dive(uv); }
    }
    // vignette — skip for XRD, RECIP3D, KIKUCHI, HOFSTADTER, SPECTRAL (own boundaries)
    if u.mode != 9u && u.mode != 10u && u.mode != 18u && u.mode != 29u && u.mode != 31u {
        col *= 1.0 - 0.35*dot(f.uv*2.0-1.0, f.uv*2.0-1.0);
    }
    // tone-map — skip for XRD and HOFSTADTER (preserves sharp features)
    if u.mode != 9u && u.mode != 29u {
        col = col / (col + vec3<f32>(0.6));
        col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(0.85));
    }
    return vec4<f32>(col, 1.0);
}
