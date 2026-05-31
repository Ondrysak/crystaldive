//! Crystal-field shader modes.
//!
//! The full WGSL fragment shader is assembled at compile time by concatenating
//! the prelude, the existing core modes, every per-mode file, and the dispatch
//! entry point. Adding a mode is purely additive: drop a new `mode_*.wgsl`
//! file, add an `include_str!` line below, add a case in `dispatch.wgsl`, and
//! bump `MODE_NAMES` in [`crate::main`].
//!
//! Each `mode_*.wgsl` file must define
//!     `fn render_<name>(uv: vec2<f32>) -> vec3<f32>`
//! and may freely use any helper from `prelude.wgsl`. WGSL function ordering
//! within a single module is flexible, so files can be concatenated in any
//! order. Keep mode-local helpers prefixed with the mode name to avoid
//! collisions.
//!
//! See `CONTRACT.md` in this directory for the full surface available to
//! mode authors.

pub mod info;
pub use info::{
    mode_params, slot_range, ModeArea, ModeInfo, ParamDesc, MODES, MODE_NAMES,
    MODE_NAMES_SLICE,
};

pub const FIELD_SHADER: &str = concat!(
    include_str!("prelude.wgsl"),
    "\n",
    include_str!("core_modes.wgsl"),
    "\n",
    include_str!("mode_phonon.wgsl"),
    "\n",
    include_str!("mode_moire.wgsl"),
    "\n",
    include_str!("mode_ewald.wgsl"),
    "\n",
    include_str!("mode_wannier.wgsl"),
    "\n",
    include_str!("mode_magnetic.wgsl"),
    "\n",
    include_str!("mode_dispersion.wgsl"),
    "\n",
    include_str!("mode_kikuchi.wgsl"),
    "\n",
    include_str!("mode_defect.wgsl"),
    "\n",
    include_str!("mode_band_surface.wgsl"),
    "\n",
    include_str!("mode_spin_texture.wgsl"),
    "\n",
    include_str!("mode_bz_path.wgsl"),
    "\n",
    include_str!("mode_cdw.wgsl"),
    "\n",
    include_str!("mode_quasicrystal.wgsl"),
    "\n",
    include_str!("mode_thermal.wgsl"),
    "\n",
    include_str!("mode_domain_wall.wgsl"),
    "\n",
    include_str!("mode_fracture.wgsl"),
    "\n",
    include_str!("mode_berry.wgsl"),
    "\n",
    include_str!("mode_hofstadter.wgsl"),
    "\n",
    include_str!("mode_stm.wgsl"),
    "\n",
    include_str!("mode_spectral.wgsl"),
    "\n",
    include_str!("mode_vortex_knot.wgsl"),
    "\n",
    include_str!("mode_bloch_wave.wgsl"),
    "\n",
    include_str!("mode_plasmon.wgsl"),
    "\n",
    include_str!("mode_nematic.wgsl"),
    "\n",
    include_str!("mode_abrikosov.wgsl"),
    "\n",
    include_str!("mode_crystal_dive.wgsl"),
    "\n",
    include_str!("mode_nano_phase.wgsl"),
    "\n",
    include_str!("mode_magnon.wgsl"),
    "\n",
    include_str!("mode_wavepacket.wgsl"),
    "\n",
    include_str!("mode_dipole_rad.wgsl"),
    "\n",
    include_str!("mode_karman.wgsl"),
    "\n",
    include_str!("mode_lorenz.wgsl"),
    "\n",
    include_str!("mode_lensing.wgsl"),
    "\n",
    include_str!("mode_grav_wave.wgsl"),
    "\n",
    include_str!("mode_zoom.wgsl"),
    "\n",
    include_str!("mode_elastic_anisotropy.wgsl"),
    "\n",
    include_str!("mode_elastic_wave.wgsl"),
    "\n",
    include_str!("mode_strain_field.wgsl"),
    "\n",
    include_str!("mode_spin_density.wgsl"),
    "\n",
    include_str!("mode_tamm_shockley.wgsl"),
    "\n",
    include_str!("dispatch.wgsl"),
);

#[cfg(test)]
mod shader_tests {
    use super::FIELD_SHADER;

    /// Parse + validate the assembled field shader with naga — the same WGSL
    /// frontend wgpu uses at runtime. Catches reserved keywords (`target`,
    /// `sample`, …), type errors, and undefined identifiers WITHOUT a GPU, so
    /// a broken mode fails `cargo test` instead of only at app launch.
    #[test]
    fn field_shader_parses_and_validates() {
        let module = naga::front::wgsl::parse_str(FIELD_SHADER)
            .unwrap_or_else(|e| panic!("FIELD_SHADER failed to parse:\n{}", e.emit_to_string(FIELD_SHADER)));

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("FIELD_SHADER failed to validate:\n{:?}", e));
    }
}
