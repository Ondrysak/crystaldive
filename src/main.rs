use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::WindowAttributes,
};

use crystal_viz::{audio, crystals, kpoints, poscar, reciprocal, renderer, symmetry};
use crystals::{all_crystals, all_groups, CrystalDef};
use poscar::Crystal;
use reciprocal::{crossfade_pack, GpuField};
use renderer::{FieldUniform, GpuState};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};

// ── field parameters ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct FieldParams {
    pub mode:        u32,
    pub kscale:      f32,
    pub speed:       f32,
    pub field_mix:   f32,
    pub iso_level:   f32,
    pub color_shift: f32,
    pub zoom:        f32,
    pub w_lattice:   f32,
    pub w_motif:     f32,
    pub w_band:      f32,
}

impl Default for FieldParams {
    fn default() -> Self {
        Self {
            mode: 4, kscale: 1.4, speed: 0.3, field_mix: 0.55,
            iso_level: 0.5, color_shift: 0.0, zoom: 1.0,
            w_lattice: 1.0, w_motif: 0.6, w_band: 0.4,
        }
    }
}

const MODE_NAMES: [&str; 36] = [
    "3D ISO", "BZ SLICE", "FERMI", "DENSITY", "NODAL", "PHASE",
    "STRIPES", "WARP", "LINKS", "XRD", "RECIP", "NONEUC",
    "PHONON", "MOIRE", "EWALD", "WANNIER", "MAGNETIC", "DISPERSION",
    "KIKUCHI", "DEFECT", "BAND SURFACE", "SPIN TEXTURE",
    "BZ PATH", "CDW", "QUASICRYSTAL", "THERMAL", "DOMAIN WALL",
    "FRACTURE",
    "BERRY", "HOFSTADTER", "STM", "SPECTRAL", "VORTEX KNOT",
    "BLOCH WAVE", "PLASMON", "NEMATIC",
];

// ── LFO ───────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct LfoParams {
    pub kscale:      bool,
    pub speed:       bool,
    pub field_mix:   bool,
    pub iso_level:   bool,
    pub color_shift: bool,
    pub zoom:        bool,
    pub w_lattice:   bool,
    pub w_motif:     bool,
    pub w_band:      bool,
    pub rate:        f32,
    pub depth:       f32,
}

impl Default for LfoParams {
    fn default() -> Self {
        Self {
            kscale: false, speed: false, field_mix: false, iso_level: false,
            color_shift: false, zoom: false, w_lattice: false, w_motif: false,
            w_band: false, rate: 0.2, depth: 0.3,
        }
    }
}

// ── Mic modulation ────────────────────────────────────────────────────────

/// Which audio band (or none) drives a parameter.
#[derive(Clone, Copy, PartialEq, Default)]
pub enum MicSrc { #[default] Off, Amp, Bass, Mid, Treble }

impl MicSrc {
    fn next(self) -> Self {
        match self {
            Self::Off    => Self::Amp,
            Self::Amp    => Self::Bass,
            Self::Bass   => Self::Mid,
            Self::Mid    => Self::Treble,
            Self::Treble => Self::Off,
        }
    }
    fn label(self) -> &'static str {
        match self { Self::Off => "·", Self::Amp => "A", Self::Bass => "B", Self::Mid => "M", Self::Treble => "T" }
    }
    fn color(self) -> egui::Color32 {
        match self {
            Self::Off    => egui::Color32::from_gray(70),
            Self::Amp    => egui::Color32::from_rgb(220, 220, 220),
            Self::Bass   => egui::Color32::from_rgb(255,  80,  80),
            Self::Mid    => egui::Color32::from_rgb( 80, 220, 120),
            Self::Treble => egui::Color32::from_rgb( 80, 160, 255),
        }
    }
    fn value(self, b: &audio::AudioBands) -> f32 {
        match self {
            Self::Off    => 0.0,
            Self::Amp    => b.amplitude,
            Self::Bass   => b.bass,
            Self::Mid    => b.mid,
            Self::Treble => b.treble,
        }
    }
}

#[derive(Clone)]
pub struct MicParams {
    pub kscale:      MicSrc,
    pub speed:       MicSrc,
    pub field_mix:   MicSrc,
    pub iso_level:   MicSrc,
    pub color_shift: MicSrc,
    pub zoom:        MicSrc,
    pub w_lattice:   MicSrc,
    pub w_motif:     MicSrc,
    pub w_band:      MicSrc,
    pub depth:       f32,
}

impl Default for MicParams {
    fn default() -> Self {
        Self {
            kscale: MicSrc::Off, speed: MicSrc::Off, field_mix: MicSrc::Off,
            iso_level: MicSrc::Off, color_shift: MicSrc::Off, zoom: MicSrc::Off,
            w_lattice: MicSrc::Off, w_motif: MicSrc::Off, w_band: MicSrc::Off,
            depth: 0.5,
        }
    }
}

// ── Combined LFO + Mic modulation ─────────────────────────────────────────

/// Returns a FieldParams with LFO and mic deltas applied additively from the base.
fn apply_modulation(
    fp:    &FieldParams,
    lfo:   &LfoParams,
    mic:   &MicParams,
    bands: &audio::AudioBands,
    t:     f32,
) -> FieldParams {
    let lfo_s = (2.0 * std::f32::consts::PI * lfo.rate * t).sin();
    macro_rules! modulate {
        ($val:expr, $lfo_en:expr, $mic_src:expr, $min:expr, $max:expr) => {{
            let range   = ($max as f32) - ($min as f32);
            let lfo_d   = if $lfo_en { lfo.depth * range * lfo_s } else { 0.0 };
            let mic_d   = $mic_src.value(bands) * mic.depth * range;
            ($val + lfo_d + mic_d).clamp($min as f32, $max as f32)
        }};
    }
    FieldParams {
        mode:        fp.mode,
        kscale:      modulate!(fp.kscale,      lfo.kscale,      mic.kscale,      0.1, 5.0),
        speed:       modulate!(fp.speed,       lfo.speed,       mic.speed,       0.0, 2.0),
        field_mix:   modulate!(fp.field_mix,   lfo.field_mix,   mic.field_mix,   0.0, 1.0),
        iso_level:   modulate!(fp.iso_level,   lfo.iso_level,   mic.iso_level,   0.0, 1.0),
        color_shift: modulate!(fp.color_shift, lfo.color_shift, mic.color_shift, 0.0, 1.0),
        zoom:        modulate!(fp.zoom,        lfo.zoom,        mic.zoom,        0.2, 5.0),
        w_lattice:   modulate!(fp.w_lattice,   lfo.w_lattice,   mic.w_lattice,   0.0, 2.0),
        w_motif:     modulate!(fp.w_motif,     lfo.w_motif,     mic.w_motif,     0.0, 2.0),
        w_band:      modulate!(fp.w_band,      lfo.w_band,      mic.w_band,      0.0, 2.0),
    }
}

// ── Tour state machine ────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum TourPhase { Settling, Walking, Fading }

struct Tour {
    pub active:      bool,
    pub crystal_idx: usize,
    phase:           TourPhase,
    phase_timer:     f32,
    pub prev_field:  Option<GpuField>,
    pub fade_t:      f32,
}

const SETTLE_DUR: f32 = 1.2;
const WALK_DUR:   f32 = 8.0;
const FADE_DUR:   f32 = 2.5;

impl Tour {
    fn new() -> Self {
        Self {
            active: false, crystal_idx: 0,
            phase: TourPhase::Settling, phase_timer: 0.0,
            prev_field: None, fade_t: 0.0,
        }
    }

    /// Returns true when it's time to snapshot and load the next crystal.
    fn tick(&mut self, dt: f32) -> bool {
        if !self.active { return false; }
        self.phase_timer += dt;
        match self.phase {
            TourPhase::Settling => {
                if self.phase_timer >= SETTLE_DUR {
                    self.phase = TourPhase::Walking;
                    self.phase_timer = 0.0;
                }
            }
            TourPhase::Walking => {
                if self.phase_timer >= WALK_DUR {
                    self.phase = TourPhase::Fading;
                    self.phase_timer = 0.0;
                    self.fade_t = 0.0;
                    return true;
                }
            }
            TourPhase::Fading => {
                self.fade_t = (self.phase_timer / FADE_DUR).clamp(0.0, 1.0);
                if self.phase_timer >= FADE_DUR {
                    self.prev_field = None;
                    self.fade_t = 0.0;
                    self.phase = TourPhase::Settling;
                    self.phase_timer = 0.0;
                }
            }
        }
        false
    }

    fn is_fading(&self) -> bool  { self.phase == TourPhase::Fading }
    fn is_walking(&self) -> bool { self.phase == TourPhase::Walking }
}

// ── App ───────────────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum RenderMode { Atoms, Field }

/// Mutations requested by the egui UI within one frame.
#[derive(Default)]
struct UiReq {
    switch_to:    Option<usize>,
    prev:         bool,
    next:         bool,
    screenshot:   bool,
    tour_toggle:  bool,
    kpath_toggle: bool,
    panel_toggle: bool,
    render_mode:  Option<RenderMode>,
}

struct App {
    gpu:          Option<GpuState>,
    #[cfg(target_arch = "wasm32")]
    event_proxy:  Option<winit::event_loop::EventLoopProxy<UserEvent>>,
    start_crystal: Crystal,
    start:        Instant,
    prev_t:       f32,

    dragging:     bool,
    last_mouse:   Option<(f64, f64)>,
    auto_rotate:  bool,

    render_mode:  RenderMode,
    field_params: FieldParams,
    mouse_norm:   [f32; 2],
    mouse_btn_down: bool,

    kpath_active: bool,
    kpt_idx:      usize,

    tour:         Tour,
    all_crystals: Vec<&'static CrystalDef>,

    panel_open:   bool,
    search_str:   String,
    lfo:          LfoParams,
    mic_params:   MicParams,
    audio:        Option<audio::AudioCapture>,
}

enum UserEvent {
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    GpuReady(GpuState),
}

impl App {
    fn new(crystal: Crystal) -> Self {
        let all = all_crystals();
        Self {
            gpu: None, start_crystal: crystal,
            #[cfg(target_arch = "wasm32")]
            event_proxy: None,
            start: Instant::now(), prev_t: 0.0,
            dragging: false, last_mouse: None, auto_rotate: true,
            render_mode: RenderMode::Field,
            field_params: FieldParams::default(),
            mouse_norm: [0.5, 0.5], mouse_btn_down: false,
            kpath_active: false, kpt_idx: 0,
            tour: Tour::new(),
            all_crystals: all,
            panel_open: true,
            search_str: String::new(),
            lfo: LfoParams::default(),
            mic_params: MicParams::default(),
            audio: audio::AudioCapture::start(),
        }
    }

    fn time(&self) -> f32 { self.start.elapsed().as_secs_f32() }

    fn load_crystal_def(&mut self, def: &CrystalDef) {
        let Some(gpu) = self.gpu.as_mut() else { return };
        let crystal = def.to_crystal();
        let new_field = GpuField::from_crystal(&crystal, 3);
        gpu.gpu_field = new_field;
        gpu.gpu_field.seed_kpoint([0.0, 0.0, 0.0], 1.0);
        gpu.update_field();
        let sys = symmetry::detect(&crystal.lattice);
        gpu.kpath = Some(kpoints::build_kpath(sys));
        self.kpt_idx = 0;
        self.kpath_active = false;
    }

    fn switch_to(&mut self, idx: usize) {
        if let Some(gpu) = &self.gpu {
            self.tour.prev_field = Some(GpuField {
                gvecs: gpu.gpu_field.gvecs.clone(),
                amps:  gpu.gpu_field.amps.clone(),
                phases: gpu.gpu_field.phases.clone(),
                b_mat: gpu.gpu_field.b_mat,
                count: gpu.gpu_field.count,
            });
        }
        self.tour.crystal_idx = idx;
        let def = self.all_crystals[idx];
        self.load_crystal_def(def);
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = WindowAttributes::default()
            .with_title("crystal-viz")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));
        #[cfg(target_arch = "wasm32")]
        let attrs = attrs.with_append(true);
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let crystal = self.start_crystal.clone();

        #[cfg(target_arch = "wasm32")]
        {
            let proxy = self.event_proxy.clone().expect("missing web event proxy");
            wasm_bindgen_futures::spawn_local(async move {
                let mut gpu = GpuState::new(window, crystal).await;
                let sys = gpu.crystal_system();
                gpu.kpath = Some(kpoints::build_kpath(sys));
                let _ = proxy.send_event(UserEvent::GpuReady(gpu));
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
        let mut gpu = pollster::block_on(GpuState::new(window, crystal));
        let sys = gpu.crystal_system();
        gpu.kpath = Some(kpoints::build_kpath(sys));
        self.gpu = Some(gpu);
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::GpuReady(gpu) => {
                gpu.window.request_redraw();
                self.gpu = Some(gpu);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gpu) = &self.gpu {
            gpu.window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // Feed to egui first
        let egui_consumed = self.gpu.as_mut()
            .map(|g| g.handle_ui_event(&event))
            .unwrap_or(false);

        let auto_rotate  = self.auto_rotate;
        let render_mode  = self.render_mode;
        let t            = self.time();
        let Some(gpu) = self.gpu.as_mut() else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{KeyCode, PhysicalKey};
                if event.state != ElementState::Pressed { return }
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                    PhysicalKey::Code(KeyCode::Tab) => {
                        self.render_mode = match self.render_mode {
                            RenderMode::Atoms => RenderMode::Field,
                            RenderMode::Field => RenderMode::Atoms,
                        };
                    }
                    PhysicalKey::Code(KeyCode::Space) => { self.auto_rotate = !self.auto_rotate; }
                    PhysicalKey::Code(KeyCode::Digit1) => gpu.set_supercell(1),
                    PhysicalKey::Code(KeyCode::Digit2) => gpu.set_supercell(2),
                    PhysicalKey::Code(KeyCode::Digit3) => gpu.set_supercell(3),
                    PhysicalKey::Code(KeyCode::KeyM) => {
                        self.field_params.mode = (self.field_params.mode + 1) % MODE_NAMES.len() as u32;
                    }
                    PhysicalKey::Code(KeyCode::KeyK) => {
                        if let Some(kp) = &gpu.kpath {
                            let n = kp.n_points();
                            self.kpt_idx = (self.kpt_idx + 1) % n;
                            let coords = kp.snap_to(self.kpt_idx);
                            gpu.gpu_field.seed_kpoint(coords, 1.0);
                            gpu.update_field();
                        }
                        self.kpath_active = false;
                    }
                    PhysicalKey::Code(KeyCode::KeyP) => {
                        self.kpath_active = !self.kpath_active;
                        if self.kpath_active {
                            if let Some(kp) = &mut gpu.kpath { kp.reset(); }
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        if render_mode == RenderMode::Field {
                            gpu.gpu_field.randomize();
                            gpu.update_field();
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyT) => {
                        self.tour.active = !self.tour.active;
                        if self.tour.active {
                            self.render_mode = RenderMode::Field;
                            self.kpath_active = true;
                            if let Some(kp) = &mut gpu.kpath { kp.reset(); }
                        }
                    }
                    PhysicalKey::Code(KeyCode::BracketRight) => {
                        self.field_params.color_shift = (self.field_params.color_shift + 0.05) % 1.0;
                    }
                    PhysicalKey::Code(KeyCode::BracketLeft) => {
                        self.field_params.color_shift = (self.field_params.color_shift - 0.05).rem_euclid(1.0);
                    }
                    _ => {}
                }
            }

            WindowEvent::Resized(size) => gpu.resize(size),

            WindowEvent::MouseInput { state, button, .. } if !egui_consumed => {
                if button == MouseButton::Left {
                    let pressed = state == ElementState::Pressed;
                    self.dragging       = pressed;
                    self.mouse_btn_down = pressed;
                    if !pressed { self.last_mouse = None; }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let sz = gpu.size;
                self.mouse_norm = [
                    (position.x as f32 / sz.width  as f32).clamp(0.0, 1.0),
                    (position.y as f32 / sz.height as f32).clamp(0.0, 1.0),
                ];
                if self.dragging && !egui_consumed && render_mode == RenderMode::Atoms {
                    if let Some((lx, ly)) = self.last_mouse {
                        gpu.orbit((position.x - lx) as f32, (position.y - ly) as f32);
                    }
                    self.last_mouse = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p)   => p.y as f32 * 0.05,
                };
                match render_mode {
                    RenderMode::Atoms => gpu.zoom(d),
                    RenderMode::Field => {
                        self.field_params.zoom = (self.field_params.zoom + d * 0.15).clamp(0.2, 5.0);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                gpu.window.request_redraw();

                let dt = (t - self.prev_t).clamp(0.0, 0.1);
                self.prev_t = t;

                // ── Tour tick ─────────────────────────────────────────
                if self.tour.active && render_mode == RenderMode::Field {
                    let should_advance = self.tour.tick(dt);
                    if should_advance {
                        let total = self.all_crystals.len();
                        let next_idx = (self.tour.crystal_idx + 1) % total;
                        // Snapshot current for crossfade
                        self.tour.prev_field = Some(GpuField {
                            gvecs: gpu.gpu_field.gvecs.clone(),
                            amps:  gpu.gpu_field.amps.clone(),
                            phases: gpu.gpu_field.phases.clone(),
                            b_mat: gpu.gpu_field.b_mat,
                            count: gpu.gpu_field.count,
                        });
                        self.tour.crystal_idx = next_idx;
                        let def = self.all_crystals[next_idx];
                        let crystal = def.to_crystal();
                        gpu.gpu_field = GpuField::from_crystal(&crystal, 3);
                        gpu.gpu_field.seed_kpoint([0.0, 0.0, 0.0], 1.0);
                        let sys = symmetry::detect(&crystal.lattice);
                        gpu.kpath = Some(kpoints::build_kpath(sys));
                        if let Some(kp) = &mut gpu.kpath { kp.reset(); }
                        self.kpt_idx = 0;
                        self.kpath_active = true;
                    }

                    if self.tour.is_walking() && self.kpath_active {
                        if let Some(kp) = &mut gpu.kpath {
                            let (k, _) = kp.tick(dt);
                            gpu.gpu_field.seed_kpoint(k, 0.4);
                        }
                    }

                    if self.tour.is_fading() {
                        if let Some(prev) = &self.tour.prev_field {
                            let packed = crossfade_pack(prev, &gpu.gpu_field, self.tour.fade_t);
                            gpu.field_pl.upload_gfield(&gpu.queue, &packed);
                        } else {
                            gpu.update_field();
                        }
                    } else {
                        gpu.update_field();
                    }
                }

                // ── Non-tour k-path walking ────────────────────────────
                if !self.tour.active && render_mode == RenderMode::Field && self.kpath_active {
                    if let Some(kp) = &mut gpu.kpath {
                        let (k, _) = kp.tick(dt);
                        gpu.gpu_field.seed_kpoint(k, 0.4);
                    }
                    gpu.update_field();
                }

                // ── Snapshot all state read by the UI closure ──────────
                // (We copy/clone so the closure doesn't hold borrows into self.)
                let cur_render_mode   = self.render_mode;
                let cur_tour_active   = self.tour.active;
                let cur_kpath_active  = self.kpath_active;
                let cur_panel_open    = self.panel_open;
                let cur_crystal_idx   = self.tour.crystal_idx;
                let cur_mode_name     = MODE_NAMES[self.field_params.mode as usize];
                let cur_crystal_name  = self.all_crystals[cur_crystal_idx].name;
                let cur_sys_name      = self.all_crystals[cur_crystal_idx].system.name();
                let cur_kpt_label: String = gpu.kpath.as_ref()
                    .map(|kp| kp.label_at(self.kpt_idx % kp.n_points()).to_owned())
                    .unwrap_or_default();
                // Snapshot field_params, lfo, mic, and current audio bands.
                let mut fp  = self.field_params.clone();
                let mut lfo = self.lfo.clone();
                let mut mic = self.mic_params.clone();
                let cur_bands = self.audio.as_ref()
                    .and_then(|a| a.bands.lock().ok().map(|b| b.clone()))
                    .unwrap_or_default();
                let mic_active = self.audio.is_some();

                // Apply LFO + mic modulation to get effective values for this frame.
                let fp_eff = apply_modulation(&fp, &lfo, &mic, &cur_bands, t);

                // Pre-build FieldUniform from the LFO-modulated snapshot so the closure can
                // freely mutate fp.* without conflicting with the match arm below.
                let cdef = self.all_crystals[cur_crystal_idx];
                let field_params_uniform = FieldUniform {
                    time:          t,
                    kscale:        fp_eff.kscale,
                    speed:         fp_eff.speed,
                    field_mix:     fp_eff.field_mix,
                    iso_level:     fp_eff.iso_level,
                    color_shift:   fp_eff.color_shift,
                    zoom:          fp_eff.zoom,
                    w_lattice:     fp_eff.w_lattice,
                    w_motif:       fp_eff.w_motif,
                    w_band:        fp_eff.w_band,
                    mode:          fp_eff.mode,
                    num_g:         gpu.gpu_field.count as u32,
                    crystal_color: [cdef.color[0], cdef.color[1], cdef.color[2], 0.0],
                    mouse:         self.mouse_norm,
                    mouse_down:    if self.mouse_btn_down { 1.0 } else { 0.0 },
                    aspect:        gpu.size.width as f32 / gpu.size.height.max(1) as f32,
                };

                // Snapshot search string
                let mut search = self.search_str.clone();

                // Crystal list (static data, no borrows)
                let groups   = all_groups();
                let all_defs = all_crystals();

                // Accumulator for mutations
                let mut req = UiReq::default();
                let req_panel_open = cur_panel_open;

                // The closure now captures &mut fp.* freely — no conflict with match below.
                let ui_fn = |ctx: &egui::Context| {
                    // Side panel
                    if req_panel_open {
                        egui::SidePanel::left("ctrl")
                            .min_width(260.0).max_width(300.0)
                            .resizable(false)
                            .show(ctx, |ui| {
                                ui.heading("Crystal Field Synthesizer");
                                ui.separator();

                                // Crystal library
                                ui.label(egui::RichText::new("CRYSTAL LIBRARY")
                                    .small().color(egui::Color32::from_rgb(120, 120, 160)));
                                ui.text_edit_singleline(&mut search);
                                let q = search.to_ascii_lowercase();

                                egui::ScrollArea::vertical()
                                    .max_height(180.0)
                                    .show(ui, |ui| {
                                        for (label, color, defs) in &groups {
                                            ui.colored_label(*color, *label);
                                            for def in defs.iter() {
                                                if !q.is_empty() && !def.name.to_ascii_lowercase().contains(&q) {
                                                    continue;
                                                }
                                                let flat_idx = all_defs.iter().position(|d| std::ptr::eq(*d, def as &CrystalDef));
                                                let is_sel = flat_idx == Some(cur_crystal_idx);
                                                let lbl = egui::RichText::new(format!("  {}", def.name))
                                                    .color(if is_sel {
                                                        egui::Color32::from_rgb(200, 200, 255)
                                                    } else {
                                                        egui::Color32::from_rgb(150, 150, 190)
                                                    });
                                                if ui.selectable_label(is_sel, lbl).clicked() {
                                                    req.switch_to = flat_idx;
                                                }
                                            }
                                        }
                                    });

                                ui.separator();

                                // Render mode buttons
                                ui.label(egui::RichText::new("RENDER MODE")
                                    .small().color(egui::Color32::from_rgb(120, 120, 160)));
                                ui.horizontal_wrapped(|ui| {
                                    for (i, name) in MODE_NAMES.iter().enumerate() {
                                        if ui.selectable_label(fp.mode == i as u32, *name).clicked() {
                                            fp.mode = i as u32;
                                        }
                                    }
                                });

                                ui.separator();

                                // Sliders — [~] LFO  [·/A/B/M/T] mic  label  [====slider====]  eff
                                ui.label(egui::RichText::new("FIELD PARAMETERS")
                                    .small().color(egui::Color32::from_rgb(120, 120, 160)));
                                macro_rules! sld {
                                    ($ui:expr, $label:literal, $val:expr, $eff:expr,
                                     $lfo_en:expr, $mic_src:expr, $min:expr, $max:expr) => {
                                        $ui.horizontal(|ui| {
                                            // [~] LFO toggle — normal sized button, coloured text
                                            let lc = if *$lfo_en {
                                                egui::Color32::from_rgb(80, 220, 120)
                                            } else {
                                                egui::Color32::from_gray(90)
                                            };
                                            if ui.add(egui::Button::new(
                                                egui::RichText::new("~").color(lc)
                                            ).min_size(egui::vec2(18.0, 18.0))).clicked() {
                                                *$lfo_en = !*$lfo_en;
                                            }

                                            // [·/A/B/M/T] mic source — cycles on click
                                            let mc = (*$mic_src).color();
                                            let ml = (*$mic_src).label();
                                            if ui.add(egui::Button::new(
                                                egui::RichText::new(ml).color(mc)
                                            ).min_size(egui::vec2(18.0, 18.0))).clicked() {
                                                let ns = (*$mic_src).next();
                                                *$mic_src = ns;
                                            }

                                            ui.label(egui::RichText::new($label).small());
                                            ui.add(egui::Slider::new($val, $min..=$max)
                                                .show_value(false));

                                            // effective value: dim if same as base, orange if modulated
                                            let eff_v: f32 = $eff;
                                            let base_v: f32 = *$val;
                                            let modulated = (eff_v - base_v).abs() > 0.001;
                                            let eff_col = if modulated {
                                                egui::Color32::from_rgb(255, 160, 60)
                                            } else {
                                                egui::Color32::from_gray(130)
                                            };
                                            ui.label(egui::RichText::new(
                                                format!("{:.2}", eff_v)
                                            ).small().color(eff_col));
                                        });
                                    }
                                }
                                sld!(ui, "kscale   ", &mut fp.kscale,      fp_eff.kscale,      &mut lfo.kscale,      &mut mic.kscale,      0.1_f32, 5.0_f32);
                                sld!(ui, "speed    ", &mut fp.speed,       fp_eff.speed,       &mut lfo.speed,       &mut mic.speed,       0.0_f32, 2.0_f32);
                                sld!(ui, "field_mix", &mut fp.field_mix,   fp_eff.field_mix,   &mut lfo.field_mix,   &mut mic.field_mix,   0.0_f32, 1.0_f32);
                                sld!(ui, "iso_level", &mut fp.iso_level,   fp_eff.iso_level,   &mut lfo.iso_level,   &mut mic.iso_level,   0.0_f32, 1.0_f32);
                                sld!(ui, "color_sft", &mut fp.color_shift, fp_eff.color_shift, &mut lfo.color_shift, &mut mic.color_shift, 0.0_f32, 1.0_f32);
                                sld!(ui, "zoom     ", &mut fp.zoom,        fp_eff.zoom,        &mut lfo.zoom,        &mut mic.zoom,        0.2_f32, 5.0_f32);
                                sld!(ui, "w_lattice", &mut fp.w_lattice,   fp_eff.w_lattice,   &mut lfo.w_lattice,   &mut mic.w_lattice,   0.0_f32, 2.0_f32);
                                sld!(ui, "w_motif  ", &mut fp.w_motif,     fp_eff.w_motif,     &mut lfo.w_motif,     &mut mic.w_motif,     0.0_f32, 2.0_f32);
                                sld!(ui, "w_band   ", &mut fp.w_band,      fp_eff.w_band,      &mut lfo.w_band,      &mut mic.w_band,      0.0_f32, 2.0_f32);

                                ui.separator();

                                // LFO section
                                ui.label(egui::RichText::new("LFO  (sin)")
                                    .small().color(egui::Color32::from_rgb(80, 220, 120)));
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("rate ").small());
                                    ui.add(egui::Slider::new(&mut lfo.rate, 0.01..=4.0)
                                        .show_value(true).suffix(" Hz"));
                                });
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("depth").small());
                                    ui.add(egui::Slider::new(&mut lfo.depth, 0.0..=1.0)
                                        .show_value(true));
                                });

                                ui.separator();

                                // Mic section
                                let mic_col = if mic_active {
                                    egui::Color32::from_rgb(255, 140, 80)
                                } else {
                                    egui::Color32::from_gray(100)
                                };
                                ui.label(egui::RichText::new(if mic_active { "MIC  (live)" } else { "MIC  (no device)" })
                                    .small().color(mic_col));
                                if mic_active {
                                    // Band meters
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new("A").small()
                                            .color(egui::Color32::from_rgb(220,220,220)));
                                        ui.add(egui::ProgressBar::new(cur_bands.amplitude).desired_width(38.0));
                                        ui.label(egui::RichText::new("B").small()
                                            .color(egui::Color32::from_rgb(255,80,80)));
                                        ui.add(egui::ProgressBar::new(cur_bands.bass).desired_width(38.0));
                                        ui.label(egui::RichText::new("M").small()
                                            .color(egui::Color32::from_rgb(80,220,120)));
                                        ui.add(egui::ProgressBar::new(cur_bands.mid).desired_width(38.0));
                                        ui.label(egui::RichText::new("T").small()
                                            .color(egui::Color32::from_rgb(80,160,255)));
                                        ui.add(egui::ProgressBar::new(cur_bands.treble).desired_width(38.0));
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new("depth").small());
                                        ui.add(egui::Slider::new(&mut mic.depth, 0.0..=2.0).show_value(true));
                                    });
                                    ui.label(egui::RichText::new("Click · on a slider to pick source")
                                        .small().color(egui::Color32::from_gray(130)));
                                }

                                ui.separator();

                                // K-path controls
                                ui.label(egui::RichText::new("K-PATH")
                                    .small().color(egui::Color32::from_rgb(120, 120, 160)));
                                ui.horizontal(|ui| {
                                    ui.monospace(format!("k = {cur_kpt_label}"));
                                    if ui.button(if cur_kpath_active { "■ STOP" } else { "▶ WALK" }).clicked() {
                                        req.kpath_toggle = true;
                                    }
                                });

                                ui.separator();

                                // Mode toggle
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(cur_render_mode == RenderMode::Atoms, "ATOMS").clicked() {
                                        req.render_mode = Some(RenderMode::Atoms);
                                    }
                                    if ui.selectable_label(cur_render_mode == RenderMode::Field, "FIELD").clicked() {
                                        req.render_mode = Some(RenderMode::Field);
                                    }
                                });

                                ui.separator();

                                // Tour controls
                                ui.horizontal(|ui| {
                                    if ui.button("◀ PREV").clicked() { req.prev = true; }
                                    let tour_lbl = if cur_tour_active { "■ STOP" } else { "▶ TOUR" };
                                    if ui.button(tour_lbl).clicked() { req.tour_toggle = true; }
                                    if ui.button("NEXT ▶").clicked() { req.next = true; }
                                });

                                if ui.button("📷 Screenshot").clicked() {
                                    req.screenshot = true;
                                }
                            });
                    }

                    // Panel toggle button
                    egui::Area::new("panel_toggle".into())
                        .fixed_pos(egui::pos2(4.0, 4.0))
                        .show(ctx, |ui| {
                            if ui.button(if req_panel_open { "◀" } else { "▶" }).clicked() {
                                req.panel_toggle = true;
                            }
                        });

                    // OSD
                    egui::Area::new("osd".into())
                        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -12.0))
                        .show(ctx, |ui| {
                            ui.visuals_mut().override_text_color =
                                Some(egui::Color32::from_rgba_unmultiplied(210, 210, 255, 210));
                            ui.label(
                                egui::RichText::new(format!(
                                    "{cur_crystal_name}  ·  {cur_sys_name}  ·  {cur_mode_name}  ·  k={cur_kpt_label}"
                                )).monospace().size(11.0)
                            );
                        });
                };

                // ── Render ────────────────────────────────────────────
                match cur_render_mode {
                    RenderMode::Atoms => {
                        if auto_rotate { gpu.orbit(0.4, 0.0); }
                        match gpu.render(t, ui_fn) {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                let sz = gpu.size; gpu.resize(sz);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("OOM"); event_loop.exit();
                            }
                            Err(e) => log::warn!("render: {e:?}"),
                        }
                    }
                    RenderMode::Field => {
                        // field_params_uniform was pre-built from `fp` snapshot before the closure.
                        match gpu.render_field(&field_params_uniform, ui_fn) {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                let sz = gpu.size; gpu.resize(sz);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("OOM"); event_loop.exit();
                            }
                            Err(e) => log::warn!("field: {e:?}"),
                        }
                    }
                }

                // ── Apply UI mutations (closure is dropped, borrows released) ──
                self.field_params = fp;
                self.lfo          = lfo;
                self.mic_params   = mic;
                self.search_str   = search;
                if req.panel_toggle { self.panel_open = !cur_panel_open; }

                if let Some(rm) = req.render_mode { self.render_mode = rm; }
                if req.kpath_toggle {
                    self.kpath_active = !cur_kpath_active;
                    if self.kpath_active {
                        if let Some(gpu2) = &mut self.gpu {
                            if let Some(kp) = &mut gpu2.kpath { kp.reset(); }
                        }
                    }
                }
                if req.tour_toggle {
                    self.tour.active = !cur_tour_active;
                    if self.tour.active {
                        self.render_mode = RenderMode::Field;
                        self.kpath_active = true;
                        if let Some(gpu2) = &mut self.gpu {
                            if let Some(kp) = &mut gpu2.kpath { kp.reset(); }
                        }
                    }
                }
                if req.prev {
                    self.tour.active = false;
                    let total = self.all_crystals.len();
                    let idx = cur_crystal_idx.checked_sub(1).unwrap_or(total - 1);
                    self.switch_to(idx);
                }
                if req.next {
                    self.tour.active = false;
                    let total = self.all_crystals.len();
                    let idx = (cur_crystal_idx + 1) % total;
                    self.switch_to(idx);
                }
                if let Some(idx) = req.switch_to {
                    self.tour.active = false;
                    self.switch_to(idx);
                }
                if req.screenshot {
                    #[cfg(target_arch = "wasm32")]
                    {
                        log::warn!("Screenshots are not wired for the browser build yet");
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(gpu2) = &mut self.gpu {
                        let png = gpu2.screenshot_field(&field_params_uniform);
                        let path = format!("crystal-viz-{}.png", chrono_stamp());
                        match std::fs::write(&path, &png) {
                            Ok(_)  => log::info!("Saved screenshot → {path}"),
                            Err(e) => log::error!("Screenshot save failed: {e}"),
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

// ── entry point ───────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn chrono_stamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    format!("{secs}")
}

fn default_crystal() -> Crystal {
    all_crystals()[0].to_crystal()
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();

    let crystal = if let Some(path) = std::env::args().nth(1) {
        poscar::Crystal::from_file(&path).unwrap_or_else(|e| {
            eprintln!("Error loading '{path}': {e}");
            std::process::exit(1);
        })
    } else if std::path::Path::new("POSCAR").exists() {
        poscar::Crystal::from_file("POSCAR").unwrap_or_else(|e| {
            eprintln!("Error loading 'POSCAR': {e}");
            std::process::exit(1);
        })
    } else {
        // No POSCAR supplied — start from the built-in crystal library
        default_crystal()
    };
    println!("Loaded {} atoms", crystal.atoms.len());
    println!("Controls:");
    println!("  Tab      — Atoms / Field toggle");
    println!("  T        — auto-tour on/off");
    println!("  M        — cycle render mode  P — k-path walk  K — next k-pt");
    println!("  [ / ]    — colour shift       scroll — zoom");
    println!("  1/2/3    — supercell  Space — auto-rotate  (atom mode)");

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new(crystal)).unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(default_crystal());
    app.event_proxy = Some(event_loop.create_proxy());
    event_loop.spawn_app(app);
}

#[cfg(target_arch = "wasm32")]
fn main() {}
