//! egui renderer integrated with the existing wgpu device/queue/surface.

use std::sync::Arc;
use winit::window::Window;

pub struct EguiRenderer {
    pub ctx:      egui::Context,
    renderer:     egui_wgpu::Renderer,
    state:        egui_winit::State,
}

impl EguiRenderer {
    pub fn new(
        device:       &wgpu::Device,
        surface_fmt:  wgpu::TextureFormat,
        window:       &Arc<Window>,
    ) -> Self {
        let ctx   = egui::Context::default();

        // Dark, monospace theme to match the HTML aesthetic
        let mut style = (*ctx.style()).clone();
        style.visuals = egui::Visuals::dark();
        // Slightly transparent panels
        style.visuals.window_fill       = egui::Color32::from_rgba_unmultiplied(5, 5, 12, 230);
        style.visuals.panel_fill        = egui::Color32::from_rgba_unmultiplied(5, 5, 12, 230);
        style.visuals.widgets.noninteractive.bg_fill =
            egui::Color32::from_rgba_unmultiplied(20, 15, 40, 200);
        style.visuals.widgets.inactive.bg_fill =
            egui::Color32::from_rgba_unmultiplied(30, 20, 60, 200);
        style.visuals.widgets.hovered.bg_fill =
            egui::Color32::from_rgba_unmultiplied(60, 40, 120, 220);
        style.visuals.widgets.active.bg_fill =
            egui::Color32::from_rgba_unmultiplied(80, 50, 180, 240);
        style.visuals.selection.bg_fill =
            egui::Color32::from_rgba_unmultiplied(80, 50, 180, 200);
        style.text_styles.insert(
            egui::TextStyle::Body,
            egui::FontId::new(11.0, egui::FontFamily::Monospace),
        );
        style.text_styles.insert(
            egui::TextStyle::Button,
            egui::FontId::new(10.0, egui::FontFamily::Monospace),
        );
        style.text_styles.insert(
            egui::TextStyle::Small,
            egui::FontId::new(9.0, egui::FontFamily::Monospace),
        );
        ctx.set_style(style);

        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            None,
            None,
            None,
        );

        let renderer = egui_wgpu::Renderer::new(device, surface_fmt, None, 1, false);

        Self { ctx, renderer, state }
    }

    /// Feed a winit event to egui; returns whether egui consumed it.
    pub fn handle_event(
        &mut self,
        window: &Window,
        event:  &winit::event::WindowEvent,
    ) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    /// Run the egui UI closure and paint the result onto `view`.
    pub fn render(
        &mut self,
        device:    &wgpu::Device,
        queue:     &wgpu::Queue,
        encoder:   &mut wgpu::CommandEncoder,
        view:      &wgpu::TextureView,
        window:    &Window,
        pixels_per_point: f32,
        ui_fn:     impl FnMut(&egui::Context),
    ) {
        let raw_input = self.state.take_egui_input(window);
        let full_output = self.ctx.run(raw_input, ui_fn);

        self.state.handle_platform_output(window, full_output.platform_output);

        let tris = self.ctx.tessellate(full_output.shapes, pixels_per_point);
        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, image_delta);
        }

        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [window.inner_size().width, window.inner_size().height],
            pixels_per_point,
        };
        self.renderer.update_buffers(device, queue, encoder, &tris, &screen);

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Load, // composite on top of field/atom render
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            }).forget_lifetime();
            self.renderer.render(&mut rpass, &tris, &screen);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}
