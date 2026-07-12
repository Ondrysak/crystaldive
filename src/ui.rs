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

        // Quiet graphite shell: the visualization stays dominant while controls
        // read as one coherent workstation instead of stacked translucent demos.
        let mut style = (*ctx.style()).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.override_text_color = Some(egui::Color32::from_rgb(224, 229, 235));
        style.visuals.window_fill = egui::Color32::from_rgba_unmultiplied(12, 15, 20, 248);
        style.visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(12, 15, 20, 242);
        style.visuals.extreme_bg_color = egui::Color32::from_rgb(7, 9, 13);
        style.visuals.faint_bg_color = egui::Color32::from_rgba_unmultiplied(34, 40, 48, 90);
        style.visuals.selection.bg_fill = egui::Color32::from_rgb(30, 121, 132);
        style.visuals.selection.stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(129, 231, 220));

        let inactive = &mut style.visuals.widgets.inactive;
        inactive.bg_fill = egui::Color32::from_rgb(25, 30, 37);
        inactive.weak_bg_fill = egui::Color32::from_rgb(25, 30, 37);
        inactive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(49, 58, 68));
        inactive.rounding = egui::Rounding::same(4.0);
        let hovered = &mut style.visuals.widgets.hovered;
        hovered.bg_fill = egui::Color32::from_rgb(35, 47, 55);
        hovered.weak_bg_fill = egui::Color32::from_rgb(35, 47, 55);
        hovered.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(86, 164, 164));
        hovered.rounding = egui::Rounding::same(4.0);
        let active = &mut style.visuals.widgets.active;
        active.bg_fill = egui::Color32::from_rgb(29, 105, 111);
        active.weak_bg_fill = egui::Color32::from_rgb(29, 105, 111);
        active.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(132, 232, 220));
        active.rounding = egui::Rounding::same(4.0);
        style.visuals.widgets.noninteractive.bg_fill = egui::Color32::TRANSPARENT;
        style.visuals.widgets.noninteractive.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(42, 49, 58));

        style.spacing.item_spacing = egui::vec2(7.0, 6.0);
        style.spacing.button_padding = egui::vec2(9.0, 5.0);
        style.spacing.interact_size.y = 24.0;
        style.spacing.slider_width = 118.0;
        style.text_styles.insert(
            egui::TextStyle::Heading,
            egui::FontId::new(17.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Body,
            egui::FontId::new(12.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Button,
            egui::FontId::new(11.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Small,
            egui::FontId::new(10.0, egui::FontFamily::Monospace),
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
