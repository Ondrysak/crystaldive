use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowExtWebSys;
use winit::window::Window;

use crate::camera::OrbitCamera;
use crate::mesh::{uv_sphere, Vertex};
use crate::modes::FIELD_SHADER;
use crate::poscar::{element_color, element_radius, Crystal};
use crate::reciprocal::{GpuField, MAX_G};
use crate::symmetry::{self, CrystalSystem};
use crate::ui::EguiRenderer;

// ── GPU-mapped types ──────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SceneUniform {
    view_proj: [[f32; 4]; 4],
    eye_pos: [f32; 3],
    time: f32,
    accent_col: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AtomInstance {
    center_radius: [f32; 4], // xyz=centre, w=radius
    color: [f32; 4],         // rgb=colour, a=ghost opacity (1=solid)
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PostParams {
    width: f32,
    height: f32,
    _pad: [f32; 2],
}

// ── Crystal-field pipeline types ─────────────────────────────────────────

/// Uniform block for the crystal-field fragment shader.
/// Layout must match the WGSL `FU` struct exactly.
///
/// `mp` is the 16-slot per-mode parameter bank (see `FieldParams::mp`). It is
/// packed as 4×vec4 (not a flat `[f32; 16]`) because uniform-address-space array
/// elements have a 16-byte stride under std140; the WGSL side reads it via the
/// `mp(i)` accessor. Slots 0..8 carry the canonical crystal-field generator
/// params (kscale, speed, field_mix, iso_level, color_shift, zoom,
/// w_lattice, w_motif, w_band); slots 9..15 are free per mode.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FieldUniform {
    pub mp: [[f32; 4]; 4],       //   0..64 — per-mode param bank (vec4-packed)
    pub time: f32,               //  64
    pub mode: u32,               //  68
    pub num_g: u32,              //  72
    pub aspect: f32,             //  76
    pub crystal_color: [f32; 4], //  80
    pub mouse: [f32; 2],         //  96
    pub mouse_down: f32,         // 104
    // feedback fields
    pub fb_enabled: u32,     // 108
    pub fb_mirror: u32,      // 112
    pub fb_zoom: f32,        // 116
    pub fb_offset_x: f32,    // 120
    pub fb_offset_y: f32,    // 124
    pub fb_rotation: f32,    // 128
    pub fb_decay: f32,       // 132
    pub fb_color_shift: f32, // 136
    pub fb_inject: f32,      // 140
    pub fb_fold_angle: f32,  // 144
    pub fb_saturation: f32,  // 148
    pub fb_brightness: f32,  // 152
    pub fb_blend_mode: u32,  // 156
    pub fb_motion_blur: f32, // 160 — un-warped temporal smoothing (Fraksl MotionBlur)
    pub _pad: [f32; 3],      // 164..176 — align to 176 (WGSL 16-byte struct align)
} // total 176 bytes

/// Pack the flat 16-slot param bank into the vec4-aligned uniform layout.
#[inline]
pub fn pack_mp(mp: &[f32; 16]) -> [[f32; 4]; 4] {
    [
        [mp[0], mp[1], mp[2], mp[3]],
        [mp[4], mp[5], mp[6], mp[7]],
        [mp[8], mp[9], mp[10], mp[11]],
        [mp[12], mp[13], mp[14], mp[15]],
    ]
}

pub struct FieldPipeline {
    pub uniform_buf: wgpu::Buffer,
    pub g_buf: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,
    transition_uniform_buf: wgpu::Buffer,
    transition_g_buf: wgpu::Buffer,
    transition_bg: wgpu::BindGroup,
    field_pl: wgpu::RenderPipeline,
    field_fade_pl: wgpu::RenderPipeline,
}

impl FieldPipeline {
    pub fn new(device: &wgpu::Device, surface_fmt: wgpu::TextureFormat) -> Self {
        let make_uniform_buf = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<FieldUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let make_g_buf = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (MAX_G * 4 * 2 * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let uniform_buf = make_uniform_buf("field uniform");
        let g_buf = make_g_buf("g_block");
        let transition_uniform_buf = make_uniform_buf("transition field uniform");
        let transition_g_buf = make_g_buf("transition g_block");
        let bgl = field_bgl(device);
        let make_bg = |label, uniform_buf: &wgpu::Buffer, g_buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: g_buf.as_entire_binding(),
                    },
                ],
            })
        };
        let uniform_bg = make_bg("field bg", &uniform_buf, &g_buf);
        let transition_bg = make_bg(
            "transition field bg",
            &transition_uniform_buf,
            &transition_g_buf,
        );

        let field_pl = build_field_pl(device, &bgl, surface_fmt);
        let field_fade_pl = build_field_fade_pl(device, &bgl, surface_fmt);
        Self {
            uniform_buf,
            g_buf,
            uniform_bg,
            transition_uniform_buf,
            transition_g_buf,
            transition_bg,
            field_pl,
            field_fade_pl,
        }
    }

    /// Upload packed G-vector data (from `GpuField::pack()`).
    /// The pack() layout matches GBlock exactly: 128 × vec4 gamp then 128 × vec4 phases.
    pub fn upload_gfield(&self, queue: &wgpu::Queue, data: &[f32]) {
        queue.write_buffer(&self.g_buf, 0, bytemuck::cast_slice(data));
    }

    fn upload_transition(&self, queue: &wgpu::Queue, params: &FieldUniform, field: &GpuField) {
        queue.write_buffer(&self.transition_uniform_buf, 0, bytemuck::bytes_of(params));
        queue.write_buffer(
            &self.transition_g_buf,
            0,
            bytemuck::cast_slice(&field.pack()),
        );
    }
}

// ── Per-shape draw group ──────────────────────────────────────────────────

struct ShapeGroup {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    icnt: u32,
    ibuf2: wgpu::Buffer, // instance buffer
    inst_cnt: u32,
}

impl ShapeGroup {
    fn new(device: &wgpu::Device, verts: &[Vertex], idx: &[u32], insts: &[AtomInstance]) -> Self {
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(idx),
            usage: wgpu::BufferUsages::INDEX,
        });
        let ibuf2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(insts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            vbuf,
            ibuf,
            icnt: idx.len() as u32,
            ibuf2,
            inst_cnt: insts.len() as u32,
        }
    }
    fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.inst_cnt == 0 {
            return;
        }
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.set_vertex_buffer(1, self.ibuf2.slice(..));
        pass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.icnt, 0, 0..self.inst_cnt);
    }
}
struct BondMesh {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    icnt: u32,
}

impl BondMesh {
    fn new(device: &wgpu::Device, vertices: &[Vertex], indices: &[u32]) -> Self {
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bond vertices"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bond indices"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vbuf,
            ibuf,
            icnt: indices.len() as u32,
        }
    }

    fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.icnt, 0, 0..1);
    }
}

// ── Post-processing (bloom) ───────────────────────────────────────────────

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

struct PostProcess {
    scene_view: wgpu::TextureView,
    bright_view: wgpu::TextureView,
    blur_a_view: wgpu::TextureView,

    sampler: wgpu::Sampler,
    params_buf: wgpu::Buffer, // PostParams: width + height

    bright_bg: wgpu::BindGroup,
    blur_h_bg: wgpu::BindGroup,
    blur_v_bg: wgpu::BindGroup,
    composite_bg: wgpu::BindGroup,

    bright_pl: wgpu::RenderPipeline,
    blur_h_pl: wgpu::RenderPipeline,
    blur_v_pl: wgpu::RenderPipeline,
    composite_pl: wgpu::RenderPipeline,
}

impl PostProcess {
    fn new(device: &wgpu::Device, w: u32, h: u32, surface_fmt: wgpu::TextureFormat) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("post params"),
            contents: bytemuck::bytes_of(&PostParams {
                width: w as f32,
                height: h as f32,
                _pad: [0.0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (scene_view, bright_view, blur_a_view) = Self::make_textures(device, w, h);

        // ── bind group layouts ─────────────────────────────────────────
        let bgl_1 = bgl_single_tex(device); // bright-pass: 1 HDR tex + sampler
        let bgl_blur = bgl_blur(device); // blur: 1 tex + sampler + params
        let bgl_comp = bgl_composite(device); // composite: 2 HDR tex + sampler

        // ── bind groups ────────────────────────────────────────────────
        let bright_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bright bg"),
            layout: &bgl_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let blur_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_h bg"),
            layout: &bgl_blur,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bright_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let blur_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_v bg"),
            layout: &bgl_blur,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite bg"),
            layout: &bgl_comp,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&blur_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // ── pipelines ──────────────────────────────────────────────────
        let bright_pl = fullscreen_pl(device, &bgl_1, BRIGHT_SHADER, "fs_bright", HDR_FORMAT);
        let blur_h_pl = fullscreen_pl(device, &bgl_blur, BLUR_H_SHADER, "fs_blur_h", HDR_FORMAT);
        let blur_v_pl = fullscreen_pl(device, &bgl_blur, BLUR_V_SHADER, "fs_blur_v", HDR_FORMAT);
        let composite_pl = fullscreen_pl(
            device,
            &bgl_comp,
            COMPOSITE_SHADER,
            "fs_composite",
            surface_fmt,
        );

        Self {
            scene_view,
            bright_view,
            blur_a_view,
            sampler,
            params_buf,
            bright_bg,
            blur_h_bg,
            blur_v_bg,
            composite_bg,
            bright_pl,
            blur_h_pl,
            blur_v_pl,
            composite_pl,
        }
    }

    fn make_textures(
        device: &wgpu::Device,
        w: u32,
        h: u32,
    ) -> (wgpu::TextureView, wgpu::TextureView, wgpu::TextureView) {
        let make = |label: &str| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: w.max(1),
                        height: h.max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: HDR_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        (make("scene"), make("bright"), make("blur_a"))
    }

    fn resize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        surface_fmt: wgpu::TextureFormat,
    ) {
        // Recreate textures and all bind groups that reference them
        let (scene_view, bright_view, blur_a_view) = Self::make_textures(device, w, h);

        queue.write_buffer(
            &self.params_buf,
            0,
            bytemuck::bytes_of(&PostParams {
                width: w as f32,
                height: h as f32,
                _pad: [0.0; 2],
            }),
        );

        let bgl_1 = bgl_single_tex(device);
        let bgl_blur = bgl_blur(device);
        let bgl_comp = bgl_composite(device);

        self.bright_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bright bg"),
            layout: &bgl_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.blur_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_h bg"),
            layout: &bgl_blur,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bright_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buf.as_entire_binding(),
                },
            ],
        });
        self.blur_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_v bg"),
            layout: &bgl_blur,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buf.as_entire_binding(),
                },
            ],
        });
        self.composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite bg"),
            layout: &bgl_comp,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&blur_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        // Rebuild composite pipeline if needed (same layout, so no — but keep for future)
        let _ = surface_fmt; // may differ post-resize; pass in if fmt can change

        self.scene_view = scene_view;
        self.bright_view = bright_view;
        self.blur_a_view = blur_a_view;
    }
}

// ── Feedback / self-similarity pass ──────────────────────────────────────

const FB_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

struct FeedbackPass {
    field_rt: wgpu::TextureView,
    accum_a: wgpu::TextureView,
    accum_b: wgpu::TextureView,
    parity: bool,
    sampler: wgpu::Sampler,
    // field_hdr_pl renders the field shader targeting Rgba16Float (not surface_fmt).
    // field_hdr_fade_pl blends the incoming field over a previous mode image.
    field_hdr_pl: wgpu::RenderPipeline,
    field_hdr_fade_pl: wgpu::RenderPipeline,
    feedback_pl: wgpu::RenderPipeline,
    blit_pl: wgpu::RenderPipeline,
    // feedback_pl bind groups: reads field_rt + one accum → writes other accum
    bg_f_then_a: wgpu::BindGroup, // field + accum_a → writes accum_b
    bg_f_then_b: wgpu::BindGroup, // field + accum_b → writes accum_a
    bg_blit_a: wgpu::BindGroup,   // blit accum_a → swapchain
    bg_blit_b: wgpu::BindGroup,   // blit accum_b → swapchain
}

impl FeedbackPass {
    fn make_tex(device: &wgpu::Device, w: u32, h: u32, label: &str) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FB_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn new(
        device: &wgpu::Device,
        w: u32,
        h: u32,
        surface_fmt: wgpu::TextureFormat,
        uniform_buf: &wgpu::Buffer,
        field_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fb sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let field_rt = Self::make_tex(device, w, h, "fb_field_rt");
        let accum_a = Self::make_tex(device, w, h, "fb_accum_a");
        let accum_b = Self::make_tex(device, w, h, "fb_accum_b");

        // BGL for feedback shader: uniform + prev_accum + field_rt + sampler
        let fb_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fb bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // BGL for blit shader: accum tex + sampler
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let make_fb_bg = |prev: &wgpu::TextureView, field: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fb bg"),
                layout: &fb_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(prev),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(field),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            })
        };
        let make_blit_bg = |accum: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit bg"),
                layout: &blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(accum),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            })
        };

        let bg_f_then_a = make_fb_bg(&accum_a, &field_rt);
        let bg_f_then_b = make_fb_bg(&accum_b, &field_rt);
        let bg_blit_a = make_blit_bg(&accum_a);
        let bg_blit_b = make_blit_bg(&accum_b);

        let field_hdr_pl = build_field_pl(device, field_bgl, FB_FORMAT);
        let field_hdr_fade_pl = build_field_fade_pl(device, field_bgl, FB_FORMAT);
        let feedback_pl = Self::build_feedback_pl(device, &fb_bgl, FB_FORMAT);
        let blit_pl = Self::build_blit_pl(device, &blit_bgl, surface_fmt);

        Self {
            field_rt,
            accum_a,
            accum_b,
            parity: false,
            sampler,
            field_hdr_pl,
            field_hdr_fade_pl,
            feedback_pl,
            blit_pl,
            bg_f_then_a,
            bg_f_then_b,
            bg_blit_a,
            bg_blit_b,
        }
    }

    fn resize(
        &mut self,
        device: &wgpu::Device,
        w: u32,
        h: u32,
        surface_fmt: wgpu::TextureFormat,
        uniform_buf: &wgpu::Buffer,
    ) {
        self.field_rt = Self::make_tex(device, w, h, "fb_field_rt");
        self.accum_a = Self::make_tex(device, w, h, "fb_accum_a");
        self.accum_b = Self::make_tex(device, w, h, "fb_accum_b");

        let fb_bgl = self.feedback_pl.get_bind_group_layout(0);
        let blit_bgl = self.blit_pl.get_bind_group_layout(0);

        let make_fb_bg = |prev: &wgpu::TextureView, field: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fb bg"),
                layout: &fb_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(prev),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(field),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };
        let make_blit_bg = |accum: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit bg"),
                layout: &blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(accum),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };

        self.bg_f_then_a = make_fb_bg(&self.accum_a, &self.field_rt);
        self.bg_f_then_b = make_fb_bg(&self.accum_b, &self.field_rt);
        self.bg_blit_a = make_blit_bg(&self.accum_a);
        self.bg_blit_b = make_blit_bg(&self.accum_b);

        // Rebuild blit pipeline if surface format changed
        self.blit_pl = Self::build_blit_pl(device, &blit_bgl, surface_fmt);
    }

    fn build_feedback_pl(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        fmt: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("feedback"),
            source: wgpu::ShaderSource::Wgsl(FEEDBACK_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[bgl],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("feedback pl"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &sm,
                entry_point: "vs_screen",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sm,
                entry_point: "fs_feedback",
                targets: &[Some(wgpu::ColorTargetState {
                    format: fmt,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn build_blit_pl(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        fmt: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[bgl],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit pl"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &sm,
                entry_point: "vs_screen",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sm,
                entry_point: "fs_blit",
                targets: &[Some(wgpu::ColorTargetState {
                    format: fmt,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }
}

// ── Main GPU state ────────────────────────────────────────────────────────

pub struct GpuState {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    pub queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,

    camera: OrbitCamera,
    scene_buf: wgpu::Buffer,
    scene_bg: wgpu::BindGroup,

    atom_pl: wgpu::RenderPipeline,
    cell_pl: wgpu::RenderPipeline,
    depth_view: wgpu::TextureView,
    bond_pl: wgpu::RenderPipeline,

    shapes: Vec<ShapeGroup>,
    bonds: Option<BondMesh>,
    cell_vbuf: wgpu::Buffer,
    cell_vcnt: u32,

    post: PostProcess,
    surface_fmt: wgpu::TextureFormat,

    pub supercell: usize,
    crystal: Crystal,
    crystal_sys: CrystalSystem,

    // ── crystal-field mode ────────────────────────────────────────────
    pub field_pl: FieldPipeline,
    pub gpu_field: GpuField,
    pub kpath: Option<crate::kpoints::KpathWalker>,

    // ── video-feedback self-similarity ────────────────────────────────
    feedback: FeedbackPass,
    pub fb_clear: bool, // set to true to wipe accumulators next frame

    #[cfg(not(target_arch = "wasm32"))]
    capture_path: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    surface_capturable: bool,

    // ── egui overlay ──────────────────────────────────────────────────
    pub egui: EguiRenderer,
}

impl GpuState {
    pub async fn new(window: Arc<Window>, crystal: Crystal) -> Self {
        let size = canvas_size(&window);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no GPU adapter");

        log::info!("GPU: {}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("crystal-viz"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .expect("device creation failed");

        let caps = surface.get_capabilities(&adapter);
        let surface_fmt = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        #[cfg(not(target_arch = "wasm32"))]
        let surface_capturable = caps.usages.contains(wgpu::TextureUsages::COPY_SRC);
        #[cfg(not(target_arch = "wasm32"))]
        let surface_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | if surface_capturable {
                wgpu::TextureUsages::COPY_SRC
            } else {
                wgpu::TextureUsages::empty()
            };
        #[cfg(target_arch = "wasm32")]
        let surface_usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let config = wgpu::SurfaceConfiguration {
            usage: surface_usage,
            format: surface_fmt,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let crystal_sys = symmetry::detect(&crystal.lattice);
        log::info!("Crystal system: {}", crystal_sys.name());

        // ── scene uniform ─────────────────────────────────────────────
        let cc = crystal.cell_center();
        let target = glam::Vec3::from(cc);
        let max_a = crystal
            .lattice
            .iter()
            .map(|r| (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt())
            .fold(0f32, f32::max);
        let camera = OrbitCamera::new(target, max_a * 2.8);

        let scene_bgl = scene_bgl(&device);
        let scene_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene uniform"),
            contents: bytemuck::bytes_of(&make_scene_uniform(
                &camera,
                aspect(&config),
                0.0,
                crystal_sys,
            )),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene bg"),
            layout: &scene_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buf.as_entire_binding(),
            }],
        });

        // ── pipelines ─────────────────────────────────────────────────
        let atom_pl = build_atom_pl(&device, &scene_bgl, HDR_FORMAT);
        let cell_pl = build_cell_pl(&device, &scene_bgl, HDR_FORMAT);
        let bond_pl = build_bond_pl(&device, &scene_bgl, HDR_FORMAT);

        // ── depth ─────────────────────────────────────────────────────
        let depth_view = make_depth(&device, &config);

        // ── scene geometry ────────────────────────────────────────────
        let (shapes, bonds) = build_shape_groups(&device, &crystal, 1);
        let cell_verts = cell_lines(&crystal.lattice);
        let cell_vcnt = cell_verts.len() as u32 / 3;
        let cell_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cell vbuf"),
            contents: bytemuck::cast_slice(&cell_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // ── post-process ──────────────────────────────────────────────
        let post = PostProcess::new(&device, size.width.max(1), size.height.max(1), surface_fmt);

        // ── crystal-field pipeline ────────────────────────────────────
        let field_pl = FieldPipeline::new(&device, surface_fmt);
        let mut gpu_field = GpuField::from_crystal(&crystal, 3);
        gpu_field.seed_kpoint([0.0, 0.0, 0.0], 1.0); // start at Γ
        field_pl.upload_gfield(&queue, &gpu_field.pack());

        // ── feedback pass ─────────────────────────────────────────────
        let fb_field_bgl = field_bgl(&device);
        let feedback = FeedbackPass::new(
            &device,
            size.width.max(1),
            size.height.max(1),
            surface_fmt,
            &field_pl.uniform_buf,
            &fb_field_bgl,
        );

        // ── egui ──────────────────────────────────────────────────────
        let egui = EguiRenderer::new(&device, surface_fmt, &window);

        // A set delay hands the trigger to the app loop (mid-run capture).
        #[cfg(not(target_arch = "wasm32"))]
        let capture_path = if std::env::var("CRYSTALVIZ_SCREENSHOT_DELAY").is_ok() {
            None
        } else {
            std::env::var("CRYSTALVIZ_SCREENSHOT").ok()
        };
        #[cfg(not(target_arch = "wasm32"))]
        let capture_path = if surface_capturable {
            capture_path
        } else {
            if capture_path.is_some() {
                log::warn!("Surface screenshots are unsupported by this GPU backend");
            }
            None
        };

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            camera,
            scene_buf,
            scene_bg,
            atom_pl,
            cell_pl,
            bond_pl,
            depth_view,
            shapes,
            bonds,
            cell_vbuf,
            cell_vcnt,
            post,
            surface_fmt,
            supercell: 1,
            crystal,
            crystal_sys,
            field_pl,
            gpu_field,
            kpath: None, // set by caller after construction
            feedback,
            fb_clear: false,
            #[cfg(not(target_arch = "wasm32"))]
            capture_path,
            #[cfg(not(target_arch = "wasm32"))]
            surface_capturable,

            egui,
        }
    }

    // ── public controls ───────────────────────────────────────────────

    /// Feed a winit window event to egui; returns true if egui consumed it.
    pub fn handle_ui_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        self.egui.handle_event(&self.window, event)
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.camera.orbit(dx, dy);
    }
    pub fn zoom(&mut self, d: f32) {
        self.camera.zoom(d);
    }

    fn frame_crystal(&mut self) {
        let scale = self.supercell as f32;
        let target = glam::Vec3::from(self.crystal.cell_center()) * scale;
        let max_axis = self
            .crystal
            .lattice
            .iter()
            .map(|axis| (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt())
            .fold(0.0_f32, f32::max);
        self.camera = OrbitCamera::new(target, max_axis * scale * 2.8);
    }

    pub fn set_crystal(&mut self, crystal: Crystal) {
        self.crystal_sys = symmetry::detect(&crystal.lattice);
        self.crystal = crystal;
        (self.shapes, self.bonds) = build_shape_groups(&self.device, &self.crystal, self.supercell);
        let cell_vertices = supercell_lines(&self.crystal.lattice, self.supercell);
        self.cell_vcnt = cell_vertices.len() as u32 / 3;
        self.cell_vbuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cell vbuf"),
                contents: bytemuck::cast_slice(&cell_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        self.frame_crystal();
    }

    pub fn set_supercell(&mut self, n: usize) {
        if n == self.supercell {
            return;
        }
        self.supercell = n;
        (self.shapes, self.bonds) = build_shape_groups(&self.device, &self.crystal, n);
        let cell_vertices = supercell_lines(&self.crystal.lattice, n);
        self.cell_vcnt = cell_vertices.len() as u32 / 3;
        self.cell_vbuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cell vbuf"),
                contents: bytemuck::cast_slice(&cell_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        self.frame_crystal();
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = make_depth(&self.device, &self.config);
        self.post.resize(
            &self.device,
            &self.queue,
            new_size.width,
            new_size.height,
            self.surface_fmt,
        );
        self.feedback.resize(
            &self.device,
            new_size.width,
            new_size.height,
            self.surface_fmt,
            &self.field_pl.uniform_buf,
        );
    }

    fn sync_window_size(&mut self) {
        let size = canvas_size(&self.window);
        if size != self.size {
            self.resize(size);
        }
    }

    /// Capture the next fully composed frame, including the egui overlay.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn request_screenshot(&mut self, path: String) {
        if self.surface_capturable {
            self.capture_path = Some(path);
        } else {
            log::warn!("Surface screenshots are unsupported by this GPU backend");
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn encode_surface_capture(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
    ) -> Option<(String, wgpu::Buffer, u32)> {
        let path = self.capture_path.take()?;
        let w = self.size.width.max(1);
        let h = self.size.height.max(1);
        let aligned_bytes_per_row = ((w * 4) + 255) & !255;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composed screenshot readback"),
            size: (aligned_bytes_per_row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        Some((path, readback, aligned_bytes_per_row))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn save_surface_capture(&self, capture: Option<(String, wgpu::Buffer, u32)>) {
        let Some((path, readback, aligned_bytes_per_row)) = capture else {
            return;
        };
        let w = self.size.width.max(1);
        let h = self.size.height.max(1);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let raw = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        let bgra = matches!(
            self.surface_fmt,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        for row in 0..h as usize {
            let start = row * aligned_bytes_per_row as usize;
            let row_bytes = &raw[start..start + (w * 4) as usize];
            if bgra {
                for pixel in row_bytes.chunks_exact(4) {
                    pixels.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
                }
            } else {
                pixels.extend_from_slice(row_bytes);
            }
        }
        drop(raw);
        readback.unmap();
        let Some(image) = image::RgbaImage::from_raw(w, h, pixels) else {
            log::error!("Screenshot buffer size mismatch");
            return;
        };
        match image.save(&path) {
            Ok(()) => log::info!("Saved screenshot → {path}"),
            Err(error) => log::error!("Screenshot save failed: {error}"),
        }
    }

    pub fn render(
        &mut self,
        time: f32,
        ui_fn: impl FnMut(&egui::Context),
    ) -> Result<(), wgpu::SurfaceError> {
        self.sync_window_size();
        // Update scene uniform
        let u = make_scene_uniform(&self.camera, aspect(&self.config), time, self.crystal_sys);
        self.queue
            .write_buffer(&self.scene_buf, 0, bytemuck::bytes_of(&u));

        let output = self.surface.get_current_texture()?;
        let screen_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // ── Pass 1: scene → HDR texture ───────────────────────────────
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.post.scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            // Cell wireframe
            pass.set_pipeline(&self.cell_pl);
            pass.set_bind_group(0, &self.scene_bg, &[]);
            pass.set_vertex_buffer(0, self.cell_vbuf.slice(..));
            pass.draw(0..self.cell_vcnt, 0..1);
            // Nearest-neighbour bonds.
            if let Some(bonds) = &self.bonds {
                pass.set_pipeline(&self.bond_pl);
                pass.set_bind_group(0, &self.scene_bg, &[]);
                bonds.draw(&mut pass);
            }
            // Atoms are drawn last so sphere surfaces cleanly cap the bonds.
            pass.set_pipeline(&self.atom_pl);
            pass.set_bind_group(0, &self.scene_bg, &[]);
            for sg in &self.shapes {
                sg.draw(&mut pass);
            }
        }

        // ── Pass 2: bright filter ─────────────────────────────────────
        fullscreen_pass(
            &mut enc,
            &self.post.bright_view,
            &self.post.bright_pl,
            &self.post.bright_bg,
            false,
        );

        // ── Pass 3: blur H ────────────────────────────────────────────
        fullscreen_pass(
            &mut enc,
            &self.post.blur_a_view,
            &self.post.blur_h_pl,
            &self.post.blur_h_bg,
            false,
        );

        // ── Pass 4: blur V (blur_a → bright reused as output) ─────────
        // Re-use bright_view as the final blur result (we no longer need raw bright)
        fullscreen_pass(
            &mut enc,
            &self.post.bright_view,
            &self.post.blur_v_pl,
            &self.post.blur_v_bg,
            false,
        );

        // ── Pass 5: composite to screen ───────────────────────────────
        // composite_bg: scene + bright(=final blur) + sampler
        fullscreen_pass(
            &mut enc,
            &screen_view,
            &self.post.composite_pl,
            &self.post.composite_bg,
            true,
        );

        // ── Pass 6: egui overlay ──────────────────────────────────────
        let ppp = self.window.scale_factor() as f32;
        let window = Arc::clone(&self.window);
        self.egui.render(
            &self.device,
            &self.queue,
            &mut enc,
            &screen_view,
            &window,
            ppp,
            ui_fn,
        );

        #[cfg(not(target_arch = "wasm32"))]
        let capture = self.encode_surface_capture(&mut enc, &output.texture);
        self.queue.submit(std::iter::once(enc.finish()));
        #[cfg(not(target_arch = "wasm32"))]
        self.save_surface_capture(capture);
        output.present();
        Ok(())
    }

    pub fn crystal_system(&self) -> CrystalSystem {
        self.crystal_sys
    }

    /// Re-upload current phase state to the G-texture.
    pub fn update_field(&self) {
        self.field_pl
            .upload_gfield(&self.queue, &self.gpu_field.pack());
    }

    /// Upload a caller-produced G-block, used by continuous crystal morphs.
    pub fn upload_field_pack(&self, packed: &[f32]) {
        self.field_pl.upload_gfield(&self.queue, packed);
    }

    /// Render the crystal field (bypasses atom/bloom pipeline — writes straight to swapchain).
    /// Render one field image. Call [`render_field_transition`] when a tour
    /// changes mode or crystal and needs to crossfade two independently shaded
    /// field images.
    pub fn render_field(
        &mut self,
        params: &FieldUniform,
        ui_fn: impl FnMut(&egui::Context),
    ) -> Result<(), wgpu::SurfaceError> {
        self.render_field_inner(None, params, ui_fn)
    }

    /// Crossfade a previous field/mode into the current `gpu_field` and
    /// parameters. Each image has an independent uniform + G-vector bind
    /// group, so mode-specific shader logic actually blends instead of merely
    /// morphing one mode's inputs.
    pub fn render_field_transition(
        &mut self,
        previous_params: &FieldUniform,
        previous_field: Option<&GpuField>,
        params: &FieldUniform,
        mix: f32,
        ui_fn: impl FnMut(&egui::Context),
    ) -> Result<(), wgpu::SurfaceError> {
        self.render_field_inner(
            Some((previous_params, previous_field, mix.clamp(0.0, 1.0))),
            params,
            ui_fn,
        )
    }

    fn render_field_inner(
        &mut self,
        transition: Option<(&FieldUniform, Option<&GpuField>, f32)>,
        params: &FieldUniform,
        ui_fn: impl FnMut(&egui::Context),
    ) -> Result<(), wgpu::SurfaceError> {
        self.sync_window_size();
        self.queue
            .write_buffer(&self.field_pl.uniform_buf, 0, bytemuck::bytes_of(params));
        if let Some((previous_params, previous_field, _)) = transition {
            self.field_pl.upload_transition(
                &self.queue,
                previous_params,
                previous_field.unwrap_or(&self.gpu_field),
            );
            // The normal G buffer may have been preloaded by the tour tick
            // with a legacy G-vector blend. The incoming pass needs the actual
            // destination field, not that packed approximation.
            self.field_pl
                .upload_gfield(&self.queue, &self.gpu_field.pack());
        }

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());

        if params.fb_enabled == 0 {
            if let Some((_, _, mix)) = transition {
                field_pass(
                    &mut enc,
                    &view,
                    &self.field_pl.field_pl,
                    &self.field_pl.transition_bg,
                    true,
                    None,
                );
                field_pass(
                    &mut enc,
                    &view,
                    &self.field_pl.field_fade_pl,
                    &self.field_pl.uniform_bg,
                    false,
                    Some(mix),
                );
            } else {
                field_pass(
                    &mut enc,
                    &view,
                    &self.field_pl.field_pl,
                    &self.field_pl.uniform_bg,
                    true,
                    None,
                );
            }
        } else {
            // Render both mode images into the feedback input before the
            // accumulation pass. Feedback then sees a continuous image rather
            // than retaining a hard cut from the old shader dispatch.
            if let Some((_, _, mix)) = transition {
                field_pass(
                    &mut enc,
                    &self.feedback.field_rt,
                    &self.feedback.field_hdr_pl,
                    &self.field_pl.transition_bg,
                    true,
                    None,
                );
                field_pass(
                    &mut enc,
                    &self.feedback.field_rt,
                    &self.feedback.field_hdr_fade_pl,
                    &self.field_pl.uniform_bg,
                    false,
                    Some(mix),
                );
            } else {
                field_pass(
                    &mut enc,
                    &self.feedback.field_rt,
                    &self.feedback.field_hdr_pl,
                    &self.field_pl.uniform_bg,
                    true,
                    None,
                );
            }

            let (dst_accum, fb_bg) = if self.feedback.parity {
                (&self.feedback.accum_b, &self.feedback.bg_f_then_a)
            } else {
                (&self.feedback.accum_a, &self.feedback.bg_f_then_b)
            };
            {
                let load = if self.fb_clear {
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                } else {
                    wgpu::LoadOp::Load
                };
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("feedback composite"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: dst_accum,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.feedback.feedback_pl);
                pass.set_bind_group(0, fb_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            let blit_bg = if self.feedback.parity {
                &self.feedback.bg_blit_b
            } else {
                &self.feedback.bg_blit_a
            };
            fullscreen_pass(&mut enc, &view, &self.feedback.blit_pl, blit_bg, true);
            self.feedback.parity = !self.feedback.parity;
            self.fb_clear = false;
        }

        let ppp = self.window.scale_factor() as f32;
        let window = Arc::clone(&self.window);
        self.egui.render(
            &self.device,
            &self.queue,
            &mut enc,
            &view,
            &window,
            ppp,
            ui_fn,
        );

        #[cfg(not(target_arch = "wasm32"))]
        let capture = self.encode_surface_capture(&mut enc, &output.texture);
        self.queue.submit([enc.finish()]);
        #[cfg(not(target_arch = "wasm32"))]
        self.save_surface_capture(capture);
        output.present();
        Ok(())
    }
}

// ── Scene geometry helpers ────────────────────────────────────────────────

fn build_shape_groups(
    device: &wgpu::Device,
    crystal: &Crystal,
    supercell: usize,
) -> (Vec<ShapeGroup>, Option<BondMesh>) {
    let n = supercell as i32;
    let lat = &crystal.lattice;
    let mut instances = Vec::<AtomInstance>::new();

    let mut add_atom = |atom: &crate::poscar::Atom, offset: [f32; 3], alpha: f32| {
        let color = element_color(&atom.species);
        instances.push(AtomInstance {
            center_radius: [
                atom.pos_cart[0] + offset[0],
                atom.pos_cart[1] + offset[1],
                atom.pos_cart[2] + offset[2],
                element_radius(&atom.species) * 0.82,
            ],
            color: [color[0], color[1], color[2], alpha],
        });
    };

    // Canonical atoms for every requested cell.
    for na in 0..n {
        for nb in 0..n {
            for nc in 0..n {
                let offset = lattice_offset(lat, na, nb, nc);
                for atom in &crystal.atoms {
                    add_atom(atom, offset, 1.0);
                }
            }
        }
    }

    // POSCAR convention stores boundary atoms only once. Mirror atoms that sit
    // on a zero face onto the opposite outer face so the displayed unit cell
    // reads as a complete periodic structure (e.g. all eight BCC corners).
    let basis = glam::Mat3::from_cols(
        glam::Vec3::from_array(lat[0]),
        glam::Vec3::from_array(lat[1]),
        glam::Vec3::from_array(lat[2]),
    );
    let inv_basis = basis.inverse();
    for atom in &crystal.atoms {
        let frac = inv_basis * glam::Vec3::from_array(atom.pos_cart);
        let sx: &[i32] = if frac.x.abs() < 1.0e-4 { &[0, n] } else { &[0] };
        let sy: &[i32] = if frac.y.abs() < 1.0e-4 { &[0, n] } else { &[0] };
        let sz: &[i32] = if frac.z.abs() < 1.0e-4 { &[0, n] } else { &[0] };
        for &na in sx {
            for &nb in sy {
                for &nc in sz {
                    if na == 0 && nb == 0 && nc == 0 {
                        continue;
                    }
                    add_atom(atom, lattice_offset(lat, na, nb, nc), 1.0);
                }
            }
        }
    }

    let bonds = build_bond_mesh(device, &instances);
    let (vertices, indices) = uv_sphere(24, 32);
    (
        vec![ShapeGroup::new(device, &vertices, &indices, &instances)],
        bonds,
    )
}

fn build_bond_mesh(device: &wgpu::Device, atoms: &[AtomInstance]) -> Option<BondMesh> {
    if atoms.len() < 2 {
        return None;
    }

    let positions: Vec<glam::Vec3> = atoms
        .iter()
        .map(|atom| {
            glam::Vec3::from_array([
                atom.center_radius[0],
                atom.center_radius[1],
                atom.center_radius[2],
            ])
        })
        .collect();
    let mut min_distance_sq = f32::INFINITY;
    for i in 0..positions.len() {
        for j in i + 1..positions.len() {
            let distance_sq = positions[i].distance_squared(positions[j]);
            if distance_sq > 1.0e-6 {
                min_distance_sq = min_distance_sq.min(distance_sq);
            }
        }
    }
    if !min_distance_sq.is_finite() {
        return None;
    }

    // Nearest-neighbour graph: enough to reveal coordination without turning
    // dense structures into an indiscriminate web.
    let cutoff_sq = min_distance_sq * 1.16;
    let radius = (min_distance_sq.sqrt() * 0.018).clamp(0.018, 0.045);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut bond_count = 0usize;
    for i in 0..positions.len() {
        for j in i + 1..positions.len() {
            let distance_sq = positions[i].distance_squared(positions[j]);
            if distance_sq <= cutoff_sq && distance_sq > 1.0e-6 {
                append_bond_cylinder(
                    &mut vertices,
                    &mut indices,
                    positions[i],
                    positions[j],
                    radius,
                );
                bond_count += 1;
                if bond_count == 4096 {
                    break;
                }
            }
        }
        if bond_count == 4096 {
            break;
        }
    }
    if indices.is_empty() {
        None
    } else {
        Some(BondMesh::new(device, &vertices, &indices))
    }
}

fn append_bond_cylinder(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    start: glam::Vec3,
    end: glam::Vec3,
    radius: f32,
) {
    const SIDES: u32 = 8;
    let axis = (end - start).normalize();
    let helper = if axis.y.abs() < 0.9 {
        glam::Vec3::Y
    } else {
        glam::Vec3::X
    };
    let tangent = axis.cross(helper).normalize();
    let bitangent = axis.cross(tangent);
    let base = vertices.len() as u32;
    for side in 0..SIDES {
        let angle = std::f32::consts::TAU * side as f32 / SIDES as f32;
        let normal = tangent * angle.cos() + bitangent * angle.sin();
        vertices.push(Vertex {
            position: (start + normal * radius).to_array(),
            normal: normal.to_array(),
        });
        vertices.push(Vertex {
            position: (end + normal * radius).to_array(),
            normal: normal.to_array(),
        });
    }
    for side in 0..SIDES {
        let next = (side + 1) % SIDES;
        let a = base + side * 2;
        let b = a + 1;
        let c = base + next * 2;
        let d = c + 1;
        indices.extend_from_slice(&[a, c, b, b, c, d]);
    }
}

fn lattice_offset(lat: &[[f32; 3]; 3], na: i32, nb: i32, nc: i32) -> [f32; 3] {
    [
        na as f32 * lat[0][0] + nb as f32 * lat[1][0] + nc as f32 * lat[2][0],
        na as f32 * lat[0][1] + nb as f32 * lat[1][1] + nc as f32 * lat[2][1],
        na as f32 * lat[0][2] + nb as f32 * lat[1][2] + nc as f32 * lat[2][2],
    ]
}

fn supercell_lines(lat: &[[f32; 3]; 3], supercell: usize) -> Vec<f32> {
    let scale = supercell as f32;
    let scaled = [
        [lat[0][0] * scale, lat[0][1] * scale, lat[0][2] * scale],
        [lat[1][0] * scale, lat[1][1] * scale, lat[1][2] * scale],
        [lat[2][0] * scale, lat[2][1] * scale, lat[2][2] * scale],
    ];
    cell_lines(&scaled)
}

fn cell_lines(lat: &[[f32; 3]; 3]) -> Vec<f32> {
    let zero = [0f32; 3];
    let a = lat[0];
    let b = lat[1];
    let c = lat[2];
    let ab = add3(a, b);
    let ac = add3(a, c);
    let bc = add3(b, c);
    let abc = add3(ab, c);
    let corners = [zero, a, b, c, ab, ac, bc, abc];
    let edges: [(usize, usize); 12] = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ];
    let mut v = Vec::with_capacity(12 * 6);
    for (i, j) in edges {
        v.extend_from_slice(&corners[i]);
        v.extend_from_slice(&corners[j]);
    }
    v
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn make_scene_uniform(cam: &OrbitCamera, asp: f32, t: f32, sys: CrystalSystem) -> SceneUniform {
    let vp = cam.view_proj(asp);
    let eye = cam.eye();
    let ac = sys.accent();
    SceneUniform {
        view_proj: vp.to_cols_array_2d(),
        eye_pos: eye.into(),
        time: t,
        accent_col: ac,
        _pad: 0.0,
    }
}

fn aspect(c: &wgpu::SurfaceConfiguration) -> f32 {
    c.width as f32 / c.height as f32
}

#[cfg(target_arch = "wasm32")]
fn canvas_size(window: &Window) -> winit::dpi::PhysicalSize<u32> {
    let Some(browser_window) = web_sys::window() else {
        return window.inner_size();
    };

    let width = browser_window
        .inner_width()
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(1280.0)
        .max(1.0);
    let height = browser_window
        .inner_height()
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(720.0)
        .max(1.0);
    let scale = browser_window.device_pixel_ratio().max(1.0);
    let physical = winit::dpi::PhysicalSize::new(
        (width * scale).round().max(1.0) as u32,
        (height * scale).round().max(1.0) as u32,
    );

    if let Some(canvas) = window.canvas() {
        canvas.set_width(physical.width);
        canvas.set_height(physical.height);
        let _ = canvas.style().set_property("width", "100vw");
        let _ = canvas.style().set_property("height", "100vh");
    }

    physical
}

#[cfg(not(target_arch = "wasm32"))]
fn canvas_size(window: &Window) -> winit::dpi::PhysicalSize<u32> {
    window.inner_size()
}

fn make_depth(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: config.width.max(1),
                height: config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

// ── Bind group layout helpers ─────────────────────────────────────────────

fn scene_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn tex_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
fn sampler_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}
fn uniform_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_single_tex(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl_1tex"),
        entries: &[tex_binding(0), sampler_binding(1)],
    })
}
fn bgl_blur(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl_blur"),
        entries: &[tex_binding(0), sampler_binding(1), uniform_binding(2)],
    })
}
fn field_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("field bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn bgl_composite(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl_comp"),
        entries: &[tex_binding(0), tex_binding(1), sampler_binding(2)],
    })
}

// ── Pipeline builders ─────────────────────────────────────────────────────

fn ds_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

fn build_atom_pl(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    fmt: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("atom"),
        source: wgpu::ShaderSource::Wgsl(ATOM_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    let vbuf = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
        ],
    };
    let ibuf = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<AtomInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 3,
            },
        ],
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("atom pl"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &sm,
            entry_point: "vs_atom",
            buffers: &[vbuf, ibuf],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &sm,
            entry_point: "fs_atom",
            targets: &[Some(wgpu::ColorTargetState {
                format: fmt,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(ds_state()),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn build_bond_pl(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    fmt: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bond"),
        source: wgpu::ShaderSource::Wgsl(BOND_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bond pl"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &sm,
            entry_point: "vs_bond",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 12,
                        shader_location: 1,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &sm,
            entry_point: "fs_bond",
            targets: &[Some(wgpu::ColorTargetState {
                format: fmt,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(ds_state()),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn build_cell_pl(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    fmt: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cell"),
        source: wgpu::ShaderSource::Wgsl(CELL_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cell pl"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &sm,
            entry_point: "vs_cell",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 12,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &sm,
            entry_point: "fs_cell",
            targets: &[Some(wgpu::ColorTargetState {
                format: fmt,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        },
        depth_stencil: Some(ds_state()),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn build_field_pl(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    fmt: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    build_field_pipeline(device, bgl, fmt, None)
}

/// The incoming half of a tour transition. Constant-factor blending lets one
/// pass render the outgoing mode opaquely and this pass interpolate the
/// independently shaded incoming mode over it.
fn build_field_fade_pl(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    fmt: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let blend = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Constant,
            dst_factor: wgpu::BlendFactor::OneMinusConstant,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        },
    };
    build_field_pipeline(device, bgl, fmt, Some(blend))
}

fn build_field_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    fmt: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("field"),
        source: wgpu::ShaderSource::Wgsl(FIELD_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("field pl"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &sm,
            entry_point: "vs_screen",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &sm,
            entry_point: "fs_field",
            targets: &[Some(wgpu::ColorTargetState {
                format: fmt,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

/// Draw one field layer. `mix` is only supplied to the constant-blend
/// transition pipeline; the opaque source pass leaves it unset.
fn field_pass(
    enc: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    clear: bool,
    mix: Option<f32>,
) {
    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(if mix.is_some() {
            "field transition incoming"
        } else {
            "field pass"
        }),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: if clear {
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    if let Some(mix) = mix {
        let mix = mix as f64;
        pass.set_blend_constant(wgpu::Color {
            r: mix,
            g: mix,
            b: mix,
            a: 1.0,
        });
    }
    pass.draw(0..3, 0..1);
}

fn fullscreen_pl(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    src: &str,
    entry: &str,
    fmt: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &sm,
            entry_point: "vs_screen",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &sm,
            entry_point: entry,
            targets: &[Some(wgpu::ColorTargetState {
                format: fmt,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn fullscreen_pass(
    enc: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pl: &wgpu::RenderPipeline,
    bg: &wgpu::BindGroup,
    clear: bool,
) {
    let load = if clear {
        wgpu::LoadOp::Clear(wgpu::Color::BLACK)
    } else {
        wgpu::LoadOp::Load
    };
    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pl);
    pass.set_bind_group(0, bg, &[]);
    pass.draw(0..3, 0..1);
}

// ── WGSL shaders ──────────────────────────────────────────────────────────

const ATOM_SHADER: &str = r#"
struct Scene {
    view_proj:  mat4x4<f32>,
    eye_pos:    vec3<f32>,
    time:       f32,
    accent_col: vec3<f32>,
}
@group(0) @binding(0) var<uniform> sc: Scene;

struct VIn {
    @location(0) mesh_pos:      vec3<f32>,
    @location(1) mesh_normal:   vec3<f32>,
    @location(2) center_radius: vec4<f32>,
    @location(3) color:         vec4<f32>,  // rgb + alpha
}
struct VOut {
    @builtin(position) clip:  vec4<f32>,
    @location(0) normal:      vec3<f32>,
    @location(1) color:       vec3<f32>,
    @location(2) world_pos:   vec3<f32>,
    @location(3) alpha:       f32,
}

@vertex fn vs_atom(v: VIn) -> VOut {
    let world = v.center_radius.xyz + v.mesh_pos * v.center_radius.w;
    var o: VOut;
    o.clip      = sc.view_proj * vec4<f32>(world, 1.0);
    o.normal    = v.mesh_normal;
    o.color     = v.color.rgb;
    o.world_pos = world;
    o.alpha     = v.color.a;
    return o;
}

@fragment fn fs_atom(f: VOut) -> @location(0) vec4<f32> {
    let n = normalize(f.normal);
    let view = normalize(sc.eye_pos - f.world_pos);
    let key = normalize(vec3<f32>(0.7, 1.0, 0.55));
    let fill = normalize(vec3<f32>(-0.8, 0.25, 0.45));

    let key_diff = max(dot(n, key), 0.0);
    let fill_diff = max(dot(n, fill), 0.0);
    let diffuse = key_diff * 0.90 + fill_diff * 0.06;

    let half_dir = normalize(key + view);
    let specular = pow(max(dot(n, half_dir), 0.0), 48.0) * 0.34;
    let rim = pow(1.0 - max(dot(n, view), 0.0), 3.0);

    let atom_color = mix(f.color, f.color * f.color, 0.45);
    let base = atom_color * (0.045 + diffuse * 0.88);
    let highlight = vec3<f32>(1.0, 0.96, 0.90) * specular;
    let edge = mix(atom_color, sc.accent_col, 0.22) * rim * 0.10;
    return vec4<f32>(base + highlight + edge, f.alpha);
}
"#;

const BOND_SHADER: &str = r#"
struct Scene {
    view_proj:  mat4x4<f32>,
    eye_pos:    vec3<f32>,
    time:       f32,
    accent_col: vec3<f32>,
}
@group(0) @binding(0) var<uniform> sc: Scene;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
}

@vertex fn vs_bond(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VOut {
    var o: VOut;
    o.clip = sc.view_proj * vec4<f32>(position, 1.0);
    o.normal = normal;
    return o;
}

@fragment fn fs_bond(f: VOut) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(0.7, 1.0, 0.55));
    let diffuse = abs(dot(normalize(f.normal), light));
    let base = mix(vec3<f32>(0.07, 0.09, 0.12), sc.accent_col * 0.35, 0.10);
    return vec4<f32>(base * (0.30 + diffuse * 0.50), 1.0);
}
"#;
const CELL_SHADER: &str = r#"
struct Scene {
    view_proj:  mat4x4<f32>,
    eye_pos:    vec3<f32>,
    time:       f32,
    accent_col: vec3<f32>,
}
@group(0) @binding(0) var<uniform> sc: Scene;

struct VOut { @builtin(position) clip: vec4<f32> }

@vertex fn vs_cell(@location(0) pos: vec3<f32>) -> VOut {
    var o: VOut;
    o.clip = sc.view_proj * vec4<f32>(pos, 1.0);
    return o;
}

@fragment fn fs_cell(f: VOut) -> @location(0) vec4<f32> {
    // The cell is a reference scaffold, not an emissive foreground object.
    let col = mix(sc.accent_col, vec3<f32>(0.72, 0.78, 0.86), 0.45);
    return vec4<f32>(col * 0.58, 0.62);
}
"#;

// ── Full-screen pass shaders ──────────────────────────────────────────────
// Rust's concat! only accepts literals, so the shared vertex shader is
// inlined into each string. The VS is identical in all four.

const BRIGHT_SHADER: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_screen(@builtin(vertex_index) vi: u32) -> VOut {
    var uv = array<vec2<f32>,3>(vec2(0.0,0.0),vec2(2.0,0.0),vec2(0.0,2.0));
    let u = uv[vi];
    var o: VOut;
    o.pos = vec4<f32>(u.x*2.0-1.0, 1.0-u.y*2.0, 0.0, 1.0);
    o.uv  = u;
    return o;
}
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@fragment fn fs_bright(f: VOut) -> @location(0) vec4<f32> {
    let c = textureSample(tex, smp, f.uv);
    let lum = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let knee = smoothstep(0.65, 1.0, lum);
    return vec4<f32>(c.rgb * knee, 1.0);
}
"#;

const BLUR_H_SHADER: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_screen(@builtin(vertex_index) vi: u32) -> VOut {
    var uv = array<vec2<f32>,3>(vec2(0.0,0.0),vec2(2.0,0.0),vec2(0.0,2.0));
    let u = uv[vi];
    var o: VOut;
    o.pos = vec4<f32>(u.x*2.0-1.0, 1.0-u.y*2.0, 0.0, 1.0);
    o.uv  = u;
    return o;
}
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct PP { width: f32, height: f32 }
@group(0) @binding(2) var<uniform> pp: PP;
@fragment fn fs_blur_h(f: VOut) -> @location(0) vec4<f32> {
    let ts = vec2<f32>(1.0/pp.width, 0.0);
    var c = textureSample(tex,smp,f.uv).rgb * 0.227027;
    c += textureSample(tex,smp,f.uv+ts*1.0).rgb*0.1945946; c += textureSample(tex,smp,f.uv-ts*1.0).rgb*0.1945946;
    c += textureSample(tex,smp,f.uv+ts*2.0).rgb*0.1216216; c += textureSample(tex,smp,f.uv-ts*2.0).rgb*0.1216216;
    c += textureSample(tex,smp,f.uv+ts*3.0).rgb*0.054054;  c += textureSample(tex,smp,f.uv-ts*3.0).rgb*0.054054;
    c += textureSample(tex,smp,f.uv+ts*4.0).rgb*0.016216;  c += textureSample(tex,smp,f.uv-ts*4.0).rgb*0.016216;
    return vec4<f32>(c, 1.0);
}
"#;

const BLUR_V_SHADER: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_screen(@builtin(vertex_index) vi: u32) -> VOut {
    var uv = array<vec2<f32>,3>(vec2(0.0,0.0),vec2(2.0,0.0),vec2(0.0,2.0));
    let u = uv[vi];
    var o: VOut;
    o.pos = vec4<f32>(u.x*2.0-1.0, 1.0-u.y*2.0, 0.0, 1.0);
    o.uv  = u;
    return o;
}
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct PP { width: f32, height: f32 }
@group(0) @binding(2) var<uniform> pp: PP;
@fragment fn fs_blur_v(f: VOut) -> @location(0) vec4<f32> {
    let ts = vec2<f32>(0.0, 1.0/pp.height);
    var c = textureSample(tex,smp,f.uv).rgb * 0.227027;
    c += textureSample(tex,smp,f.uv+ts*1.0).rgb*0.1945946; c += textureSample(tex,smp,f.uv-ts*1.0).rgb*0.1945946;
    c += textureSample(tex,smp,f.uv+ts*2.0).rgb*0.1216216; c += textureSample(tex,smp,f.uv-ts*2.0).rgb*0.1216216;
    c += textureSample(tex,smp,f.uv+ts*3.0).rgb*0.054054;  c += textureSample(tex,smp,f.uv-ts*3.0).rgb*0.054054;
    c += textureSample(tex,smp,f.uv+ts*4.0).rgb*0.016216;  c += textureSample(tex,smp,f.uv-ts*4.0).rgb*0.016216;
    return vec4<f32>(c, 1.0);
}
"#;

const COMPOSITE_SHADER: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_screen(@builtin(vertex_index) vi: u32) -> VOut {
    var uv = array<vec2<f32>,3>(vec2(0.0,0.0),vec2(2.0,0.0),vec2(0.0,2.0));
    let u = uv[vi];
    var o: VOut;
    o.pos = vec4<f32>(u.x*2.0-1.0, 1.0-u.y*2.0, 0.0, 1.0);
    o.uv  = u;
    return o;
}
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var bloom: texture_2d<f32>;
@group(0) @binding(2) var smp:   sampler;
@fragment fn fs_composite(f: VOut) -> @location(0) vec4<f32> {
    let s = textureSample(scene, smp, f.uv).rgb;
    let b = textureSample(bloom, smp, f.uv).rgb;
    let hdr = s + b * 1.8;
    let mapped = hdr / (hdr + vec3<f32>(1.0));
    let gamma  = pow(mapped, vec3<f32>(1.0/2.2));
    return vec4<f32>(gamma, 1.0);
}
"#;

// ── Feedback / self-similarity shaders ───────────────────────────────────

pub const FEEDBACK_SHADER: &str = r#"
const TAU: f32 = 6.28318530718;

// Layout MUST match `FieldUniform` / the WGSL `FU` in prelude.wgsl exactly
// (see field_and_feedback_fu_match_uniform_size test). The feedback pass only
// reads fb_*/mouse/aspect/crystal_color, but the struct must mirror the whole
// uploaded buffer or every field after the first mismatch is read at the wrong
// offset (was the "feedback = blank screen" bug after the mp-bank refactor).
struct FU {
    mp: array<vec4<f32>, 4>,
    time: f32,
    mode: u32,
    num_g: u32,
    aspect: f32,
    crystal_color: vec4<f32>,
    mouse: vec2<f32>,
    mouse_down: f32,
    fb_enabled: u32, fb_mirror: u32,
    fb_zoom: f32, fb_offset_x: f32, fb_offset_y: f32,
    fb_rotation: f32, fb_decay: f32, fb_color_shift: f32, fb_inject: f32,
    fb_fold_angle: f32, fb_saturation: f32, fb_brightness: f32, fb_blend_mode: u32,
    fb_motion_blur: f32,
}
@group(0) @binding(0) var<uniform> u: FU;
@group(0) @binding(1) var prev_accum: texture_2d<f32>;
@group(0) @binding(2) var field_rt:   texture_2d<f32>;
@group(0) @binding(3) var smp: sampler;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_screen(@builtin(vertex_index) vi: u32) -> VOut {
    var uvs = array<vec2<f32>,3>(vec2(0.0,0.0),vec2(2.0,0.0),vec2(0.0,2.0));
    let uv = uvs[vi];
    var o: VOut;
    o.pos = vec4<f32>(uv.x*2.0-1.0, 1.0-uv.y*2.0, 0.0, 1.0);
    o.uv  = uv;
    return o;
}

fn rotate_hue(c: vec3<f32>, h: f32) -> vec3<f32> {
    let k = vec3<f32>(0.57735, 0.57735, 0.57735);
    let cs = cos(h); let sn = sin(h);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let k = vec4<f32>(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    let p = mix(vec4<f32>(c.bg, k.wz), vec4<f32>(c.gb, k.xy), step(c.b, c.g));
    let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), step(p.x, c.r));
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let k = vec4<f32>(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    let p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

fn polar_fold(uv: vec2<f32>, n: f32, angle_offset: f32) -> vec2<f32> {
    let center = vec2<f32>(0.5, 0.5);
    let p = uv - center;
    let r = length(p);
    let theta = atan2(p.y, p.x) + angle_offset;
    let sector = TAU / n;
    let folded = abs(fract(theta / sector + 0.5) - 0.5) * sector;
    return vec2<f32>(cos(folded), sin(folded)) * r + center;
}

fn mirror_uv(uv: vec2<f32>, mode: u32) -> vec2<f32> {
    var v = fract(uv);
    switch mode {
        case 1u: { v.x = abs(fract(v.x * 0.5) * 2.0 - 1.0); }
        case 2u: { v.y = abs(fract(v.y * 0.5) * 2.0 - 1.0); }
        case 3u: {
            v.x = abs(fract(v.x * 0.5) * 2.0 - 1.0);
            v.y = abs(fract(v.y * 0.5) * 2.0 - 1.0);
        }
        case 4u: {
            v.x = abs(fract(v.x * 0.5) * 2.0 - 1.0);
            v.y = abs(fract(v.y * 0.5) * 2.0 - 1.0);
        }
        case 5u: { v = polar_fold(v, 3.0, u.fb_fold_angle); }
        case 6u: { v = polar_fold(v, 6.0, u.fb_fold_angle); }
        case 7u: { v = polar_fold(v, 8.0, u.fb_fold_angle); }
        case 8u: {
            v = polar_fold(v, 3.0, u.fb_fold_angle);
            v.x = abs(fract(v.x * 0.5) * 2.0 - 1.0);
            v.y = abs(fract(v.y * 0.5) * 2.0 - 1.0);
        }
        case 9u:  { v = polar_fold(v, 6.0, u.fb_fold_angle + u.time * 0.3); }
        case 10u: { v = polar_fold(v, 4.0, u.fb_fold_angle); }
        case 11u: { v = polar_fold(v, 12.0, u.fb_fold_angle); }
        default: {}
    }
    return v;
}

@fragment fn fs_feedback(f: VOut) -> @location(0) vec4<f32> {
    let c = vec2<f32>(0.5, 0.5);
    let p = f.uv - c;
    let cr = cos(u.fb_rotation); let sr = sin(u.fb_rotation);
    let rotated = vec2<f32>(cr * p.x - sr * p.y, sr * p.x + cr * p.y) / u.fb_zoom;
    let prev_uv = mirror_uv(rotated + c + vec2<f32>(u.fb_offset_x, u.fb_offset_y), u.fb_mirror);

    var prev = textureSample(prev_accum, smp, prev_uv).rgb;
    let cur  = textureSample(field_rt,   smp, f.uv).rgb;

    // HSV modulate the accumulated frame
    var hsv = rgb_to_hsv(prev);
    hsv.y = clamp(hsv.y * u.fb_saturation, 0.0, 1.0);
    hsv.z = clamp(hsv.z * u.fb_brightness, 0.0, 1.0);
    let prev_col = rotate_hue(hsv_to_rgb(hsv), u.fb_color_shift);
    let cur_col  = cur * u.fb_inject;

    // Additive terms for blend modes that need pre-multiplied prev
    let pa = prev_col * u.fb_decay;
    let eps = vec3<f32>(1e-5);

    var mixed: vec3<f32>;
    switch u.fb_blend_mode {
        // 0: lerp — Fraksl-style normalized mix, energy-preserving
        case 0u:  { mixed = mix(cur_col, prev_col, u.fb_decay); }
        // 1: add — HDR additive accumulation (original behaviour)
        case 1u:  { mixed = pa + cur_col; }
        // 2: screen
        case 2u:  { mixed = pa + cur_col - pa * cur_col; }
        // 3: multiply
        case 3u:  { mixed = pa * cur_col; }
        // 4: overlay
        case 4u:  {
            mixed = select(2.0 * pa * cur_col,
                           1.0 - 2.0 * (1.0 - pa) * (1.0 - cur_col),
                           pa > vec3<f32>(0.5));
        }
        // 5: difference
        case 5u:  { mixed = abs(pa - cur_col); }
        // 6: lighten (max)
        case 6u:  { mixed = max(pa, cur_col); }
        // 7: color burn
        case 7u:  { mixed = 1.0 - clamp((1.0 - cur_col) / (pa + eps), vec3<f32>(0.0), vec3<f32>(1.0)); }
        // 8: color dodge
        case 8u:  { mixed = clamp(cur_col / (1.0 - pa + eps), vec3<f32>(0.0), vec3<f32>(2.0)); }
        // 9: exclusion (a+b−2ab)
        case 9u:  { mixed = pa + cur_col - 2.0 * pa * cur_col; }
        // 10: hard light (overlay with layers swapped)
        case 10u: {
            mixed = select(2.0 * cur_col * pa,
                           1.0 - 2.0 * (1.0 - cur_col) * (1.0 - pa),
                           cur_col > vec3<f32>(0.5));
        }
        // 11: soft light (Pegtop formula)
        case 11u: {
            mixed = (1.0 - 2.0*cur_col)*pa*pa + 2.0*cur_col*pa;
        }
        default: { mixed = mix(cur_col, prev_col, u.fb_decay); }
    }
    // Fraksl-style MotionBlur: un-warped temporal smoothing.
    // Mixes the previous frame at the SAME pixel (no zoom/rot/mirror), so trails
    // accumulate even when the feedback loop is geometrically active. fb_motion_blur
    // in [0,1] — 0 disables, ~0.5 gives long tails, ~0.95 near-static smear.
    if (u.fb_motion_blur > 0.001) {
        let mb_prev = textureSample(prev_accum, smp, f.uv).rgb;
        mixed = mix(mixed, mb_prev, u.fb_motion_blur);
    }
    return vec4<f32>(mixed, 1.0);
}
"#;

// ── Blit color-pipeline math (CPU mirror of `fs_blit`) ────────────────────
// The feedback accumulation is Rgba16Float (linear) and is blitted to an
// sRGB swapchain view, which automatically applies the sRGB OETF on write.
// The blit shader therefore MUST NOT pre-apply gamma — doing so produces a
// "double gamma" that washes the picture to white. This Rust helper stays
// in sync with `fs_blit` so tests can detect regressions.
pub fn blit_pixel(linear: f32) -> f32 {
    // Clamp HDR into LDR range; identity for [0,1]. sRGB encoding happens
    // in the swapchain — do NOT raise to 1/2.2 here.
    linear.clamp(0.0, 1.0)
}

/// IEC 61966-2-1 sRGB OETF (linear → encoded). Mirrors what an sRGB-format
/// render target performs on write.
pub fn srgb_encode(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

pub const BLIT_SHADER: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_screen(@builtin(vertex_index) vi: u32) -> VOut {
    var uvs = array<vec2<f32>,3>(vec2(0.0,0.0),vec2(2.0,0.0),vec2(0.0,2.0));
    let uv = uvs[vi];
    var o: VOut;
    o.pos = vec4<f32>(uv.x*2.0-1.0, 1.0-uv.y*2.0, 0.0, 1.0);
    o.uv  = uv;
    return o;
}
@group(0) @binding(0) var accum: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@fragment fn fs_blit(f: VOut) -> @location(0) vec4<f32> {
    // accum is Rgba16Float (linear). Swapchain view is sRGB → it applies the
    // sRGB OETF on write. Just clamp HDR overflow into LDR and pass through;
    // any manual gamma here would double-encode and tint the picture white.
    let c = textureSample(accum, smp, f.uv).rgb;
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"#;

// ── Tests for the color/blit pipeline ────────────────────────────────────
#[cfg(test)]
mod blit_tests {
    use super::{blit_pixel, srgb_encode, BLIT_SHADER};

    /// What a pixel looks like ON-SCREEN: linear value → fs_blit → sRGB OETF.
    fn on_screen(linear: f32) -> f32 {
        srgb_encode(blit_pixel(linear))
    }

    /// The non-feedback ("direct") path renders the field straight to the sRGB
    /// swapchain — so the on-screen value is just sRGB(linear). The feedback
    /// path must agree for LDR values; otherwise it looks tinted.
    #[test]
    fn feedback_blit_matches_direct_path_for_ldr_midgray() {
        let direct = srgb_encode(0.5);
        let fb = on_screen(0.5);
        assert!(
            (fb - direct).abs() < 1e-4,
            "BUG: feedback path mid-gray ({fb:.4}) does not match direct ({direct:.4}). \
             If fb > direct + ~0.05, fs_blit is double-gamma-encoding."
        );
    }

    #[test]
    fn feedback_blit_matches_direct_path_for_dark_and_bright() {
        for &x in &[0.05_f32, 0.18, 0.25, 0.5, 0.75, 0.95] {
            let direct = srgb_encode(x);
            let fb = on_screen(x);
            assert!(
                (fb - direct).abs() < 1e-4,
                "LDR linear {x:.2}: feedback={fb:.4}, direct={direct:.4}"
            );
        }
    }

    /// Regression guard: BLIT_SHADER must not contain `1.0/2.2` (the manual
    /// gamma exponent that caused the whitish tint when the target is sRGB).
    #[test]
    fn blit_shader_does_not_apply_manual_gamma() {
        assert!(
            !BLIT_SHADER.contains("1.0/2.2") && !BLIT_SHADER.contains("1.0 / 2.2"),
            "fs_blit must not pre-apply gamma — the sRGB swapchain encodes for us"
        );
    }

    /// Regression guard: don't accidentally re-introduce Reinhard, which by
    /// itself (without manual gamma) would darken LDR mid-grays.
    #[test]
    fn blit_shader_does_not_reinhard_tone_map() {
        assert!(
            !BLIT_SHADER.contains("c / (c + vec3"),
            "fs_blit must not apply Reinhard for the LDR feedback pipeline"
        );
    }

    /// HDR values >1 from additive blend modes must not blow up or appear
    /// negative on screen — just clamp to white.
    #[test]
    fn feedback_blit_clamps_hdr_overflow() {
        for &x in &[1.0_f32, 1.5, 5.0, 100.0] {
            let v = on_screen(x);
            assert!(
                v >= 0.999 && v <= 1.0001,
                "HDR linear {x} should clamp at sRGB 1.0, got {v:.4}"
            );
        }
    }

    /// Negative values from numerical edge cases must clamp to black, not wrap.
    #[test]
    fn feedback_blit_clamps_negative_to_black() {
        let v = on_screen(-0.5);
        assert!(v.abs() < 1e-6, "negative input must clamp to 0; got {v}");
    }

    /// Sanity-check on the sRGB OETF itself.
    #[test]
    fn srgb_encode_endpoints() {
        assert!(srgb_encode(0.0).abs() < 1e-6);
        assert!((srgb_encode(1.0) - 1.0).abs() < 1e-4);
        // Mid-gray reference: linear 0.5 → ~0.7354 sRGB (well-known value).
        assert!((srgb_encode(0.5) - 0.7354).abs() < 1e-3);
    }
}

/// Every shader that binds the `FieldUniform` buffer declares its own WGSL `FU`
/// struct. If any of those drift from the Rust layout, the GPU reads each field
/// past the first mismatch at the wrong offset — which silently rendered the
/// feedback pass as a blank screen after the `mp`-bank refactor. These tests
/// pin all `FU` declarations to `size_of::<FieldUniform>()` with naga (no GPU).
#[cfg(test)]
mod uniform_layout_tests {
    use super::FieldUniform;

    /// Total byte span (incl. trailing padding) of the `struct FU` declared in
    /// `wgsl`, as computed by naga's WGSL layout rules.
    fn fu_struct_span(wgsl: &str) -> u32 {
        let module = naga::front::wgsl::parse_str(wgsl)
            .unwrap_or_else(|e| panic!("shader failed to parse:\n{}", e.emit_to_string(wgsl)));
        for (_h, ty) in module.types.iter() {
            if ty.name.as_deref() == Some("FU") {
                if let naga::TypeInner::Struct { span, .. } = &ty.inner {
                    return *span;
                }
            }
        }
        panic!("no `struct FU` found in shader");
    }

    #[test]
    fn field_and_feedback_fu_match_uniform_size() {
        let want = std::mem::size_of::<FieldUniform>() as u32;
        assert_eq!(
            fu_struct_span(crate::modes::FIELD_SHADER),
            want,
            "FIELD_SHADER `FU` struct size != size_of::<FieldUniform>()"
        );
        assert_eq!(
            fu_struct_span(super::FEEDBACK_SHADER),
            want,
            "FEEDBACK_SHADER `FU` struct size != size_of::<FieldUniform>() — \
             feedback pass would read fb_* at wrong offsets (blank-screen bug)"
        );
    }
}
