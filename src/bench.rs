//! Headless GPU benchmark harness for the field render modes.
//!
//! Used by `examples/bench_modes.rs` for human-readable timing reports and by
//! `tests/mode_perf.rs` as a CI regression guard against newly-added slow modes.

use std::time::Instant;

use bytemuck::{Pod, Zeroable};

use crate::modes::FIELD_SHADER;
use crate::poscar::{Atom, Crystal};
use crate::reciprocal::{GpuField, MAX_G};
use crate::renderer::{FEEDBACK_SHADER, BLIT_SHADER};

pub use crate::modes::MODE_NAMES_SLICE as MODE_NAMES;

#[derive(Debug, Clone)]
pub struct ModeTiming {
    pub idx:  usize,
    pub name: &'static str,
    pub ms:   f64,
}

pub struct BenchOpts {
    pub width:   u32,
    pub height:  u32,
    pub warmup:  u32,
    pub iters:   u32,
    /// Force the wgpu fallback (software) adapter — useful for CI without a real GPU.
    pub force_fallback: bool,
}

impl Default for BenchOpts {
    fn default() -> Self {
        Self { width: 1280, height: 720, warmup: 8, iters: 120, force_fallback: false }
    }
}

// Mirror of `renderer::FieldUniform` — layout must match the WGSL `FU` struct.
// `mp` is the 16-slot per-mode param bank, vec4-packed (std140 stride).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FieldUniform {
    mp:             [[f32; 4]; 4],
    time:           f32,
    mode:           u32,
    num_g:          u32,
    aspect:         f32,
    crystal_color:  [f32; 4],
    mouse:          [f32; 2],
    mouse_down:     f32,
    fb_enabled:     u32,
    fb_mirror:      u32,
    fb_zoom:        f32,
    fb_offset_x:    f32,
    fb_offset_y:    f32,
    fb_rotation:    f32,
    fb_decay:       f32,
    fb_color_shift: f32,
    fb_inject:      f32,
    fb_fold_angle:  f32,
    fb_saturation:  f32,
    fb_brightness:  f32,
    fb_blend_mode:  u32,
    fb_motion_blur: f32,
    _pad:           [f32; 3],
}

fn default_uniform(mode: u32, time: f32, aspect: f32) -> FieldUniform {
    // Canonical generator defaults in slots 0..8 (kscale, speed, field_mix,
    // iso_level, color_shift, zoom, w_lattice, w_motif, w_band); rest 0.
    FieldUniform {
        mp: [
            [1.4, 0.3, 0.55, 0.5],
            [0.0, 1.0, 1.0, 0.6],
            [0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        time,
        mode, num_g: 0,
        aspect,
        crystal_color: [0.5, 0.7, 1.0, 0.0],
        mouse: [0.5, 0.5], mouse_down: 0.0,
        fb_enabled: 0, fb_mirror: 0,
        fb_zoom: 1.0, fb_offset_x: 0.0, fb_offset_y: 0.0,
        fb_rotation: 0.0, fb_decay: 0.85, fb_color_shift: 0.0, fb_inject: 1.0,
        fb_fold_angle: 0.0, fb_saturation: 1.0, fb_brightness: 1.0, fb_blend_mode: 0,
        fb_motion_blur: 0.0,
        _pad: [0.0; 3],
    }
}

fn nacl() -> Crystal {
    Crystal {
        lattice: [[5.64, 0.0, 0.0], [0.0, 5.64, 0.0], [0.0, 0.0, 5.64]],
        atoms: vec![
            Atom { species: "Na".into(), pos_cart: [0.0, 0.0, 0.0] },
            Atom { species: "Cl".into(), pos_cart: [2.82, 2.82, 2.82] },
        ],
    }
}

/// Run the benchmark across all modes. Returns timings in mode-index order.
///
/// Returns `Err` if no compatible adapter is available.
pub fn run(opts: &BenchOpts) -> Result<(String, Vec<ModeTiming>), String> {
    pollster::block_on(run_async(opts))
}

async fn run_async(opts: &BenchOpts) -> Result<(String, Vec<ModeTiming>), String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: opts.force_fallback,
        })
        .await
        .ok_or_else(|| "no GPU adapter available".to_string())?;

    let info = adapter.get_info();
    let info_str = format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("bench"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .map_err(|e| format!("device creation failed: {e}"))?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench target"),
        size: wgpu::Extent3d { width: opts.width, height: opts.height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());

    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform"),
        size:  std::mem::size_of::<FieldUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let g_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("g_block"),
        size:  (MAX_G * 4 * 2 * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut field = GpuField::from_crystal(&nacl(), 3);
    field.seed_kpoint([0.0, 0.0, 0.0], 1.0);
    let packed = field.pack();
    queue.write_buffer(&g_buf, 0, bytemuck::cast_slice(&packed));
    let num_g = field.count as u32;

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("field bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("field bg"), layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: g_buf.as_entire_binding() },
        ],
    });

    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("field"),
        source: wgpu::ShaderSource::Wgsl(FIELD_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("field pl"), layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &sm, entry_point: "vs_screen", buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &sm, entry_point: "fs_field",
            targets: &[Some(wgpu::ColorTargetState {
                format, blend: None, write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None, cache: None,
    });

    let aspect = opts.width as f32 / opts.height.max(1) as f32;
    let mut results = Vec::with_capacity(MODE_NAMES.len());

    for (mode_idx, mode_name) in MODE_NAMES.iter().enumerate() {
        // Warmup
        for i in 0..opts.warmup {
            let mut u = default_uniform(mode_idx as u32, i as f32 * 0.016, aspect);
            u.num_g = num_g;
            queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&u));
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("warmup"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view, resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None, timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            queue.submit([enc.finish()]);
        }
        device.poll(wgpu::Maintain::Wait);

        // Timed
        let t0 = Instant::now();
        for i in 0..opts.iters {
            let mut u = default_uniform(mode_idx as u32, (opts.warmup + i) as f32 * 0.016, aspect);
            u.num_g = num_g;
            queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&u));
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bench"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view, resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None, timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            queue.submit([enc.finish()]);
        }
        device.poll(wgpu::Maintain::Wait);
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / opts.iters as f64;
        results.push(ModeTiming { idx: mode_idx, name: mode_name, ms });
    }

    Ok((info_str, results))
}

/// Headless feedback smoke test: renders N frames with fb_enabled=1 and checks
/// that at least one output pixel is non-zero (proves the feedback path produces output).
/// Returns Err if no GPU is available. Returns Ok(max_pixel_value).
pub fn run_feedback_smoke(frames: u32, w: u32, h: u32) -> Result<u8, String> {
    pollster::block_on(run_feedback_smoke_async(frames, w, h))
}

async fn run_feedback_smoke_async(frames: u32, w: u32, h: u32) -> Result<u8, String> {
    use bytemuck::cast_slice;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(), ..Default::default()
    });
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None, force_fallback_adapter: false,
    }).await.ok_or_else(|| "no GPU adapter".to_string())?;

    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("fb_smoke"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(), memory_hints: Default::default(),
    }, None).await.map_err(|e| format!("{e}"))?;

    const FB_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    const OUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    // ── Uniform buffer ────────────────────────────────────────────────────
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform"),
        size: std::mem::size_of::<FieldUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── G-block uniform buffer ────────────────────────────────────────────
    let g_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("g_block"),
        size: (MAX_G * 4 * 2 * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut field = GpuField::from_crystal(&nacl(), 3);
    field.seed_kpoint([0.0, 0.0, 0.0], 1.0);
    let packed = field.pack();
    queue.write_buffer(&g_buf, 0, cast_slice(&packed));

    // ── Field bind group layout ───────────────────────────────────────────
    let field_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("field bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });
    let field_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("field bg"), layout: &field_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: g_buf.as_entire_binding() },
        ],
    });

    // ── Feedback BGL ──────────────────────────────────────────────────────
    let fb_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fb bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
        ],
    });
    let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("blit bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
        ],
    });

    // ── Textures ──────────────────────────────────────────────────────────
    let make_tex = |label: &str, fmt: wgpu::TextureFormat| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default())
    };
    let field_rt = make_tex("field_rt", FB_FORMAT);
    let accum_a  = make_tex("accum_a",  FB_FORMAT);
    let accum_b  = make_tex("accum_b",  FB_FORMAT);
    let out_tex  = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("out"),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2, format: OUT_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_view = out_tex.create_view(&Default::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat, address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let make_fb_bg = |prev: &wgpu::TextureView, field: &wgpu::TextureView| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fb bg"), layout: &fb_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(prev) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(field) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        })
    };
    let make_blit_bg = |accum: &wgpu::TextureView| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit bg"), layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(accum) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        })
    };
    let bg_f_then_a = make_fb_bg(&accum_a, &field_rt);
    let bg_f_then_b = make_fb_bg(&accum_b, &field_rt);
    let bg_blit_a   = make_blit_bg(&accum_a);
    let bg_blit_b   = make_blit_bg(&accum_b);

    // ── Pipelines ─────────────────────────────────────────────────────────
    let build_pl = |src: &str, entry: &str, bgl: &wgpu::BindGroupLayout, fmt: wgpu::TextureFormat| {
        let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry), source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[bgl], push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &sm, entry_point: "vs_screen", buffers: &[],
                compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &sm, entry_point: entry,
                targets: &[Some(wgpu::ColorTargetState { format: fmt, blend: None,
                    write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None, multisample: wgpu::MultisampleState::default(),
            multiview: None, cache: None,
        })
    };
    let field_pl    = build_pl(FIELD_SHADER, "fs_field",    &field_bgl, FB_FORMAT);
    let feedback_pl = build_pl(FEEDBACK_SHADER, "fs_feedback", &fb_bgl, FB_FORMAT);
    let blit_pl     = build_pl(BLIT_SHADER, "fs_blit", &blit_bgl, OUT_FORMAT);

    // ── Readback buffer ───────────────────────────────────────────────────
    let aligned_row = ((w * 4) + 255) & !255;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (aligned_row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // ── Render N frames ───────────────────────────────────────────────────
    let mut parity = false;
    for i in 0..frames {
        let mut u = default_uniform(4, i as f32 * 0.016, w as f32 / h.max(1) as f32);
        u.num_g     = field.count as u32;
        u.fb_enabled = 1;
        u.fb_zoom    = 0.97;
        u.fb_decay   = 0.90;
        u.fb_inject  = 1.0;
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&u));

        let mut enc = device.create_command_encoder(&Default::default());

        // field → field_rt
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("field→rt"), color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &field_rt, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })], depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&field_pl);
            pass.set_bind_group(0, &field_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // feedback composite
        let (dst_accum, fb_bg) = if parity { (&accum_b, &bg_f_then_a) } else { (&accum_a, &bg_f_then_b) };
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fb composite"), color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dst_accum, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })], depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&feedback_pl);
            pass.set_bind_group(0, fb_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // blit to out
        let blit_bg = if parity { &bg_blit_b } else { &bg_blit_a };
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"), color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out_view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })], depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&blit_pl);
            pass.set_bind_group(0, blit_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        parity = !parity;
        queue.submit([enc.finish()]);
    }

    // ── Readback last frame ───────────────────────────────────────────────
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture { texture: &out_tex, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::ImageCopyBuffer { buffer: &readback,
            layout: wgpu::ImageDataLayout { offset: 0,
                bytes_per_row: Some(aligned_row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let raw = slice.get_mapped_range();
    let max_val = raw.iter().copied().max().unwrap_or(0);
    drop(raw);
    readback.unmap();

    Ok(max_val)
}

/// Find modes whose timing exceeds `max_ratio × median(all modes)`.
pub fn slow_modes(results: &[ModeTiming], max_ratio: f64) -> (f64, Vec<ModeTiming>) {
    let mut all_ms: Vec<f64> = results.iter().map(|r| r.ms).collect();
    all_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = all_ms[all_ms.len() / 2];
    let cutoff = median * max_ratio;
    let slow: Vec<_> = results.iter().filter(|r| r.ms > cutoff).cloned().collect();
    (median, slow)
}

// ── Headless clip rendering ──────────────────────────────────────────────
//
// `render_clip` is the workhorse behind `crystal-viz --render preset.json …`:
// it spins up an offscreen wgpu device sized to the requested resolution,
// renders one frame per fps tick by calling `eval(t)` to get a FieldUniform,
// copies the framebuffer back, and writes a PNG per frame.
//
// Feedback path is intentionally NOT wired in here — the headless renderer
// does the single-pass field shader only. Adding feedback would mean keeping a
// persistent ping-pong texture across frames; doable but out of scope for v1.

pub struct ClipOpts {
    pub width:    u32,
    pub height:   u32,
    pub fps:      u32,
    pub duration: f32,         // seconds
    pub start:    f32,         // seconds offset added to t
    pub out_dir:  std::path::PathBuf,
}

pub fn render_clip<F>(opts: ClipOpts, crystal: &Crystal, eval: F) -> Result<(), String>
where
    F: FnMut(f32) -> crate::renderer::FieldUniform,
{
    pollster::block_on(render_clip_async(opts, crystal, eval))
}

async fn render_clip_async<F>(opts: ClipOpts, crystal: &Crystal, mut eval: F) -> Result<(), String>
where
    F: FnMut(f32) -> crate::renderer::FieldUniform,
{
    type FU = crate::renderer::FieldUniform;
    use std::io::Write;
    std::fs::create_dir_all(&opts.out_dir).map_err(|e| format!("mkdir {}: {e}", opts.out_dir.display()))?;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(), ..Default::default()
    });
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None, force_fallback_adapter: false,
    }).await.ok_or_else(|| "no GPU adapter".to_string())?;
    let info = adapter.get_info();
    eprintln!("GPU: {} ({:?}, {:?})", info.name, info.backend, info.device_type);

    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("render-clip"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: Default::default(),
    }, None).await.map_err(|e| format!("device: {e}"))?;

    // sRGB output: matches the live-app surface format so the offline renders
    // look the same as what the user sees. The GPU encodes shader values into
    // sRGB on write; we copy those raw bytes verbatim into a PNG, which is the
    // sRGB colour space by default.
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("clip target"),
        size: wgpu::Extent3d { width: opts.width, height: opts.height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());

    // Per-row alignment requirement for buffer→texture copies.
    let bytes_per_pixel = 4u32;
    let unpadded_row = opts.width * bytes_per_pixel;
    let row_align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_row = unpadded_row.div_ceil(row_align) * row_align;
    let buf_size   = (padded_row as u64) * (opts.height as u64);

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform"),
        size: std::mem::size_of::<FU>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let g_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("g_block"),
        size: (MAX_G * 4 * 2 * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut field = GpuField::from_crystal(crystal, 3);
    field.seed_kpoint([0.0, 0.0, 0.0], 1.0);
    let num_g = field.count as u32;
    let packed = field.pack();
    queue.write_buffer(&g_buf, 0, bytemuck::cast_slice(&packed));

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clip bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            },
        ],
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("clip bg"), layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: g_buf.as_entire_binding() },
        ],
    });
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("field"),
        source: wgpu::ShaderSource::Wgsl(FIELD_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("clip pl"), layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &sm, entry_point: "vs_screen", buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &sm, entry_point: "fs_field",
            targets: &[Some(wgpu::ColorTargetState {
                format, blend: None, write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None, multisample: wgpu::MultisampleState::default(),
        multiview: None, cache: None,
    });

    let total_frames = (opts.duration * opts.fps as f32).ceil() as u32;
    eprintln!(
        "Rendering {} frames at {}×{} @ {}fps ({:.2}s)…",
        total_frames, opts.width, opts.height, opts.fps, opts.duration,
    );

    let t_start = Instant::now();
    for frame_idx in 0..total_frames {
        let t = opts.start + frame_idx as f32 / opts.fps as f32;
        let mut u = eval(t);
        // Caller can't know num_g (it's set inside this fn from the crystal),
        // so we patch it on every frame.
        u.num_g = num_g;
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&u));

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clip"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &target, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(opts.height),
                },
            },
            wgpu::Extent3d { width: opts.width, height: opts.height, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);

        // Map readback and write PNG.
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().map_err(|e| format!("recv: {e}"))?
            .map_err(|e| format!("map: {e:?}"))?;
        let data = slice.get_mapped_range();

        // Strip per-row padding.
        let mut pixels = Vec::with_capacity((unpadded_row * opts.height) as usize);
        for row in 0..opts.height {
            let off = (row * padded_row) as usize;
            pixels.extend_from_slice(&data[off..off + unpadded_row as usize]);
        }
        drop(data);
        readback.unmap();

        // Encode PNG (image::Rgba8 + RgbaImage from raw).
        let img = image::RgbaImage::from_raw(opts.width, opts.height, pixels)
            .ok_or_else(|| "RgbaImage::from_raw: size mismatch".to_string())?;
        let path = opts.out_dir.join(format!("frame_{frame_idx:06}.png"));
        img.save(&path).map_err(|e| format!("save {}: {e}", path.display()))?;

        if frame_idx % 30 == 0 || frame_idx + 1 == total_frames {
            let pct = 100.0 * (frame_idx + 1) as f32 / total_frames as f32;
            eprint!("\r  frame {:>5}/{}  ({:.0}%)", frame_idx + 1, total_frames, pct);
            let _ = std::io::stderr().flush();
        }
    }
    eprintln!();
    let elapsed = t_start.elapsed().as_secs_f64();
    eprintln!(
        "Done in {:.1}s ({:.1} render-fps). Frames at: {}",
        elapsed, total_frames as f64 / elapsed, opts.out_dir.display(),
    );
    eprintln!();
    eprintln!("To encode the result with ffmpeg, run:");
    eprintln!(
        "  ffmpeg -framerate {} -i {}/frame_%06d.png -c:v libx264 \\",
        opts.fps,
        opts.out_dir.display(),
    );
    eprintln!(
        "         -pix_fmt yuv420p -crf 18 -preset slow {}.mp4",
        opts.out_dir.file_stem().and_then(|s| s.to_str()).unwrap_or("clip"),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(idx: usize, ms: f64) -> ModeTiming {
        ModeTiming { idx, name: "x", ms }
    }

    #[test]
    fn slow_modes_flags_outliers() {
        // 5-element set with median 1.0; 50.0 ms is 50× — way over a 10× cutoff.
        let r = vec![t(0, 0.5), t(1, 0.8), t(2, 1.0), t(3, 1.5), t(4, 50.0)];
        let (median, slow) = slow_modes(&r, 10.0);
        assert_eq!(median, 1.0);
        assert_eq!(slow.len(), 1);
        assert_eq!(slow[0].idx, 4);
    }

    #[test]
    fn slow_modes_empty_when_clean() {
        let r = vec![t(0, 0.8), t(1, 1.0), t(2, 1.2), t(3, 2.5)];
        let (_, slow) = slow_modes(&r, 10.0);
        assert!(slow.is_empty());
    }

    #[test]
    fn slow_modes_boundary_strict() {
        // Exactly at the cutoff is NOT slow (filter uses `> cutoff`).
        let r = vec![t(0, 1.0), t(1, 1.0), t(2, 10.0)];
        let (_, slow) = slow_modes(&r, 10.0);
        assert!(slow.is_empty());
    }

    #[test]
    fn mode_names_count_matches_dispatch() {
        // Sanity: the mode name table must match the count rendered by the shader
        // (dispatch cases 0..35).
        assert_eq!(MODE_NAMES.len(), 36);
    }
}
