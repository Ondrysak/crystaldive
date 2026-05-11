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

pub const MODE_NAMES: &[&str] = &[
    "3D ISO", "BZ SLICE", "FERMI", "DENSITY", "NODAL", "PHASE",
    "STRIPES", "WARP", "LINKS", "XRD", "RECIP", "NONEUC",
    "PHONON", "MOIRE", "EWALD", "WANNIER", "MAGNETIC", "DISPERSION",
    "KIKUCHI", "DEFECT", "BAND SURFACE", "SPIN TEXTURE",
    "BZ PATH", "CDW", "QUASICRYSTAL", "THERMAL", "DOMAIN WALL",
    "FRACTURE",
    "BERRY", "HOFSTADTER", "STM", "SPECTRAL", "VORTEX KNOT",
    "BLOCH WAVE", "PLASMON", "NEMATIC",
];

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

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FieldUniform {
    time:           f32,
    kscale:         f32,
    speed:          f32,
    field_mix:      f32,
    iso_level:      f32,
    color_shift:    f32,
    zoom:           f32,
    w_lattice:      f32,
    w_motif:        f32,
    w_band:         f32,
    mode:           u32,
    num_g:          u32,
    crystal_color:  [f32; 4],
    mouse:          [f32; 2],
    mouse_down:     f32,
    aspect:         f32,
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
    _pad:           [f32; 3],
}

fn default_uniform(mode: u32, time: f32, aspect: f32) -> FieldUniform {
    FieldUniform {
        time, kscale: 1.4, speed: 0.3, field_mix: 0.55,
        iso_level: 0.5, color_shift: 0.0, zoom: 1.0,
        w_lattice: 1.0, w_motif: 0.6, w_band: 0.4,
        mode, num_g: 0,
        crystal_color: [0.5, 0.7, 1.0, 0.0],
        mouse: [0.5, 0.5], mouse_down: 0.0, aspect,
        fb_enabled: 0, fb_mirror: 0,
        fb_zoom: 1.0, fb_offset_x: 0.0, fb_offset_y: 0.0,
        fb_rotation: 0.0, fb_decay: 0.0, fb_color_shift: 0.0, fb_inject: 1.0,
        fb_fold_angle: 0.0, fb_saturation: 1.0, fb_brightness: 1.0, fb_blend_mode: 0,
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

    let g_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("g_tex"),
        size: wgpu::Extent3d { width: MAX_G as u32, height: 2, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let g_view = g_texture.create_view(&Default::default());

    let mut field = GpuField::from_crystal(&nacl(), 3);
    field.seed_kpoint([0.0, 0.0, 0.0], 1.0);
    let packed = field.pack();
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &g_texture, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&packed),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some((MAX_G * 4 * 4) as u32),
            rows_per_image: Some(2),
        },
        wgpu::Extent3d { width: MAX_G as u32, height: 2, depth_or_array_layers: 1 },
    );
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
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("field bg"), layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&g_view) },
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

    // ── G-texture ─────────────────────────────────────────────────────────
    let g_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("g_tex"),
        size: wgpu::Extent3d { width: MAX_G as u32, height: 2, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let g_view = g_texture.create_view(&Default::default());
    let mut field = GpuField::from_crystal(&nacl(), 3);
    field.seed_kpoint([0.0, 0.0, 0.0], 1.0);
    let packed = field.pack();
    queue.write_texture(
        wgpu::ImageCopyTexture { texture: &g_texture, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        cast_slice(&packed),
        wgpu::ImageDataLayout { offset: 0,
            bytes_per_row: Some((MAX_G * 4 * 4) as u32), rows_per_image: Some(2) },
        wgpu::Extent3d { width: MAX_G as u32, height: 2, depth_or_array_layers: 1 },
    );

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
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                },
                count: None,
            },
        ],
    });
    let field_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("field bg"), layout: &field_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&g_view) },
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
        // Sanity: the mode name table must match the count rendered by the shader.
        // After CLOUD/ORBITAL/NEUTRON removal we expect exactly 36.
        assert_eq!(MODE_NAMES.len(), 36);
    }
}
