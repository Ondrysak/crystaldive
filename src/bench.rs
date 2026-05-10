//! Headless GPU benchmark harness for the field render modes.
//!
//! Used by `examples/bench_modes.rs` for human-readable timing reports and by
//! `tests/mode_perf.rs` as a CI regression guard against newly-added slow modes.

use std::time::Instant;

use bytemuck::{Pod, Zeroable};

use crate::modes::FIELD_SHADER;
use crate::poscar::{Atom, Crystal};
use crate::reciprocal::{GpuField, MAX_G};

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
    time:          f32,
    kscale:        f32,
    speed:         f32,
    field_mix:     f32,
    iso_level:     f32,
    color_shift:   f32,
    zoom:          f32,
    w_lattice:     f32,
    w_motif:       f32,
    w_band:        f32,
    mode:          u32,
    num_g:         u32,
    crystal_color: [f32; 4],
    mouse:         [f32; 2],
    mouse_down:    f32,
    aspect:        f32,
}

fn default_uniform(mode: u32, time: f32, aspect: f32) -> FieldUniform {
    FieldUniform {
        time, kscale: 1.4, speed: 0.3, field_mix: 0.55,
        iso_level: 0.5, color_shift: 0.0, zoom: 1.0,
        w_lattice: 1.0, w_motif: 0.6, w_band: 0.4,
        mode, num_g: 0,
        crystal_color: [0.5, 0.7, 1.0, 0.0],
        mouse: [0.5, 0.5], mouse_down: 0.0, aspect,
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
