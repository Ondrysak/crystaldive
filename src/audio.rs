//! Microphone capture + real-time FFT band analysis.
//!
//! Native builds use `cpal`. The browser build currently exposes the same
//! types but disables mic capture so the WebGPU app can run without WebAudio
//! permission plumbing.

#[cfg(target_arch = "wasm32")]
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
pub struct AudioBands {
    pub amplitude: f32,
    pub bass: f32,
    pub mid: f32,
    pub treble: f32,
}

#[cfg(target_arch = "wasm32")]
pub struct AudioCapture {
    pub bands: Arc<Mutex<AudioBands>>,
}

#[cfg(target_arch = "wasm32")]
impl AudioCapture {
    pub fn start() -> Option<Self> {
        None
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::AudioBands;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use rustfft::{num_complex::Complex, FftPlanner};
    use std::sync::{Arc, Mutex};

    const FFT_SIZE: usize = 512;
    const ATK: f32 = 0.35;
    const REL: f32 = 0.06;

    pub struct AudioCapture {
        pub bands: Arc<Mutex<AudioBands>>,
        _stream: cpal::Stream,
    }

    impl AudioCapture {
        pub fn start() -> Option<Self> {
            let host = cpal::default_host();
            let device = host.default_input_device()?;
            let config = device.default_input_config().ok()?;

            let sr = config.sample_rate().0 as f32;
            let ch = config.channels() as usize;
            let fmt = config.sample_format();
            let sc = cpal::StreamConfig {
                channels: config.channels(),
                sample_rate: config.sample_rate(),
                buffer_size: cpal::BufferSize::Default,
            };

            let bands: Arc<Mutex<AudioBands>> = Arc::new(Mutex::new(AudioBands::default()));
            let ring: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(FFT_SIZE);

            let stream = build_stream(&device, &sc, fmt, ring, Arc::clone(&bands), fft, sr, ch)?;
            stream.play().ok()?;
            log::info!("audio: mic open  ({} ch @ {sr:.0} Hz)", ch);
            Some(AudioCapture { bands, _stream: stream })
        }
    }

    fn err_fn(e: cpal::StreamError) {
        log::error!("audio stream: {e}");
    }

    fn build_stream(
        device: &cpal::Device,
        sc: &cpal::StreamConfig,
        fmt: cpal::SampleFormat,
        ring: Arc<Mutex<Vec<f32>>>,
        bands: Arc<Mutex<AudioBands>>,
        fft: Arc<dyn rustfft::Fft<f32>>,
        sr: f32,
        ch: usize,
    ) -> Option<cpal::Stream> {
        use cpal::SampleFormat::*;
        match fmt {
            F32 => {
                let (r, b, f) = (Arc::clone(&ring), Arc::clone(&bands), Arc::clone(&fft));
                device
                    .build_input_stream(
                        sc,
                        move |data: &[f32], _| {
                            let mono = to_mono_f32(data, ch);
                            analyse(&mono, &r, &b, &*f, sr);
                        },
                        err_fn,
                        None,
                    )
                    .ok()
            }
            I16 => {
                let (r, b, f) = (Arc::clone(&ring), Arc::clone(&bands), Arc::clone(&fft));
                device
                    .build_input_stream(
                        sc,
                        move |data: &[i16], _| {
                            let conv: Vec<f32> =
                                data.iter().map(|&s| s as f32 / 32_768.0).collect();
                            let mono = to_mono_f32(&conv, ch);
                            analyse(&mono, &r, &b, &*f, sr);
                        },
                        err_fn,
                        None,
                    )
                    .ok()
            }
            U16 => {
                let (r, b, f) = (Arc::clone(&ring), Arc::clone(&bands), Arc::clone(&fft));
                device
                    .build_input_stream(
                        sc,
                        move |data: &[u16], _| {
                            let conv: Vec<f32> =
                                data.iter().map(|&s| s as f32 / 32_768.0 - 1.0).collect();
                            let mono = to_mono_f32(&conv, ch);
                            analyse(&mono, &r, &b, &*f, sr);
                        },
                        err_fn,
                        None,
                    )
                    .ok()
            }
            other => {
                log::warn!("audio: unsupported sample format {other:?}");
                None
            }
        }
    }

    fn to_mono_f32(data: &[f32], ch: usize) -> Vec<f32> {
        data.chunks(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect()
    }

    fn analyse(
        mono: &[f32],
        ring: &Mutex<Vec<f32>>,
        bands: &Mutex<AudioBands>,
        fft: &dyn rustfft::Fft<f32>,
        sr: f32,
    ) {
        {
            let mut r = ring.lock().unwrap();
            r.extend_from_slice(mono);
            if r.len() < FFT_SIZE {
                return;
            }
        }

        let chunk: Vec<f32> = {
            let mut r = ring.lock().unwrap();
            r.drain(..FFT_SIZE).collect()
        };

        let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / FFT_SIZE as f32).sqrt();
        let mut buf: Vec<Complex<f32>> = chunk
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.5
                    - 0.5
                        * (2.0 * std::f32::consts::PI * i as f32 / (FFT_SIZE - 1) as f32).cos();
                Complex { re: s * w, im: 0.0 }
            })
            .collect();
        fft.process(&mut buf);

        let hpb = sr / FFT_SIZE as f32;
        let (mut b_p, mut b_n) = (0.0f32, 0u32);
        let (mut m_p, mut m_n) = (0.0f32, 0u32);
        let (mut t_p, mut t_n) = (0.0f32, 0u32);
        for (i, bin) in buf.iter().enumerate().take(FFT_SIZE / 2).skip(1) {
            let hz = i as f32 * hpb;
            let p = bin.norm_sqr();
            if hz < 250.0 {
                b_p += p;
                b_n += 1;
            } else if hz < 2500.0 {
                m_p += p;
                m_n += 1;
            } else {
                t_p += p;
                t_n += 1;
            }
        }

        let nrm = |p: f32, n: u32| -> f32 {
            if n == 0 {
                0.0
            } else {
                ((p / n as f32).sqrt() * 5.0).clamp(0.0, 1.0)
            }
        };

        let new_amp = (rms * 12.0).clamp(0.0, 1.0);
        let new_bass = nrm(b_p, b_n);
        let new_mid = nrm(m_p, m_n);
        let new_treble = nrm(t_p, t_n);

        let smooth = |old: f32, new: f32| -> f32 {
            let a = if new > old { ATK } else { REL };
            old + a * (new - old)
        };

        let mut bnd = bands.lock().unwrap();
        bnd.amplitude = smooth(bnd.amplitude, new_amp);
        bnd.bass = smooth(bnd.bass, new_bass);
        bnd.mid = smooth(bnd.mid, new_mid);
        bnd.treble = smooth(bnd.treble, new_treble);
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::AudioCapture;
