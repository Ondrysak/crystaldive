//! Microphone capture + real-time FFT band analysis.
//!
//! Spawns a cpal input stream on the default mic. Fills an `AudioBands`
//! shared value with smoothed amplitude, bass, mid, and treble levels
//! (each 0..1) that the render loop reads every frame.

use std::sync::{Arc, Mutex};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{FftPlanner, num_complex::Complex};

const FFT_SIZE: usize = 512;
const ATK: f32 = 0.35;   // fast attack (new > old)
const REL: f32 = 0.06;   // slow release (new < old)

// ── Shared output ─────────────────────────────────────────────────────────

#[derive(Clone, Default)]
pub struct AudioBands {
    pub amplitude: f32,   // RMS  0..1
    pub bass:      f32,   // <250 Hz
    pub mid:       f32,   // 250..2500 Hz
    pub treble:    f32,   // >2500 Hz
}

// ── Public handle ─────────────────────────────────────────────────────────

pub struct AudioCapture {
    pub bands: Arc<Mutex<AudioBands>>,
    _stream:   cpal::Stream,   // must stay alive
}

impl AudioCapture {
    /// Try to open the default input device. Returns None on any error so the
    /// rest of the app continues working without audio.
    pub fn start() -> Option<Self> {
        let host   = cpal::default_host();
        let device = host.default_input_device()?;
        let config = device.default_input_config().ok()?;

        let sr  = config.sample_rate().0 as f32;
        let ch  = config.channels()    as usize;
        let fmt = config.sample_format();
        let sc  = cpal::StreamConfig {
            channels:    config.channels(),
            sample_rate: config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        let bands: Arc<Mutex<AudioBands>> = Arc::new(Mutex::new(AudioBands::default()));
        let ring:  Arc<Mutex<Vec<f32>>>   = Arc::new(Mutex::new(Vec::new()));

        let mut pl = FftPlanner::<f32>::new();
        let fft    = pl.plan_fft_forward(FFT_SIZE);

        let stream = build_stream(&device, &sc, fmt, ring, Arc::clone(&bands), fft, sr, ch)?;
        stream.play().ok()?;
        log::info!("audio: mic open  ({} ch @ {sr:.0} Hz)", ch);
        Some(AudioCapture { bands, _stream: stream })
    }
}

// ── Stream builder ────────────────────────────────────────────────────────

fn err_fn(e: cpal::StreamError) { log::error!("audio stream: {e}"); }

fn build_stream(
    device: &cpal::Device,
    sc:     &cpal::StreamConfig,
    fmt:    cpal::SampleFormat,
    ring:   Arc<Mutex<Vec<f32>>>,
    bands:  Arc<Mutex<AudioBands>>,
    fft:    Arc<dyn rustfft::Fft<f32>>,
    sr:     f32,
    ch:     usize,
) -> Option<cpal::Stream> {
    use cpal::SampleFormat::*;
    match fmt {
        F32 => {
            let (r, b, f) = (Arc::clone(&ring), Arc::clone(&bands), Arc::clone(&fft));
            device.build_input_stream(sc, move |data: &[f32], _| {
                let mono = to_mono_f32(data, ch);
                analyse(&mono, &r, &b, &*f, sr);
            }, err_fn, None).ok()
        }
        I16 => {
            let (r, b, f) = (Arc::clone(&ring), Arc::clone(&bands), Arc::clone(&fft));
            device.build_input_stream(sc, move |data: &[i16], _| {
                let conv: Vec<f32> = data.iter().map(|&s| s as f32 / 32_768.0).collect();
                let mono = to_mono_f32(&conv, ch);
                analyse(&mono, &r, &b, &*f, sr);
            }, err_fn, None).ok()
        }
        U16 => {
            let (r, b, f) = (Arc::clone(&ring), Arc::clone(&bands), Arc::clone(&fft));
            device.build_input_stream(sc, move |data: &[u16], _| {
                let conv: Vec<f32> = data.iter().map(|&s| s as f32 / 32_768.0 - 1.0).collect();
                let mono = to_mono_f32(&conv, ch);
                analyse(&mono, &r, &b, &*f, sr);
            }, err_fn, None).ok()
        }
        other => {
            log::warn!("audio: unsupported sample format {other:?}");
            None
        }
    }
}

// ── DSP ───────────────────────────────────────────────────────────────────

fn to_mono_f32(data: &[f32], ch: usize) -> Vec<f32> {
    data.chunks(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

fn analyse(
    mono:  &[f32],
    ring:  &Mutex<Vec<f32>>,
    bands: &Mutex<AudioBands>,
    fft:   &dyn rustfft::Fft<f32>,
    sr:    f32,
) {
    {
        let mut r = ring.lock().unwrap();
        r.extend_from_slice(mono);
        if r.len() < FFT_SIZE { return; }
    }

    let chunk: Vec<f32> = {
        let mut r = ring.lock().unwrap();
        r.drain(..FFT_SIZE).collect()
    };

    // RMS amplitude
    let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / FFT_SIZE as f32).sqrt();

    // Hann window + forward FFT
    let mut buf: Vec<Complex<f32>> = chunk.iter().enumerate().map(|(i, &s)| {
        let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (FFT_SIZE - 1) as f32).cos();
        Complex { re: s * w, im: 0.0 }
    }).collect();
    fft.process(&mut buf);

    // Band power (only positive frequencies)
    let hpb = sr / FFT_SIZE as f32;
    let (mut b_p, mut b_n) = (0.0f32, 0u32);
    let (mut m_p, mut m_n) = (0.0f32, 0u32);
    let (mut t_p, mut t_n) = (0.0f32, 0u32);
    for i in 1..FFT_SIZE / 2 {
        let hz = i as f32 * hpb;
        let p  = buf[i].norm_sqr();
        if      hz < 250.0  { b_p += p; b_n += 1; }
        else if hz < 2500.0 { m_p += p; m_n += 1; }
        else                { t_p += p; t_n += 1; }
    }
    let nrm = |p: f32, n: u32| -> f32 {
        if n == 0 { 0.0 } else { ((p / n as f32).sqrt() * 5.0).clamp(0.0, 1.0) }
    };

    let new_amp    = (rms * 12.0).clamp(0.0, 1.0);
    let new_bass   = nrm(b_p, b_n);
    let new_mid    = nrm(m_p, m_n);
    let new_treble = nrm(t_p, t_n);

    let sm = |old: f32, new: f32| -> f32 {
        let a = if new > old { ATK } else { REL };
        old + a * (new - old)
    };

    let mut bnd = bands.lock().unwrap();
    bnd.amplitude = sm(bnd.amplitude, new_amp);
    bnd.bass      = sm(bnd.bass,      new_bass);
    bnd.mid       = sm(bnd.mid,       new_mid);
    bnd.treble    = sm(bnd.treble,    new_treble);
}
