use realfft::num_complex::Complex;
use realfft::FftNum;

use super::sample::FloatSample;
use super::signal::Signal;

pub(crate) mod stft;
pub(crate) mod window;

use stft::Stft;

pub use window::*;

//////////////////////////////////////////////////  Spectrum  //////////////////////////////////////////////////

pub struct Spectrum<F>
where
    F: FloatSample + FftNum,
{
    pub freqs: Vec<Vec<Vec<Complex<F>>>>,
    pub sample_rate: u32,
}

impl<F> Spectrum<F>
where
    F: FloatSample + FftNum,
{
    pub fn new(freqs: Vec<Vec<Vec<Complex<F>>>>, sample_rate: u32) -> Self {
        Spectrum { freqs, sample_rate }
    }

    #[inline(always)]
    pub fn channels(&self) -> usize {
        self.freqs.len()
    }

    #[allow(clippy::len_without_is_empty)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.freqs[0].len()
    }

    #[inline(always)]
    pub fn frames(&self) -> usize {
        self.freqs[0][0].len()
    }

    pub fn istft(
        &self,
        frame_len: usize,
        hop_len: Option<usize>,
        window_len: Option<usize>,
        window_type: WindowType,
        center: bool,
    ) -> Signal<F>
    where
        F: FftNum,
    {
        let stft = Stft::new(frame_len, hop_len, window_len, window_type, center);

        stft.istft(self)
    }
}
