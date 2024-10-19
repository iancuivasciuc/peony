use num_traits::Zero;
use realfft::{num_complex::Complex, FftNum, RealFftPlanner};

use crate::core::{sample::FloatSample, signal::Signal};

use super::window::{Window, WindowType};
use super::Spectrum;

//////////////////////////////////////////////////  Stft  //////////////////////////////////////////////////

pub struct Stft<F>
where
    F: FloatSample + FftNum,
{
    pub frame_len: usize,
    pub hop_len: usize,
    pub window_len: usize,
    pub window_type: WindowType,
    pub center: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F> Stft<F>
where
    F: FloatSample + FftNum,
{
    pub fn new(
        frame_len: usize,
        hop_len: Option<usize>,
        window_len: Option<usize>,
        window_type: WindowType,
        center: bool,
    ) -> Self {
        Stft {
            frame_len,
            hop_len: hop_len.unwrap_or(frame_len / 4),
            window_len: window_len.unwrap_or(frame_len),
            window_type,
            center,
            _marker: std::marker::PhantomData,
        }
    }

    fn _window(&self) -> Vec<F> {
        let mut window = Vec::with_capacity(self.frame_len);
        window.resize((self.frame_len - self.window_len) / 2, F::zero());
        window.extend(Window::<F>::new(self.window_len, self.window_type).window_iter());
        window.resize(self.frame_len, F::zero());

        window
    }

    pub fn stft(&self, signal: &Signal<F>) -> Spectrum<F> {
        // Constants
        let zero = F::zero();

        let padding = if self.center { self.frame_len / 2 } else { 0 };
        let padded_len = signal.len() + 2 * padding;

        let mut freqs =
            vec![
                vec![
                    vec![Complex::<F>::zero(); (padded_len - self.frame_len) / self.hop_len + 1];
                    self.frame_len / 2 + 1
                ];
                signal.channels()
            ];

        let window = self._window();

        let mut planner = RealFftPlanner::<F>::new();
        let rfft = planner.plan_fft_forward(self.frame_len);

        let mut input = rfft.make_input_vec();
        let mut output = rfft.make_output_vec();

        for (channel_index, channel) in signal.samples.iter().enumerate() {
            for (frame_index, sample_index) in (0..=padded_len - self.frame_len)
                .step_by(self.hop_len)
                .enumerate()
            {
                for buffer_index in 0..self.frame_len {
                    let index = sample_index + buffer_index;

                    input[buffer_index] = if index < padding || index >= signal.len() + padding {
                        zero
                    } else {
                        channel[index - padding]
                    } * window[buffer_index]
                }

                rfft.process(&mut input, &mut output).unwrap();

                for (freq_index, freq) in output.iter().enumerate() {
                    freqs[channel_index][freq_index][frame_index] = *freq;
                }
            }
        }

        Spectrum {
            freqs,
            sample_rate: signal.sample_rate,
        }
    }

    pub fn istft(&self, spectrum: &Spectrum<F>) -> Signal<F> {
        // Constants
        let zero = F::zero();

        let channels = spectrum.channels();
        let frames = spectrum.frames();
        let padded_len = (frames - 1) * self.hop_len + self.frame_len;
        let padding = if self.center { self.frame_len / 2 } else { 0 };
        let samples_len = padded_len - 2 * padding;

        let mut samples = vec![vec![zero; samples_len]; channels];
        let mut normalization = vec![zero; samples_len];

        let window = self._window();

        let mut planner = RealFftPlanner::<F>::new();
        let irfft = planner.plan_fft_inverse(self.frame_len);

        let mut input = irfft.make_input_vec();
        let mut output = irfft.make_output_vec();

        for (channel_index, channel) in spectrum.freqs.iter().enumerate() {
            for frame_index in 0..frames {
                let start_sample = frame_index * self.hop_len;

                for (freq_index, freq) in input.iter_mut().enumerate() {
                    *freq = channel[freq_index][frame_index];
                }

                irfft.process(&mut input, &mut output).unwrap();

                for (index, sample) in output.iter().enumerate() {
                    if start_sample + index >= padding
                        && start_sample + index < samples_len + padding
                    {
                        samples[channel_index][start_sample + index - padding] = samples
                            [channel_index][start_sample + index - padding]
                            + *sample * window[index] / F::from_expect(self.frame_len);
                        normalization[start_sample + index - padding] = normalization
                            [start_sample + index - padding]
                            + window[index] * window[index];
                    }
                }
            }
        }

        // Normalization
        for channel in samples.iter_mut() {
            for (index, sample) in channel.iter_mut().enumerate() {
                if normalization[index] != zero {
                    *sample = *sample / normalization[index];
                }
            }
        }

        Signal {
            samples,
            sample_rate: spectrum.sample_rate,
        }
    }
}
