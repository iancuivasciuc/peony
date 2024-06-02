use std::error::Error;
use std::time::Duration;

use ::realfft::num_complex::Complex;
use realfft::{num_complex::ComplexFloat, RealFftPlanner};
use symphonia::core::conv::{ConvertibleSample as SymphoniaSample, FromSample};

use rodio::{Sample as RodioSample, Source};

mod load;
mod resample;
mod samples;
mod util;

use load::SignalLoader;
use resample::{ResampleType, Resampler};

//////////////////////////////////////////////////  Signal  //////////////////////////////////////////////////

#[derive(Clone)]
pub struct Signal<S>
where
    S: SymphoniaSample,
{
    pub samples: Vec<Vec<S>>,
    pub sample_rate: u32,
}

impl<S> Signal<S>
where
    S: SymphoniaSample,
{
    //  Utility functions
    pub fn len(&self) -> usize {
        self.samples[0].len()
    }

    pub fn channels(&self) -> usize {
        self.samples.len()
    }

    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(self.len() as f64 / self.sample_rate as f64)
    }

    pub fn bits_per_sample(&self) -> usize {
        std::mem::size_of::<S>() * 8
    }

    pub fn data_type(&self) -> &'static str {
        std::any::type_name::<S>()
    }

    //  True functions
    pub fn load(
        path: &str,
        offset: Duration,
        duration: Option<Duration>,
    ) -> Result<Signal<S>, Box<dyn Error>> {
        let loader = SignalLoader {
            path: path.to_string(),
            offset,
            duration,
            _marker: std::marker::PhantomData,
        };

        loader.load()
    }

    pub fn load_default(path: &str) -> Result<Signal<S>, Box<dyn Error>> {
        Self::load(path, Duration::ZERO, None)
    }

    pub fn rodio_source<R>(self) -> SignalRodioSource<S, R>
    where
        R: RodioSample + FromSample<S>,
    {
        SignalRodioSource {
            signal: self,
            index: 0,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn to_mono(&mut self)
    where
        f32: symphonia::core::conv::FromSample<S>,
    {
        if self.channels() == 1 {
            return;
        }

        for index in 0..self.len() {
            let sum: f32 = self
                .samples
                .iter()
                .map(|channel| f32::from_sample(channel[index]))
                .sum();

            self.samples[0][index] = S::from_sample(sum / self.channels() as f32);
        }

        self.samples.truncate(1);
        self.samples.shrink_to_fit();
    }

    // pub fn to_mono(&self) -> Signal<S>
    // where
    //     f32: symphonia::core::conv::FromSample<S>,
    // {
    //     let mut new_signal = self.clone();
    //     new_signal._to_mono();
    //     new_signal
    // }

    // pub fn to_mono_in_place(&mut self)
    // where
    //     f32: symphonia::core::conv::FromSample<S>
    // {
    //     self._to_mono();
    // }

    pub fn resample(
        &mut self,
        new_sample_rate: u32,
        resample_type: ResampleType,
    ) -> Result<(), Box<dyn Error>>
    where
        f64: symphonia::core::conv::FromSample<S>,
    {
        let resampler = Resampler::new(resample_type);

        resampler.resample(self, new_sample_rate)?;

        Ok(())
    }

    // pub fn resample(
    //     &self,
    //     new_sample_rate: u32,
    //     resample_type: ResampleType,
    // ) -> Result<Signal<S>, Box<dyn Error>>
    // where
    //     f64: symphonia::core::conv::FromSample<S>,
    // {
    //     let mut new_signal = self.clone();
    //     new_signal._resample(new_sample_rate, resample_type)?;
    //     Ok(new_signal)
    // }

    // pub fn resample_in_place(
    //     &mut self,
    //     new_sample_rate: u32,
    //     resample_type: ResampleType,
    // ) -> Result<(), Box<dyn Error>>
    // where
    //     f64: symphonia::core::conv::FromSample<S>,
    // {
    //     self._resample(new_sample_rate, resample_type)
    // }

    pub fn autocorrelate(&self, max_len: Option<usize>) -> Vec<Vec<f32>>
    where
        f32: symphonia::core::conv::FromSample<S>,
    {
        let len = match max_len {
            Some(size) => std::cmp::min(size, self.len()),
            None => self.len(),
        };

        let padded_len = (self.len() * 2 - 1).next_power_of_two();

        let mut planner = RealFftPlanner::<f32>::new();

        let rfft = planner.plan_fft_forward(padded_len);
        let irfft = planner.plan_fft_inverse(padded_len);

        let mut autocorr = vec![vec![0.0; padded_len]; self.channels()];

        let mut spectrum = rfft.make_output_vec();

        for (channel_index, channel) in self.samples.iter().enumerate() {
            autocorr[channel_index]
                .iter_mut()
                .zip(channel.iter())
                .for_each(|(autocorr_sample, sample)| *autocorr_sample = f32::from_sample(*sample));

            rfft.process(&mut autocorr[channel_index], &mut spectrum)
                .unwrap();

            spectrum.iter_mut().for_each(|freq| *freq *= freq.conj());

            irfft
                .process(&mut spectrum, &mut autocorr[channel_index])
                .unwrap();

            autocorr[channel_index].truncate(len);
            autocorr[channel_index]
                .iter_mut()
                .for_each(|sample| *sample /= padded_len as f32);
        }

        autocorr
    }

    pub fn lpc(&self, order: usize) -> Vec<Vec<f64>>
    where
        f64: symphonia::core::conv::FromSample<S>,
    {
        let n = self.len() - 1;

        // Coefficients
        let mut coeffs = vec![vec![0.0; order + 1]; self.channels()];

        // Forward and backward errors
        let mut f = vec![0.0; self.len()];
        let mut b = vec![0.0; self.len()];

        // Mu
        let mut mu = 0.0;

        // Denominator
        let mut dk = 0.0;

        for (channel_index, channel) in self.samples.iter().enumerate() {
            // Initialize a0
            coeffs[channel_index][0] = 1.0;

            // Initialize f0 and b0 => eqn 5, 7
            for ((fx, bx), &x) in f.iter_mut().zip(b.iter_mut()).zip(channel.iter()) {
                *fx = f64::from_sample(x);
                *bx = f64::from_sample(x);
            }

            // Compute d0 => eqn 15
            for fx in &f {
                dk += 2.0 * fx * fx;
            }
            dk -= f[0] * f[0] + b[n] * b[n];

            // Burg's recursion
            for k in 0..order {
                // Calculate mu
                mu = 0.0;
                for i in 0..n - k {
                    mu -= 2.0 * f[i + k + 1] * b[i];
                }
                mu /= dk;

                // Update ak => eqn 8
                for i in 0..=(k + 1) / 2 {
                    let t1 = coeffs[channel_index][i] + mu * coeffs[channel_index][k + 1 - i];
                    let t2 = coeffs[channel_index][k + 1 - i] + mu * coeffs[channel_index][i];

                    coeffs[channel_index][i] = t1;
                    coeffs[channel_index][k + 1 - i] = t2;
                }

                // Update f and b => eqn 10, 11
                for i in 0..n - k {
                    let t1 = f[i + k + 1] + mu * b[i];
                    let t2 = b[i] + mu * f[i + k + 1];
                    f[i + k + 1] = t1;
                    b[i] = t2;
                }

                // Update dk
                dk = (1.0 - mu * mu) * dk - f[k + 1] * f[k + 1] - b[n - k - 1] * b[n - k - 1];
            }
        }

        coeffs
    }

    pub fn _zero_crossings(&self, threshold: Option<S>) -> Vec<Vec<bool>> {
        let mut crossings = vec![vec![false; self.len()]; self.channels()];
        let threshold = threshold.unwrap_or_default();

        let mut prev = true;
        let mut curr;

        for (channel_index, channel) in self.samples.iter().enumerate() {
            for (index, sample) in channel.iter().enumerate() {
                curr = *sample >= S::MID;

                if !curr && (S::MID - *sample <= threshold) {
                    curr = true;
                }

                if prev != curr {
                    crossings[channel_index][index] = true;
                }

                prev = curr;
            }
        }

        crossings
    }

    pub fn zero_crossings(&self) -> Vec<Vec<bool>> {
        self._zero_crossings(None)
    }

    pub fn zero_crossings_with_threshold(&self, threshold: S) -> Vec<Vec<bool>> {
        self._zero_crossings(Some(threshold))
    }

    pub fn mu_compress(&mut self, mu: u16)
    where
        f32: symphonia::core::conv::FromSample<S>,
    {
        let mu = mu as f32;
        let mu_log = (1.0 + mu).ln();

        self.samples.iter_mut().for_each(|channel| {
            channel.iter_mut().for_each(|sample| {
                let sign = if *sample < S::MID { -1.0} else { 1.0 };
                let abs = f32::from_sample(*sample).abs();
                *sample = S::from_sample(sign * (1.0 + mu * abs).ln() / mu_log);
            });
        });
    }

    pub fn mu_compress_quantized(&self, mu: u16) -> Vec<Vec<i8>>
    where f32: symphonia::core::conv::FromSample<S>,
    {
        let mut new_samples = vec![Vec::with_capacity(self.len()); self.channels()];

        new_samples
    }



    pub fn tone(frequency: f32, sample_rate: u32, duration: Duration) -> Signal<S> {
        let ts = 1.0 / sample_rate as f32;

        let length = (duration.as_secs_f32() * sample_rate as f32) as u64;

        let samples = (0..length)
            .map(|n| S::from_sample((2.0 * std::f32::consts::PI * frequency * n as f32 * ts).cos()))
            .collect();

        Signal {
            samples: vec![samples],
            sample_rate,
        }
    }

    pub fn chirp(
        freq1: f32,
        freq2: f32,
        sample_rate: u32,
        duration: Duration,
        linear: bool,
    ) -> Signal<S> {
        let ts = 1.0 / sample_rate as f32;

        let length = (duration.as_secs_f32() * sample_rate as f32) as u64;

        let samples = if linear {
            (0..length)
                .map(|n| {
                    S::from_sample(
                        (2.0 * std::f32::consts::PI
                            * (freq1 + (freq2 - freq1) * n as f32 * ts / duration.as_secs_f32())
                            * n as f32
                            * ts)
                            .cos(),
                    )
                })
                .collect()
        } else {
            (0..length)
                .map(|n| {
                    S::from_sample(
                        (2.0 * std::f32::consts::PI
                            * (freq1
                                * (freq2 / freq1).powf(n as f32 * ts / duration.as_secs_f32())
                                * n as f32
                                * ts))
                            .cos(),
                    )
                })
                .collect()
        };

        Signal {
            samples: vec![samples],
            sample_rate,
        }
    }
}

//////////////////////////////////////////////////  SignalRodioSource  //////////////////////////////////////////////////

pub struct SignalRodioSource<S, R>
where
    S: SymphoniaSample,
    R: RodioSample + FromSample<S>,
{
    pub signal: Signal<S>,
    pub index: usize,
    pub _marker: std::marker::PhantomData<R>,
}

impl<S, R> SignalRodioSource<S, R>
where
    S: SymphoniaSample,
    R: RodioSample + FromSample<S>,
{
    pub fn inner(self) -> Signal<S> {
        self.signal
    }
}

impl<S, R> Iterator for SignalRodioSource<S, R>
where
    S: SymphoniaSample,
    R: RodioSample + FromSample<S>,
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        let channels = self.signal.channels();

        if self.index < self.signal.len() * channels {
            self.index += 1;

            Some(<R as FromSample<S>>::from_sample(
                self.signal.samples[(self.index - 1) % channels][(self.index - 1) / channels],
            ))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.index - self.signal.len(),
            Some(self.index - self.signal.len()),
        )
    }
}

impl<S, R> ExactSizeIterator for SignalRodioSource<S, R>
where
    S: SymphoniaSample,
    R: RodioSample + FromSample<S>,
{
}

impl<S, R> Source for SignalRodioSource<S, R>
where
    S: SymphoniaSample,
    R: RodioSample + FromSample<S>,
{
    fn current_frame_len(&self) -> Option<usize> {
        Some(self.signal.len() * self.signal.channels() - self.index)
    }

    fn channels(&self) -> u16 {
        self.signal.channels() as u16
    }

    fn sample_rate(&self) -> u32 {
        self.signal.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        Some(self.signal.duration())
    }
}

//////////////////////////////////////////////////  SignalRodioSource  //////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use rodio::Sink;

    use super::*;

    #[test]
    fn load_test() {
        let mut signal: Signal<f32> = Signal::load_default("Shape of You.wav").unwrap();

        signal.to_mono();

        println!(
            "Length: {}, Duration: {:?}",
            signal.len(),
            signal.duration()
        );

        let source = signal.rodio_source::<f32>();

        let (_stream, stream_handle) = rodio::OutputStream::try_default().unwrap();
        let sink = Sink::try_new(&stream_handle).unwrap();
        sink.append(source);
        sink.sleep_until_end()
    }

    #[test]
    fn resample_test() {
        let mut signal: Signal<f32> = Signal::load(
            "Shape of You.wav",
            Duration::ZERO,
            Some(Duration::from_secs(10)),
        )
        .unwrap();

        signal.to_mono();

        signal
            .resample(60000, ResampleType::SincVeryHighQuality)
            .unwrap();

        println!(
            "Length: {}, Duration: {:?}",
            signal.len(),
            signal.duration()
        );

        let source = signal.rodio_source::<f32>();

        let (_stream, stream_handle) = rodio::OutputStream::try_default().unwrap();
        let sink = Sink::try_new(&stream_handle).unwrap();
        sink.append(source);
        sink.sleep_until_end()
    }

    #[test]
    fn autocorrelate_test() {
        let mut signal: Signal<f32> = Signal::load(
            "Shape of You.wav",
            Duration::ZERO,
            Some(Duration::from_secs(10)),
        )
        .unwrap();

        signal.to_mono();

        for channel in &mut signal.samples {
            channel.truncate(10);
        }

        let autocorr = signal.autocorrelate(None);

        for channel in autocorr {
            for sample in channel {
                print!("{} ", sample);
            }
            println!();
        }
    }

    #[test]
    fn zc_test() {
        let mut signal: Signal<i32> = Signal::load(
            "Shape of You.wav",
            Duration::ZERO,
            Some(Duration::from_secs(10)),
        )
        .unwrap();

        signal.to_mono();

        let crossings = signal.zero_crossings();

        for channel in crossings {
            println!("{}", channel.iter().filter(|&&x| x).count())
        }
    }

    #[test]
    fn lpc_test() {
        let mut signal: Signal<f32> = Signal::load(
            "Shape of You.wav",
            Duration::ZERO,
            Some(Duration::from_secs(10)),
        )
        .unwrap();

        signal.to_mono();

        println!("{} ", signal.len());

        let coeffs = signal.lpc(16);

        for channel in coeffs {
            for coef in channel {
                print!("{} ", coef);
            }
            println!();
        }
    }

    #[test]
    fn compress_test() {
        let mut signal: Signal<f32> = Signal::load(
            "Shape of You.wav",
            Duration::ZERO,
            Some(Duration::from_secs(10)),
        )
        .unwrap();

        signal.to_mono();

        for channel in signal.samples.iter() {
            for sample in channel.iter().take(10) {
                print!("{:.4} ", sample);
            }
            println!();
        }

        signal.mu_compress(255);

        for channel in signal.samples {
            for sample in channel.iter().take(10) {
                print!("{:.4} ", sample);
            }
            println!();
        }
    }
}
