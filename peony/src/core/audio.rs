use std::error::Error;
use std::time::Duration;

use realfft::{num_complex::ComplexFloat, FftNum, RealFftPlanner};
use rodio::{Sample as RodioSample, Source};
use rubato::Sample as RubatoSample;
use symphonia::core::conv::{ConvertibleSample as SymphoniaSample, FromSample};

pub mod load;
pub mod resample;
pub mod sample;
pub mod util;

use load::SignalLoader;
use resample::{ResampleType, Resampler};
use sample::Sample;

//////////////////////////////////////////////////  Signal  //////////////////////////////////////////////////

#[derive(Clone)]
pub struct Signal<S>
where
    S: Sample,
{
    pub samples: Vec<Vec<S>>,
    pub sample_rate: u32,
}

impl<S> Signal<S>
where
    S: Sample,
{
    pub fn new(samples: Vec<Vec<S>>, sample_rate: u32) -> Result<Signal<S>, Box<dyn Error>> {
        if samples.is_empty() {
            return Err("Samples is empty".into());
        }

        let len = samples[0].len();

        for channel in samples.iter().skip(1) {
            if channel.len() != len {
                return Err("Channels have different lengths".into());
            }
        }

        Ok(Signal {
            samples,
            sample_rate,
        })
    }

    // Information
    #[inline(always)]
    pub fn channels(&self) -> usize {
        self.samples.len()
    }

    #[inline(always)]
    pub fn has_channels(&self) -> bool {
        !self.samples.is_empty()
    }

    #[allow(clippy::len_without_is_empty)] 
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.samples[0].len()
    }

    #[inline(always)]
    pub fn has_samples(&self) -> bool {
        self.has_channels() && !self.samples[0].is_empty()
    }

    #[inline(always)]
    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(self.len() as f64 / self.sample_rate as f64)
    }

    #[inline(always)]
    pub fn bits_per_sample(&self) -> usize {
        std::mem::size_of::<S>() * 8
    }

    #[inline(always)]
    pub fn data_type(&self) -> &'static str {
        std::any::type_name::<S>()
    }

    #[must_use]
    pub fn rodio_source<R>(self) -> SignalRodioSource<S, R>
    where
        R: RodioSample,
    {
        SignalRodioSource {
            signal: self,
            index: 0,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn to_mono(&mut self) {
        if self.channels() <= 1 {
            return;
        }

        for index in 0..self.len() {
            let mut sum = S::zero();

            for channel in &self.samples {
                sum = sum + channel[index];
            }

            self.samples[0][index] = sum / S::from_usize(self.channels()).unwrap();
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

    #[must_use]
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
        let mut mu;

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
        let threshold = threshold.unwrap_or(S::zero());

        let mut prev = true;
        let mut curr;

        for (channel_index, channel) in self.samples.iter().enumerate() {
            for (index, sample) in channel.iter().enumerate() {
                curr = *sample >= S::zero();

                if !curr && (S::zero() - *sample <= threshold) {
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

    #[must_use]
    pub fn zero_crossings(&self) -> Vec<Vec<bool>> {
        self._zero_crossings(None)
    }

    #[must_use]
    pub fn zero_crossings_with_threshold(&self, threshold: S) -> Vec<Vec<bool>> {
        self._zero_crossings(Some(threshold))
    }

    #[must_use]
    pub fn mu_compress(&mut self, mu: u16) -> Signal<S> {
        let mut new_samples = vec![Vec::with_capacity(self.len()); self.channels()];

        let mu_s = S::from_u16(mu).unwrap();
        let mu_log = (S::one() + mu_s).ln();

        for (channel_index, channel) in self.samples.iter().enumerate() {
            new_samples[channel_index] = channel.iter().map(|sample| {
                let sign = if *sample < S::zero() { S::from(-1.0).unwrap() } else { S::from(1.0).unwrap() };

                sign * (S::from(1.0).unwrap() + mu_s * sample.abs()).ln() / mu_log
            }).collect();
        }

        Signal {
            samples: new_samples,
            sample_rate: self.sample_rate,
        }
    }

    #[must_use]
    pub fn mu_compress_to_u8(&self, mu: u8) -> Vec<Vec<u8>> {
        let mut new_samples = vec![Vec::with_capacity(self.len()); self.channels()];

        let mu_s = S::from_u8(mu).unwrap();
        let mu_log = (S::from(1.0).unwrap() + mu_s).ln();

        let linspace: Vec<S> = (0..=mu as usize + 1).map(|i| {
            S::from_usize(i).unwrap() * S::from_f64(2.0).unwrap() / mu_s - S::one()
        }).collect();

        for (channel_index, channel) in self.samples.iter().enumerate() {
            new_samples[channel_index] = channel.iter().map(|sample| {
                let sign = if *sample < S::zero() { S::from(-1.0).unwrap() } else { S::from(1.0).unwrap() };

                let compressed = sign * (S::from(1.0).unwrap() + mu_s * sample.abs()).ln() / mu_log;
                let clamped = compressed.max(S::from(i8::MIN).unwrap()).min(S::from(i8::MAX).unwrap());

                let bin = linspace.binary_search_by(|&val| {
                    if clamped < val { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Less }
                }).unwrap_or_else(|x| x);

                bin as u8
            }).collect();
        }

        new_samples
    }

    #[must_use]
    pub fn mu_compress_to_u16(&self, mu: u16) -> Vec<Vec<u16>> {
        let mut new_samples = vec![Vec::with_capacity(self.len()); self.channels()];

        let mu_s = S::from_u16(mu).unwrap();
        let mu_log = (S::from(1.0).unwrap() + mu_s).ln();

        let linspace: Vec<S> = (0..=mu as usize + 1).map(|i| {
            S::from_usize(i).unwrap() * S::from_f64(2.0).unwrap() / mu_s - S::one()
        }).collect();

        for (channel_index, channel) in self.samples.iter().enumerate() {
            new_samples[channel_index] = channel.iter().map(|sample| {
                let sign = if *sample < S::zero() { S::from(-1.0).unwrap() } else { S::from(1.0).unwrap() };

                let compressed = sign * (S::from(1.0).unwrap() + mu_s * sample.abs()).ln() / mu_log;
                let clamped = compressed.max(S::from(i8::MIN).unwrap()).min(S::from(i8::MAX).unwrap());

                let bin = linspace.binary_search_by(|&val| {
                    if clamped < val { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Less }
                }).unwrap_or_else(|x| x);

                bin as u16
            }).collect();
        }

        new_samples
    }

    #[must_use]
    pub fn mu_compress_to_i8(&self, mu: u8) -> Vec<Vec<i8>> {
        let mut new_samples = vec![Vec::with_capacity(self.len()); self.channels()];

        let mu_s = S::from_u8(mu).unwrap();
        let mu_log = (S::from(1.0).unwrap() + mu_s).ln();

        let linspace: Vec<S> = (0..=mu as usize + 1).map(|i| {
            S::from_usize(i).unwrap() * S::from_f64(2.0).unwrap() / mu_s - S::one()
        }).collect();

        for (channel_index, channel) in self.samples.iter().enumerate() {
            new_samples[channel_index] = channel.iter().map(|sample| {
                let sign = if *sample < S::zero() { S::from(-1.0).unwrap() } else { S::from(1.0).unwrap() };

                let compressed = sign * (S::from(1.0).unwrap() + mu_s * sample.abs()).ln() / mu_log;
                let clamped = compressed.max(S::from(i8::MIN).unwrap()).min(S::from(i8::MAX).unwrap());

                let bin = linspace.binary_search_by(|&val| {
                    if clamped < val { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Less }
                }).unwrap_or_else(|x| x);

                (bin as i32 - ((mu as usize + 1) / 2) as i32) as i8
            }).collect();
        }

        new_samples
    }

    #[must_use]
    pub fn mu_compress_to_i16(&self, mu: u16) -> Vec<Vec<i16>> {
        let mut new_samples = vec![Vec::with_capacity(self.len()); self.channels()];

        let mu_s = S::from_u16(mu).unwrap();
        let mu_log = (S::from(1.0).unwrap() + mu_s).ln();

        let linspace: Vec<S> = (0..=mu as usize + 1).map(|i| {
            S::from_usize(i).unwrap() * S::from_f64(2.0).unwrap() / mu_s - S::one()
        }).collect();

        for (channel_index, channel) in self.samples.iter().enumerate() {
            new_samples[channel_index] = channel.iter().map(|sample| {
                let sign = if *sample < S::zero() { S::from(-1.0).unwrap() } else { S::from(1.0).unwrap() };

                let compressed = sign * (S::from(1.0).unwrap() + mu_s * sample.abs()).ln() / mu_log;
                let clamped = compressed.max(S::from(i8::MIN).unwrap()).min(S::from(i8::MAX).unwrap());

                let bin = linspace.binary_search_by(|&val| {
                    if clamped < val { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Less }
                }).unwrap_or_else(|x| x);

                (bin as i32 - ((mu as usize + 1) / 2) as i32) as i16
            }).collect();
        }

        new_samples
    }

    #[must_use]
    pub fn mu_expand(&mut self, mu: u16) -> Signal<S> {
        let mut new_samples = vec![Vec::with_capacity(self.len()); self.channels()];

        let mu_s = S::from_u16(mu).unwrap();
        let mu_div = S::one() / mu_s;
        let mu_add = S::one() + mu_s;

        for (channel_index, channel) in self.samples.iter().enumerate() {
            new_samples[channel_index] = channel.iter().map(|sample| {
                let sign = if *sample < S::zero() { S::from(-1.0).unwrap() } else { S::from(1.0).unwrap() };

                sign * mu_div * (mu_add.powf(*sample) - S::one())
            }).collect();
        }

        Signal {
            samples: new_samples,
            sample_rate: self.sample_rate,
        }
    }

    #[must_use]
    pub fn mu_expand_from_u8(samples: Vec<Vec<u8>>, mu: u8) -> Vec<Vec<S>> {
        let channels = samples.len();

        let mut new_samples = vec![Vec::new(); channels];

        let mu_s = S::from_u8(mu).unwrap();
        let mu_div = S::one() / mu_s;
        let mu_add = S::one() + mu_s;

        for (channel_index, channel) in samples.iter().enumerate() {
            new_samples[channel_index].reserve_exact(channel.len());

            new_samples[channel_index] = channel.iter().map(|sample| {
                let sample = S::from_u8(*sample).unwrap() * S::from(2).unwrap() / S::from_usize(mu as usize + 1).unwrap() - S::one();

                let sign = if sample < S::zero() { S::from(-1.0).unwrap() } else { S::from(1.0).unwrap() };

                sign * mu_div * (mu_add.powf(sample) - S::one())
            }).collect();
        }

        new_samples
    }

    #[must_use]
    pub fn mu_expand_from_u16(samples: Vec<Vec<u16>>, mu: u16) -> Vec<Vec<S>> {
        let channels = samples.len();

        let mut new_samples = vec![Vec::new(); channels];

        let mu_s = S::from_u16(mu).unwrap();
        let mu_div = S::one() / mu_s;
        let mu_add = S::one() + mu_s;

        for (channel_index, channel) in samples.iter().enumerate() {
            new_samples[channel_index].reserve_exact(channel.len());

            new_samples[channel_index] = channel.iter().map(|sample| {
                let sample = S::from_u16(*sample).unwrap() * S::from(2).unwrap() / S::from_usize(mu as usize + 1).unwrap() - S::one();

                let sign = if sample < S::zero() { S::from(-1.0).unwrap() } else { S::from(1.0).unwrap() };

                sign * mu_div * (mu_add.powf(sample) - S::one())
            }).collect();
        }

        new_samples
    }

    #[must_use]
    pub fn mu_expand_from_i8(samples: Vec<Vec<i8>>, mu: u8) -> Vec<Vec<S>> {
        let channels = samples.len();

        let mut new_samples = vec![Vec::new(); channels];

        let mu_s = S::from_u8(mu).unwrap();
        let mu_div = S::one() / mu_s;
        let mu_add = S::one() + mu_s;

        for (channel_index, channel) in samples.iter().enumerate() {
            new_samples[channel_index].reserve_exact(channel.len());

            new_samples[channel_index] = channel.iter().map(|sample| {
                let sample = S::from_i8(*sample).unwrap() * S::from(2).unwrap() / S::from_usize(mu as usize + 1).unwrap();

                let sign = if sample < S::zero() { S::from(-1.0).unwrap() } else { S::from(1.0).unwrap() };

                sign * mu_div * (mu_add.powf(sample) - S::one())
            }).collect();
        }

        new_samples
    }

    #[must_use]
    pub fn mu_expand_from_i16(samples: Vec<Vec<i16>>, mu: u16) -> Vec<Vec<S>> {
        let channels = samples.len();

        let mut new_samples = vec![Vec::new(); channels];

        let mu_s = S::from_u16(mu).unwrap();
        let mu_div = S::one() / mu_s;
        let mu_add = S::one() + mu_s;

        for (channel_index, channel) in samples.iter().enumerate() {
            new_samples[channel_index].reserve_exact(channel.len());

            new_samples[channel_index] = channel.iter().map(|sample| {
                let sample = S::from_i16(*sample).unwrap() * S::from(2).unwrap() / S::from_usize(mu as usize + 1).unwrap();

                let sign = if sample < S::zero() { S::from(-1.0).unwrap() } else { S::from(1.0).unwrap() };

                sign * mu_div * (mu_add.powf(sample) - S::one())
            }).collect();
        }

        new_samples
    }

    pub fn tone(frequency: S, sample_rate: u32, duration: Duration) -> Signal<S> {
        let ts = S::one() / S::from_u32(sample_rate).unwrap();

        let length = (duration.as_secs_f64() * sample_rate as f64) as u64;

        let samples = (0..length)
            .map(|n| {
                (S::from(2.0).unwrap() * S::PI() * frequency * S::from_u64(n).unwrap() * ts).cos()
            })
            .collect();

        Signal {
            samples: vec![samples],
            sample_rate,
        }
    }

    pub fn chirp(
        freq1: S,
        freq2: S,
        sample_rate: u32,
        duration: Duration,
        linear: bool,
    ) -> Signal<S> {
        let ts = S::one() / S::from_u32(sample_rate).unwrap();

        let length = (duration.as_secs_f64() * sample_rate as f64) as u64;

        let samples = if linear {
            (0..length)
                .map(|n| {
                    (S::from(2.0).unwrap()
                        * S::PI()
                        * (freq1
                            + (freq2 - freq1) * S::from_u64(n).unwrap() * ts
                                / S::from_f64(duration.as_secs_f64()).unwrap())
                        * S::from_u64(n).unwrap()
                        * ts)
                        .cos()
                })
                .collect()
        } else {
            (0..length)
                .map(|n| {
                    (S::from(2.0).unwrap()
                        * S::PI()
                        * (freq1
                            * (freq2 / freq1).powf(
                                S::from_u64(n).unwrap() * ts
                                    / S::from_f64(duration.as_secs_f64()).unwrap(),
                            )
                            * S::from_u64(n).unwrap()
                            * ts))
                        .cos()
                })
                .collect()
        };

        Signal {
            samples: vec![samples],
            sample_rate,
        }
    }
}

impl<S> Signal<S>
where
    S: Sample + SymphoniaSample,
{
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
}

impl<S> Signal<S>
where
    S: Sample + RubatoSample,
{
    pub fn resample(
        &mut self,
        new_sample_rate: u32,
        resample_type: ResampleType,
    ) -> Result<(), Box<dyn Error>>
    where {
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
}

impl<S> Signal<S>
where
    S: Sample + FftNum,
{
    pub fn autocorrelate(&self, max_len: Option<usize>) -> Vec<Vec<S>> {
        let len = match max_len {
            Some(size) => std::cmp::min(size, self.len()),
            None => self.len(),
        };

        let padded_len = (self.len() * 2 - 1).next_power_of_two();

        let mut planner = RealFftPlanner::<S>::new();

        let rfft = planner.plan_fft_forward(padded_len);
        let irfft = planner.plan_fft_inverse(padded_len);

        let mut autocorr = vec![vec![S::zero(); padded_len]; self.channels()];

        let mut spectrum = rfft.make_output_vec();

        for (channel_index, channel) in self.samples.iter().enumerate() {
            autocorr[channel_index].copy_from_slice(channel);

            rfft.process(&mut autocorr[channel_index], &mut spectrum)
                .unwrap();

            spectrum
                .iter_mut()
                .for_each(|freq| *freq = *freq * freq.conj());

            irfft
                .process(&mut spectrum, &mut autocorr[channel_index])
                .unwrap();

            autocorr[channel_index].truncate(len);

            autocorr[channel_index]
                .iter_mut()
                .for_each(|sample| *sample = *sample / S::from_usize(padded_len).unwrap());
        }

        autocorr
    }
}

//////////////////////////////////////////////////  SignalRodioSource  //////////////////////////////////////////////////

pub struct SignalRodioSource<S, R>
where
    S: Sample,
    R: RodioSample,
{
    pub signal: Signal<S>,
    pub index: usize,
    _marker: std::marker::PhantomData<R>,
}

impl<S, R> SignalRodioSource<S, R>
where
    S: Sample,
    R: RodioSample,
{
    #[inline(always)]
    pub fn inner(self) -> Signal<S> {
        self.signal
    }
}

impl<S, R> Iterator for SignalRodioSource<S, R>
where
    S: Sample + SymphoniaSample,
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
    S: Sample + SymphoniaSample,
    R: RodioSample + FromSample<S>,
{
}

impl<S, R> Source for SignalRodioSource<S, R>
where
    S: Sample + SymphoniaSample,
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
        let mut signal: Signal<f32> = Signal::load(
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

        print!("Original signal: ");
        for channel in signal.samples.iter() {
            for sample in channel.iter().take(10) {
                print!("{:.4} ", sample);
            }
            println!();
        }

        let compressed = signal.mu_compress(255);

        print!("Compressed signal: ");
        for channel in compressed.samples.iter() {
            for sample in channel.iter().take(10) {
                print!("{:.4} ", sample);
            }
            println!();
        }

        let quantized = signal.mu_compress_to_i16(1023);

        print!("Quantized signal: ");
        for channel in quantized.iter() {
            for sample in channel.iter().take(10) {
                print!("{:.4} ", sample);
            }
            println!();
        }
    }
}
