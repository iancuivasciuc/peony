use std::error::Error;
use std::time::Duration;

use num_traits::Zero;

use realfft::{num_complex::ComplexFloat, FftNum, RealFftPlanner};
use rodio::{Sample as RodioSample, Source};
use rubato::Sample as RubatoSample;
use symphonia::core::conv::{ConvertibleSample as SymphoniaSample, FromSample};

pub mod load;
pub mod resample;

use load::SignalLoader;
use super::sample::{Sample, IntSample, FloatSample};
use resample::{ResampleType, Resampler};

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
    pub fn new(samples: Vec<Vec<S>>, sample_rate: u32) -> Result<Self, Box<dyn Error>> {
        if !samples.is_empty() {
            let len = samples[0].len();

            for channel in samples.iter().skip(1) {
                if len != channel.len() {
                    return Err("Inconsistent channel lengths".into());
                }
            }
        }

        Ok(Signal {
            samples,
            sample_rate,
        })
    }

    pub fn load(
        path: &str,
        offset: Duration,
        duration: Option<Duration>,
    ) -> Result<Self, Box<dyn Error>>
    where
        S: SymphoniaSample,
    {
        let loader = SignalLoader {
            path: path.to_string(),
            offset,
            duration,
            _marker: std::marker::PhantomData,
        };

        loader.load()
    }

    pub fn load_default(path: &str) -> Result<Self, Box<dyn Error>>
    where
        S: SymphoniaSample,
    {
        Self::load(path, Duration::ZERO, None)
    }

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
    pub fn bits_per_sample(&self) -> usize {
        std::mem::size_of::<Self>() * 8
    }

    #[inline(always)]
    pub fn data_type(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    #[inline(always)]
    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(self.len() as f64 / self.sample_rate as f64)
    }

    fn _to_mono(&mut self) {
        if self.channels() <= 1 {
            return;
        }

        // Constants
        let zero = S::MonoType::zero();

        for index in 0..self.len() {
            let mut sum = zero;

            for channel in &self.samples {
                sum = sum + <<S as Sample>::MonoType as Sample>::from_expect(channel[index]);
            }

            self.samples[0][index] = S::from_expect(
                sum / <<S as Sample>::MonoType as Sample>::from_expect(self.channels()),
            );
        }

        self.samples.truncate(1);
        self.samples.shrink_to_fit();
    }

    #[must_use]
    pub fn to_mono(&self) -> Signal<S> {
        let mut new_signal = self.clone();

        new_signal._to_mono();

        new_signal
    }

    pub fn to_mono_mut(&mut self) {
        self._to_mono()
    }

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
}

//////////////////////////////////////////////////  FloatSignal  //////////////////////////////////////////////////

impl<F> Signal<F>
where
    F: FloatSample,
{
    fn _resample(
        &mut self,
        new_sample_rate: u32,
        resample_type: ResampleType,
    ) -> Result<(), Box<dyn Error>>
    where
        F: RubatoSample,
    {
        let resampler = Resampler::new(resample_type);

        resampler.resample(self, new_sample_rate)?;

        Ok(())
    }

    pub fn resample(
        &self,
        new_sample_rate: u32,
        resample_type: ResampleType,
    ) -> Result<Signal<F>, Box<dyn Error>>
    where
        F: RubatoSample,
    {
        let mut new_signal = self.clone();

        new_signal._resample(new_sample_rate, resample_type)?;

        Ok(new_signal)
    }

    pub fn resample_mut(
        &mut self,
        new_sample_rate: u32,
        resample_type: ResampleType,
    ) -> Result<(), Box<dyn Error>>
    where
        F: RubatoSample,
    {
        self._resample(new_sample_rate, resample_type)
    }

    pub fn autocorrelate(&self, max_len: Option<usize>) -> Vec<Vec<F>>
    where
        F: FftNum,
    {
        let len = match max_len {
            Some(size) => std::cmp::min(size, self.len()),
            None => self.len(),
        };

        let padded_len = (self.len() * 2 - 1).next_power_of_two();

        let mut planner = RealFftPlanner::<F>::new();

        let rfft = planner.plan_fft_forward(padded_len);
        let irfft = planner.plan_fft_inverse(padded_len);

        let mut autocorr = vec![vec![F::zero(); padded_len]; self.channels()];

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
                .for_each(|sample| *sample = *sample / F::from_expect(padded_len));
        }

        autocorr
    }

    pub fn lpc(&self, order: usize) -> Vec<Vec<F>> {
        // Constants
        let zero = F::zero();
        let one = F::one();
        let two = F::from_expect(2);

        let n = self.len() - 1;

        // Coefficients
        let mut coeffs = vec![vec![zero; order + 1]; self.channels()];

        // Forward and backward errors
        let mut f = vec![zero; self.len()];
        let mut b = vec![zero; self.len()];

        // Mu
        let mut mu;

        // Denominator
        let mut dk = zero;

        for (channel_index, channel) in self.samples.iter().enumerate() {
            // Initialize a0
            coeffs[channel_index][0] = one;

            // Initialize f0 and b0 => eqn 5, 7
            for ((fx, bx), &x) in f.iter_mut().zip(b.iter_mut()).zip(channel.iter()) {
                *fx = x;
                *bx = x;
            }

            // Compute d0 => eqn 15
            for fx in &f {
                dk = dk + two * *fx * *fx;
            }
            dk = dk - f[0] * f[0] + b[n] * b[n];

            // Burg's recursion
            for k in 0..order {
                // Calculate mu
                mu = zero;
                for i in 0..n - k {
                    mu = mu - two * f[i + k + 1] * b[i];
                }
                mu = mu / dk;

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
                dk = (one - mu * mu) * dk - f[k + 1] * f[k + 1] - b[n - k - 1] * b[n - k - 1];
            }
        }

        coeffs
    }

    fn _zero_crossings(&self, threshold: Option<F>) -> Vec<Vec<bool>> {
        let mut crossings = vec![vec![false; self.len()]; self.channels()];
        let threshold = threshold.unwrap_or(F::zero());

        let mut prev = true;
        let mut curr;

        for (channel_index, channel) in self.samples.iter().enumerate() {
            for (index, sample) in channel.iter().enumerate() {
                curr = *sample >= F::zero();

                if !curr && (F::zero() - *sample <= threshold) {
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

    pub fn zero_crossings_with_threshold(&self, threshold: F) -> Vec<Vec<bool>> {
        self._zero_crossings(Some(threshold))
    }

    fn _mu_compress(&mut self, mu: usize) {
        let zero = F::zero();
        let one = F::one();

        let mu = F::from_expect(mu);
        let mu_log = (one + mu).ln();

        for channel in self.samples.iter_mut() {
            for sample in channel.iter_mut() {
                let sign = if *sample < zero { -one } else { one };

                *sample = sign * (one + mu * sample.abs()).ln() / mu_log;
            }
        }
    }

    #[must_use]
    pub fn mu_compress(&self, mu: usize) -> Signal<F> {
        let mut new_signal = self.clone();

        new_signal._mu_compress(mu);

        new_signal
    }

    pub fn mu_compress_mut(&mut self, mu: usize) {
        self._mu_compress(mu)
    }

    pub fn mu_compress_to_quantized<I>(&self, mu: usize) -> Signal<I>
    where
        I: IntSample,
    {
        // Constants
        let zero = F::zero();
        let one = F::one();
        let two = F::from_expect(2);

        let mut new_samples = vec![Vec::with_capacity(self.len()); self.channels()];

        let mu = std::cmp::min(mu, I::max_bins());
        let mu_f = F::from_expect(mu);
        let mu_log = (one + mu_f).ln();

        let linspace: Vec<F> = (0..=mu + 1)
            .map(|i| F::from_expect(i) * two / mu_f - one)
            .collect();

        for (channel_index, channel) in self.samples.iter().enumerate() {
            new_samples[channel_index] = channel
                .iter()
                .map(|sample| {
                    let sign = if *sample < zero { -one } else { one };

                    let compressed = sign * (one + mu_f * sample.abs()).ln() / mu_log;

                    let clamped = compressed
                        .max(F::from_expect(I::min_value()))
                        .min(F::from_expect(I::max_value()));

                    let bin = linspace
                        .binary_search_by(|&val| {
                            if clamped < val {
                                std::cmp::Ordering::Greater
                            } else {
                                std::cmp::Ordering::Less
                            }
                        })
                        .unwrap_or_else(|x| x);

                    if I::IS_SIGNED {
                        let bin = bin as i32 - ((mu + 1) / 2) as i32;

                        I::from_expect(bin)
                    } else {
                        I::from_expect(bin)
                    }
                })
                .collect();
        }

        Signal {
            samples: new_samples,
            sample_rate: self.sample_rate,
        }
    }

    fn _mu_expand(&mut self, mu: usize) {
        let zero = F::zero();
        let one = F::one();

        let mu = F::from_expect(mu);
        let mu_div = one / mu;
        let mu_add = one + mu;

        for channel in self.samples.iter_mut() {
            for sample in channel.iter_mut() {
                let sign = if *sample < zero { -one } else { one };

                *sample = sign * mu_div * (mu_add.powf(sample.abs()) - one);
            }
        }
    }

    pub fn mu_expand(&self, mu: usize) -> Signal<F> {
        let mut new_signal = self.clone();

        new_signal._mu_expand(mu);

        new_signal
    }

    pub fn mu_expand_mut(&mut self, mu: usize) {
        self._mu_expand(mu)
    }

    pub fn tone(frequency: F, sample_rate: u32, duration: Duration) -> Self {
        // Constants
        let two = F::from_expect(2);
        let pi = F::PI();

        let ts = F::one() / F::from_expect(sample_rate);

        let length = (duration.as_secs_f64() * sample_rate as f64) as u64;

        let samples = (0..length)
            .map(|n| (two * pi * frequency * F::from_expect(n) * ts).cos())
            .collect();

        Signal {
            samples: vec![samples],
            sample_rate,
        }
    }

    pub fn chirp(freq1: F, freq2: F, sample_rate: u32, duration: Duration, linear: bool) -> Self {
        // Constants
        let two = F::from_expect(2);
        let pi = F::PI();

        let ts = F::one() / F::from_expect(sample_rate);

        let length = (duration.as_secs_f64() * sample_rate as f64) as u64;

        let samples = if linear {
            (0..length)
                .map(|n| {
                    (two * pi
                        * (freq1
                            + (freq2 - freq1) * F::from_expect(n) * ts
                                / F::from_expect(duration.as_secs_f64()))
                        * F::from_expect(n)
                        * ts)
                        .cos()
                })
                .collect()
        } else {
            (0..length)
                .map(|n| {
                    (two * pi
                        * (freq1
                            * (freq2 / freq1).powf(
                                F::from_expect(n) * ts / F::from_expect(duration.as_secs_f64()),
                            )
                            * F::from_expect(n)
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

//////////////////////////////////////////////////  IntSignal  //////////////////////////////////////////////////

impl<I> Signal<I>
where
    I: IntSample,
{
    #[must_use]
    pub fn mu_expand_from_quantized<F>(&self, mu: usize) -> Signal<F>
    where
        F: FloatSample,
    {
        // Constants
        let zero = F::zero();
        let one = F::one();
        let two = F::from_expect(2);

        let mut new_samples = vec![Vec::new(); self.channels()];

        let mu = F::from_expect(mu);
        let mu_div = one / mu;
        let mu_add = one + mu;

        for (channel_index, channel) in self.samples.iter().enumerate() {
            new_samples[channel_index].reserve_exact(channel.len());

            new_samples[channel_index] = channel
                .iter()
                .map(|sample| {
                    let mut sample = F::from_expect(*sample) * two / mu_add;
                    
                    if !I::IS_SIGNED {
                        sample = sample - one;
                    }

                    let sign = if sample < zero { -one } else { one };

                    sign * mu_div * (mu_add.powf(sample.abs()) - one)
                })
                .collect();
        }

        Signal {
            samples: new_samples,
            sample_rate: self.sample_rate,
        }
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

//////////////////////////////////////////////////  Tests  //////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use rodio::Sink;

    use super::*;

    #[test]
    fn load_test() {
        let mut signal: Signal<f32> = Signal::load_default("Shape of You.wav").unwrap();

        signal.to_mono_mut();

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

        signal.to_mono_mut();

        signal
            ._resample(60000, ResampleType::SincVeryHighQuality)
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

        signal.to_mono_mut();

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

        signal.to_mono_mut();

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

        signal.to_mono_mut();

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
    fn mu_test() {
        let mut signal: Signal<f32> = Signal::load(
            "Shape of You.wav",
            Duration::ZERO,
            Some(Duration::from_secs(10)),
        )
        .unwrap();

        signal.to_mono_mut();

        print!("Original signal: ");
        for channel in signal.samples.iter() {
            for sample in channel.iter().take(10) {
                print!("{:.4} ", sample);
            }
            println!();
        }

        let signal: Signal<u8> = signal.mu_compress_to_quantized(255);

        print!("Compressed signal: ");
        for channel in signal.samples.iter() {
            for sample in channel.iter().take(10) {
                print!("{} ", sample);
            }
            println!();
        }

        let signal: Signal<f32> = signal.mu_expand_from_quantized(255);

        print!("Expended signal: ");
        for channel in signal.samples.iter() {
            for sample in channel.iter().take(10) {
                print!("{:.4} ", sample);
            }
            println!();
        }
    }
}
