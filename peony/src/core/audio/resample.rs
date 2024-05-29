use std::error::Error;
use symphonia::core::conv::{ConvertibleSample as SymphoniaSample, FromSample};

use rubato::{
    FastFixedIn, FftFixedInOut, PolynomialDegree, SincFixedIn, SincFixedOut,
    SincInterpolationParameters, SincInterpolationType, VecResampler as RubatoResampler,
    WindowFunction,
};

use super::util::Util;
use super::Signal;

pub enum ResampleType {
    Fft,
    SincVeryHighQuality,
    SincHighQuality,
    SincMediumQuality,
    SincLowQuality,
    Fastest,
}

pub struct Resampler {
    pub resample_type: ResampleType,
}

impl Resampler {
    pub fn new(resample_type: ResampleType) -> Resampler {
        Resampler { resample_type }
    }

    pub fn resample<S>(
        &self,
        signal: &mut Signal<S>,
        new_sample_rate: u32,
    ) -> Result<(), Box<dyn Error>>
    where
        S: SymphoniaSample,
        f64: symphonia::core::conv::FromSample<S>,
    {
        if signal.sample_rate == new_sample_rate {
            return Ok(());
        }

        let n_frames = signal.len() / signal.channels as usize;

        let samples_f64: Vec<f64> = signal
            .samples
            .iter()
            .map(|sample| f64::from_sample(*sample))
            .collect();

        let mut new_deinterleaved = vec![
            Vec::with_capacity(
                n_frames * new_sample_rate as usize / signal.sample_rate as usize
            );
            signal.channels as usize
        ];

        let mut resampler: Box<dyn RubatoResampler<f64>> = match self.resample_type {
            ResampleType::Fft => {
                Resampler::_resampler_fft(signal, new_sample_rate, signal.channels)?
            }
            ResampleType::SincVeryHighQuality => {
                Resampler::_resampler_sinc_vhq(signal, new_sample_rate, signal.channels)?
            }
            ResampleType::SincHighQuality => {
                Resampler::_resampler_sinc_hq(signal, new_sample_rate, signal.channels)?
            }
            ResampleType::SincMediumQuality => {
                Resampler::_resampler_sinc_mq(signal, new_sample_rate, signal.channels)?
            }
            ResampleType::SincLowQuality => {
                Resampler::_resampler_sinc_lq(signal, new_sample_rate, signal.channels)?
            }
            ResampleType::Fastest => {
                Resampler::_resampler_fastest(signal, new_sample_rate, signal.channels)?
            }
        };

        let input_size = resampler.input_frames_max();
        let output_size = input_size * new_sample_rate as usize / signal.sample_rate as usize;

        let mut output_buffer = resampler.output_buffer_allocate(true);

        let input_iter = samples_f64.chunks_exact(input_size * signal.channels as usize);

        let mut last_input_buffer = Util::deinterleave(input_iter.remainder(), signal.channels)?;

        for input in input_iter {
            let input_buffer = Util::deinterleave(input, signal.channels)?;

            resampler.process_into_buffer(&input_buffer, &mut output_buffer, None)?;

            for (channel_index, channel) in output_buffer.iter().enumerate() {
                new_deinterleaved[channel_index].extend(
                    channel[..output_size]
                        .iter()
                        .map(|sample| S::from_sample(*sample)),
                );
            }
        }

        let last_input_size = last_input_buffer[0].len();
        let last_output_size =
            last_input_size * new_sample_rate as usize / signal.sample_rate as usize;

        for channel in last_input_buffer.iter_mut() {
            channel.extend_from_slice(&vec![0.0; input_size - last_input_size]);
        }

        resampler.process_into_buffer(&last_input_buffer, &mut output_buffer, None)?;

        for (channel_index, channel) in output_buffer.iter().enumerate() {
            new_deinterleaved[channel_index].extend(
                channel[..last_output_size]
                    .iter()
                    .map(|sample| S::from_sample(*sample)),
            );
        }

        let new_samples = Util::into_interleave(new_deinterleaved)?;

        signal.samples = new_samples;
        signal.sample_rate = new_sample_rate;

        Ok(())
    }

    fn _resampler_fft<S>(
        signal: &mut Signal<S>,
        new_sample_rate: u32,
        channels: u16,
    ) -> Result<Box<dyn RubatoResampler<f64>>, Box<dyn Error>>
    where
        S: SymphoniaSample,
        f64: symphonia::core::conv::FromSample<S>,
    {
        let resampler = Box::new(FftFixedInOut::<f64>::new(
            signal.sample_rate as usize,
            new_sample_rate as usize,
            2048,
            channels as usize,
        )?);

        Ok(resampler)
    }

    fn _resampler_sinc_vhq<S>(
        signal: &mut Signal<S>,
        new_sample_rate: u32,
        channels: u16,
    ) -> Result<Box<dyn RubatoResampler<f64>>, Box<dyn Error>>
    where
        S: SymphoniaSample,
        f64: symphonia::core::conv::FromSample<S>,
    {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = Box::new(SincFixedIn::<f64>::new(
            new_sample_rate as f64 / signal.sample_rate as f64,
            1.0,
            params,
            2048,
            channels as usize,
        )?);

        Ok(resampler)
    }

    fn _resampler_sinc_hq<S>(
        signal: &mut Signal<S>,
        new_sample_rate: u32,
        channels: u16,
    ) -> Result<Box<dyn RubatoResampler<f64>>, Box<dyn Error>>
    where
        S: SymphoniaSample,
        f64: symphonia::core::conv::FromSample<S>,
    {
        let params = SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 64,
            window: WindowFunction::Blackman2,
        };
        let resampler = Box::new(SincFixedIn::<f64>::new(
            new_sample_rate as f64 / signal.sample_rate as f64,
            1.0,
            params,
            1024,
            channels as usize,
        )?);

        Ok(resampler)
    }

    fn _resampler_sinc_mq<S>(
        signal: &mut Signal<S>,
        new_sample_rate: u32,
        channels: u16,
    ) -> Result<Box<dyn RubatoResampler<f64>>, Box<dyn Error>>
    where
        S: SymphoniaSample,
        f64: symphonia::core::conv::FromSample<S>,
    {
        let params = SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.90,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 32,
            window: WindowFunction::Hann2,
        };
        let resampler = Box::new(SincFixedIn::<f64>::new(
            new_sample_rate as f64 / signal.sample_rate as f64,
            1.0,
            params,
            1024,
            channels as usize,
        )?);

        Ok(resampler)
    }

    fn _resampler_sinc_lq<S>(
        signal: &mut Signal<S>,
        new_sample_rate: u32,
        channels: u16,
    ) -> Result<Box<dyn RubatoResampler<f64>>, Box<dyn Error>>
    where
        S: SymphoniaSample,
        f64: symphonia::core::conv::FromSample<S>,
    {
        let params = SincInterpolationParameters {
            sinc_len: 32,
            f_cutoff: 0.85,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 16,
            window: WindowFunction::Hann2,
        };
        let resampler = Box::new(SincFixedIn::<f64>::new(
            new_sample_rate as f64 / signal.sample_rate as f64,
            1.0,
            params,
            512,
            channels as usize,
        )?);

        Ok(resampler)
    }

    fn _resampler_fastest<S>(
        signal: &mut Signal<S>,
        new_sample_rate: u32,
        channels: u16,
    ) -> Result<Box<dyn RubatoResampler<f64>>, Box<dyn Error>>
    where
        S: SymphoniaSample,
        f64: symphonia::core::conv::FromSample<S>,
    {
        let resampler = Box::new(FastFixedIn::<f64>::new(
            new_sample_rate as f64 / signal.sample_rate as f64,
            1.0,
            PolynomialDegree::Linear,
            512,
            channels as usize,
        )?);

        Ok(resampler)
    }
}
