use std::error::Error;
use symphonia::core::conv::{ConvertibleSample as SymphoniaSample, FromSample};

use rubato::{
    FastFixedIn, FftFixedInOut, PolynomialDegree, SincFixedIn, SincInterpolationParameters,
    SincInterpolationType, VecResampler as RubatoResampler, WindowFunction,
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

        let n_channels = signal.channels as usize;
        let n_frames = signal.len() / n_channels;

        let samples_f64 = signal
            .samples
            .iter()
            .map(|sample| f64::from_sample(*sample))
            .collect();

        let deinterleaved = Util::into_deinterleave(samples_f64, n_channels as u16)?;

        let mut resampler: Box<dyn RubatoResampler<f64>> = match self.resample_type {
            ResampleType::Fft => {
                Resampler::_resampler_fft(signal, new_sample_rate, n_channels as u16)?
            }
            ResampleType::SincVeryHighQuality => {
                Resampler::_resampler_sinc_vhq(signal, new_sample_rate, n_channels as u16)?
            }
            ResampleType::SincHighQuality => {
                Resampler::_resampler_sinc_hq(signal, new_sample_rate, n_channels as u16)?
            }
            ResampleType::SincMediumQuality => {
                Resampler::_resampler_sinc_mq(signal, new_sample_rate, n_channels as u16)?
            }
            ResampleType::SincLowQuality => {
                Resampler::_resampler_sinc_lq(signal, new_sample_rate, n_channels as u16)?
            }
            ResampleType::Fastest => {
                Resampler::_resampler_fastest(signal, new_sample_rate, n_channels as u16)?
            }
        };

        let mut input_buffer = resampler.input_buffer_allocate(true);
        let mut output_buffer = resampler.output_buffer_allocate(true);

        let input_size = input_buffer[0].len();
        let output_size = output_buffer[0].len();

        println!("Input {}, Output {}", input_size, output_size);

        let mut new_deinterleaved = vec![
            Vec::with_capacity(
                n_frames * output_size / input_size
            );
            n_channels
        ];

        println!("New_deinterleaved capacity: {}", new_deinterleaved[0].capacity());

        let mut current_input_size = input_size;
        let mut current_output_size = output_size;

        for chunk_index in (0..n_frames).step_by(input_size) {
            if chunk_index + input_size > n_frames {
                current_input_size = n_frames - chunk_index;
                current_output_size =
                    current_input_size * output_size / input_size;
            }

            for channel_index in 0..n_channels {
                input_buffer[channel_index][0..current_input_size].copy_from_slice(
                    &deinterleaved[channel_index][chunk_index..chunk_index + current_input_size],
                );

                for index in current_input_size..input_size {
                    input_buffer[channel_index][index] = 0.0;
                }
            }

            resampler.process_into_buffer(&input_buffer, &mut output_buffer, None)?;

            println!("{}", current_output_size);

            for channel_index in 0..n_channels {
                new_deinterleaved[channel_index].extend(
                    output_buffer[channel_index][0..current_output_size]
                        .iter()
                        .map(|sample| S::from_sample(*sample)),
                );
            }
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
            1024,
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
            f_cutoff: 2.0,
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
