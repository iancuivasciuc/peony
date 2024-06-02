use std::error::Error;

use rubato::{
    FastFixedIn, FftFixedInOut, PolynomialDegree, Sample as RubatoSample, SincFixedIn,
    SincInterpolationParameters, SincInterpolationType, VecResampler as RubatoResampler,
    WindowFunction,
};

use super::sample::Sample;
use super::Signal;

pub enum ResampleType {
    Fft,
    SincVeryHighQuality,
    SincHighQuality,
    SincMediumQuality,
    SincLowQuality,
    Fastest,
}

pub struct Resampler<S>
where
    S: Sample + RubatoSample,
{
    pub resample_type: ResampleType,
    _marker: std::marker::PhantomData<S>,
}

impl<S> Resampler<S>
where
    S: Sample + RubatoSample,
{
    pub fn new(resample_type: ResampleType) -> Resampler<S> {
        Resampler {
            resample_type,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn resample(
        &self,
        signal: &mut Signal<S>,
        new_sample_rate: u32,
    ) -> Result<(), Box<dyn Error>> {
        if signal.sample_rate == new_sample_rate {
            return Ok(());
        }

        let mut new_samples = vec![
            Vec::with_capacity(
                signal.len() * new_sample_rate as usize / signal.sample_rate as usize
            );
            signal.channels()
        ];

        let mut resampler: Box<dyn RubatoResampler<S>> = match self.resample_type {
            ResampleType::Fft => Resampler::_resampler_fft(signal, new_sample_rate)?,
            ResampleType::SincVeryHighQuality => {
                Resampler::_resampler_sinc_vhq(signal, new_sample_rate)?
            }
            ResampleType::SincHighQuality => {
                Resampler::_resampler_sinc_hq(signal, new_sample_rate)?
            }
            ResampleType::SincMediumQuality => {
                Resampler::_resampler_sinc_mq(signal, new_sample_rate)?
            }
            ResampleType::SincLowQuality => Resampler::_resampler_sinc_lq(signal, new_sample_rate)?,
            ResampleType::Fastest => Resampler::_resampler_fastest(signal, new_sample_rate)?,
        };

        let input_size = resampler.input_frames_max();
        let output_size = input_size * new_sample_rate as usize / signal.sample_rate as usize;

        let mut input_buffer = resampler.input_buffer_allocate(true);
        let mut output_buffer = resampler.output_buffer_allocate(true);

        for frame_index in (0..signal.len() - input_size).step_by(input_size) {
            for (channel_index, channel) in signal.samples.iter().enumerate() {
                input_buffer[channel_index][..input_size]
                    .copy_from_slice(&channel[frame_index..(frame_index + input_size)]);
            }

            resampler.process_into_buffer(&input_buffer, &mut output_buffer, None)?;

            for (channel_index, channel) in output_buffer.iter().enumerate() {
                new_samples[channel_index].extend(&channel[..output_size]);
            }
        }

        //  Last input buffer with a size smaller than input size

        let last_frame_index = signal.len() / input_size * input_size;

        let last_input_size = signal.len() - last_frame_index;
        let last_output_size =
            last_input_size * new_sample_rate as usize / signal.sample_rate as usize;

        for (channel_index, channel) in signal.samples.iter().enumerate() {
            input_buffer[channel_index][..last_input_size]
                .copy_from_slice(&channel[last_frame_index..(last_frame_index + last_input_size)]);

            for index in last_input_size..input_size {
                input_buffer[channel_index][index] = S::zero();
            }
        }

        resampler.process_into_buffer(&input_buffer, &mut output_buffer, None)?;

        for (channel_index, channel) in output_buffer.iter().enumerate() {
            new_samples[channel_index].extend(&channel[..last_output_size]);
        }

        signal.samples = new_samples;
        signal.sample_rate = new_sample_rate;

        Ok(())
    }

    fn _resampler_fft(
        signal: &mut Signal<S>,
        new_sample_rate: u32,
    ) -> Result<Box<dyn RubatoResampler<S>>, Box<dyn Error>> {
        let resampler = Box::new(FftFixedInOut::<S>::new(
            signal.sample_rate as usize,
            new_sample_rate as usize,
            2048,
            signal.channels(),
        )?);

        Ok(resampler)
    }

    fn _resampler_sinc_vhq(
        signal: &mut Signal<S>,
        new_sample_rate: u32,
    ) -> Result<Box<dyn RubatoResampler<S>>, Box<dyn Error>> {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = Box::new(SincFixedIn::<S>::new(
            new_sample_rate as f64 / signal.sample_rate as f64,
            1.0,
            params,
            2048,
            signal.channels(),
        )?);

        Ok(resampler)
    }

    fn _resampler_sinc_hq(
        signal: &mut Signal<S>,
        new_sample_rate: u32,
    ) -> Result<Box<dyn RubatoResampler<S>>, Box<dyn Error>> {
        let params = SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 64,
            window: WindowFunction::Blackman2,
        };
        let resampler = Box::new(SincFixedIn::<S>::new(
            new_sample_rate as f64 / signal.sample_rate as f64,
            1.0,
            params,
            1024,
            signal.channels(),
        )?);

        Ok(resampler)
    }

    fn _resampler_sinc_mq(
        signal: &mut Signal<S>,
        new_sample_rate: u32,
    ) -> Result<Box<dyn RubatoResampler<S>>, Box<dyn Error>> {
        let params = SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.90,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 32,
            window: WindowFunction::Hann2,
        };
        let resampler = Box::new(SincFixedIn::<S>::new(
            new_sample_rate as f64 / signal.sample_rate as f64,
            1.0,
            params,
            1024,
            signal.channels(),
        )?);

        Ok(resampler)
    }

    fn _resampler_sinc_lq(
        signal: &mut Signal<S>,
        new_sample_rate: u32,
    ) -> Result<Box<dyn RubatoResampler<S>>, Box<dyn Error>> {
        let params = SincInterpolationParameters {
            sinc_len: 32,
            f_cutoff: 0.85,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 16,
            window: WindowFunction::Hann2,
        };
        let resampler = Box::new(SincFixedIn::<S>::new(
            new_sample_rate as f64 / signal.sample_rate as f64,
            1.0,
            params,
            512,
            signal.channels(),
        )?);

        Ok(resampler)
    }

    fn _resampler_fastest(
        signal: &mut Signal<S>,
        new_sample_rate: u32,
    ) -> Result<Box<dyn RubatoResampler<S>>, Box<dyn Error>> {
        let resampler = Box::new(FastFixedIn::<S>::new(
            new_sample_rate as f64 / signal.sample_rate as f64,
            1.0,
            PolynomialDegree::Linear,
            512,
            signal.channels(),
        )?);

        Ok(resampler)
    }
}
