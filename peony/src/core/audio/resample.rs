use std::error::Error;
use symphonia::core::conv::{ConvertibleSample as SymphoniaSample, FromSample};

use rubato::{
    FftFixedInOut, SincFixedIn, SincInterpolationParameters, SincInterpolationType, VecResampler,
    WindowFunction, VecResampler as RubatoResampler
};

use super::Signal;
use super::util::Util;

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
        Resampler {
            resample_type,
        }
    }

    pub fn resample<S>(&self, signal: &mut Signal<S>, new_sample_rate: u32) -> Result<(), Box<dyn Error>>
    where
        S: SymphoniaSample,
        f64: symphonia::core::conv::FromSample<S>,
    {
        let n_channels = signal.channels as usize;
        let n_frames = signal.len() / n_channels;

        let samples_f64 = signal
            .samples
            .iter()
            .map(|sample| f64::from_sample(*sample))
            .collect();

        let deinterleaved = Util::into_deinterleave(samples_f64, n_channels as u16)?;

        let resampler: Box<dyn RubatoResampler::<f64>> = match self.resample_type {
            ResampleType::Fft => Box::new(FftFixedInOut::<f64>::new(
                signal.sample_rate as usize,
                new_sample_rate as usize,
                1024,
                n_channels,
            ).unwrap()),
            // ResampleType::SincVeryHighQuality => Resampler::_resample_sinc_vhq(signal, new_sample_rate),
            _ => Box::new(FftFixedInOut::<f64>::new(
                signal.sample_rate as usize,
                new_sample_rate as usize,
                1024,
                n_channels,
            ).unwrap()),


            // ResampleType::SincHighQuality => self._resample_sinc_hq(signal, new_sample_rate),
            // ResampleType::SincLowQuality => self._resample_sinc_lq(signal, new_sample_rate),
            // ResampleType::Fastest => self._resample_fastest(signal, new_sample_rate),
        };

        Ok(())
    }

    //     match resample_type {
    //         ResampleType::Fft => Ok(()),
    //         ResampleType::SincHighQuality => Ok(()),
    //         _ => Ok(()),

    // fn _resample_fft<S>(signal: &mut Signal<S>, new_sample_rate: u32) -> Result<(), Box<dyn Error>>
    // where
    //     S: SymphoniaSample,
    //     f64: symphonia::core::conv::FromSample<S>,
    // {
    //     let mut resampler = FftFixedInOut::<f64>::new(
    //         signal.sample_rate as usize,
    //         new_sample_rate as usize,
    //         1024,
    //         n_channels,
    //     )?;

    //     let mut new_deinterleaved = vec![
    //         Vec::with_capacity(
    //             n_frames * new_sample_rate as usize / signal.sample_rate as usize
    //         );
    //         n_channels
    //     ];

    //     let mut input_buffer = resampler.input_buffer_allocate(true);
    //     let mut output_buffer = resampler.output_buffer_allocate(true);

    //     let input_size = input_buffer[0].len();
    //     let output_size = output_buffer[0].len();

    //     let mut current_input_size = input_size;
    //     let mut current_output_size = output_size;

    //     for chunk_index in (0..n_frames).step_by(input_size) {
    //         if chunk_index + input_size > n_frames {
    //             current_input_size = n_frames - chunk_index;
    //             current_output_size = current_input_size * new_sample_rate as usize / signal.sample_rate as usize;
    //         }

    //         for channel_index in 0..n_channels {
    //             input_buffer[channel_index][0..current_input_size].copy_from_slice(
    //                 &deinterleaved[channel_index]
    //                     [chunk_index..chunk_index + current_input_size],
    //             );

    //             for index in current_input_size..input_size {
    //                 input_buffer[channel_index][index] = 0.0;
    //             }
    //         }

    //         resampler.process_into_buffer(&input_buffer, &mut output_buffer, None)?;

    //         for channel_index in 0..n_channels {
    //             new_deinterleaved[channel_index].extend(
    //                 output_buffer[channel_index][0..current_output_size]
    //                     .iter()
    //                     .map(|sample| S::from_sample(*sample)),
    //             );
    //         }
    //     }

    //     let new_samples = Util::into_interleave(new_deinterleaved)?;

    //     signal.samples = new_samples;
    //     signal.sample_rate = new_sample_rate;

    //     Ok(())
    // }

    fn _resample_sinc_vhq<S>(signal: &mut Signal<S>, new_sample_rate: u32) -> Result<(), Box<dyn Error>>
    where
        S: SymphoniaSample,
        f64: symphonia::core::conv::FromSample<S>,
    {
        let n_channels = signal.channels as usize;
        let n_frames = signal.len() / n_channels;

        let samples_f64 = signal
            .samples
            .iter()
            .map(|sample| f64::from_sample(*sample))
            .collect();

        let deinterleaved = Util::into_deinterleave(samples_f64, n_channels as u16)?;

        let mut resampler = FftFixedInOut::<f64>::new(
            signal.sample_rate as usize,
            new_sample_rate as usize,
            1024,
            n_channels,
        )?;

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f64>::new(
            new_sample_rate as f64 / signal.sample_rate as f64,
            1.0,
            params,
            1024,
            2,
        )?;

        Ok(())
    }

    fn _resample_sinc_hq<T>(&self, signal: &mut Signal<T>, new_sample_rate: u32)
    where
        T: SymphoniaSample
    {
        
    }

    fn _resample_sinc_mq<T>(&self, signal: &mut Signal<T>, new_sample_rate: u32)
    where
        T: SymphoniaSample
    {
        
    }

    fn _resample_sinc_lq<T>(&self, signal: &mut Signal<T>, new_sample_rate: u32)
    where
        T: SymphoniaSample
    {

    }

    fn _resample_fastest<T>(&self, signal: &mut Signal<T>, new_sample_rate: u32)
    where
        T: SymphoniaSample
    {

    }
}