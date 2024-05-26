use std::error::Error;

//////////////////////////////////////////////////  Util  //////////////////////////////////////////////////

pub struct Util;

impl Util {
    fn _interleave<T: Copy + Default>(samples: &[Vec<T>]) -> Result<Vec<T>, Box<dyn Error>> {
        if samples.is_empty() {
            return Err("Samples is empty".into());
        }

        let n_channels = samples.len();
        let n_frames = samples[0].len();

        for channel in samples {
            if n_frames != channel.len() {
                return Err("Channels have different lengths".into());
            }
        }

        let mut interleaved = vec![Default::default(); n_channels * n_frames];

        for (channel_index, channel) in samples.iter().enumerate() {
            for (frame_index, sample) in channel.iter().enumerate() {
                interleaved[frame_index * n_channels + channel_index] = *sample;
            }
        }

        Ok(interleaved)
    }

    pub fn interleave<T: Copy + Default>(samples: &[Vec<T>]) -> Result<Vec<T>, Box<dyn Error>> {
        Util::_interleave(samples)
    }

    pub fn into_interleave<T: Copy + Default>(samples: Vec<Vec<T>>) -> Result<Vec<T>, Box<dyn Error>> {
        Util::_interleave(&samples)
    }

    fn _deinterleave<T: Copy + Default>(samples: &[T], channels: u16) -> Result<Vec<Vec<T>>, Box<dyn Error>> {
        if samples.is_empty() {
            return Err("Samples is empty".into());
        }

        let n_channels = channels as usize;

        if samples.len() % n_channels != 0 {
            return Err("Samples length not divisible by channels".into());
        }

        let n_frames = samples.len() / n_channels;

        let mut deinterleaved = vec![Vec::with_capacity(n_frames); n_channels];

        for (channel_index, channel_samples) in deinterleaved.iter_mut().enumerate() {
            channel_samples.extend(samples.iter().skip(channel_index).step_by(n_channels));
        }

        Ok(deinterleaved)
    }

    pub fn deinterleave<T: Copy + Default>(
        samples: &[T],
        channels: u16,
    ) -> Result<Vec<Vec<T>>, Box<dyn Error>> {
        Util::_deinterleave(samples, channels)
    }

    pub fn into_deinterleave<T: Copy + Default>(
        samples: Vec<T>,
        channels: u16,
    ) -> Result<Vec<Vec<T>>, Box<dyn Error>> {
        Util::_deinterleave(&samples, channels)
    }
}