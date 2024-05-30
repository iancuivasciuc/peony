use std::error::Error;

fn _interleave<T: Copy + Default>(samples: &[Vec<T>]) -> Result<Vec<T>, Box<dyn Error>> {
    if samples.is_empty() {
        return Err("Samples is empty".into());
    }

    let channels = samples.len();
    let frames = samples[0].len();

    for channel in samples {
        if frames != channel.len() {
            return Err("Channels have different lengths".into());
        }
    }

    let mut interleaved = vec![Default::default(); channels * frames];

    for (channel_index, channel) in samples.iter().enumerate() {
        for (frame_index, sample) in channel.iter().enumerate() {
            interleaved[frame_index * channels + channel_index] = *sample;
        }
    }

    Ok(interleaved)
}

pub fn interleave<T: Copy + Default>(samples: &[Vec<T>]) -> Result<Vec<T>, Box<dyn Error>> {
    _interleave(samples)
}

pub fn into_interleave<T: Copy + Default>(samples: Vec<Vec<T>>) -> Result<Vec<T>, Box<dyn Error>> {
    _interleave(&samples)
}

fn _deinterleave<T: Copy + Default>(
    samples: &[T],
    channels: usize,
) -> Result<Vec<Vec<T>>, Box<dyn Error>> {
    if samples.is_empty() {
        return Err("Samples is empty".into());
    }

    if samples.len() % channels != 0 {
        return Err("Samples length not divisible by channels".into());
    }

    let frames = samples.len() / channels;

    let mut deinterleaved = vec![Vec::with_capacity(frames); channels];

    for (channel_index, channel_samples) in deinterleaved.iter_mut().enumerate() {
        channel_samples.extend(samples.iter().skip(channel_index).step_by(channels));
    }

    Ok(deinterleaved)
}

pub fn deinterleave<T: Copy + Default>(
    samples: &[T],
    channels: usize,
) -> Result<Vec<Vec<T>>, Box<dyn Error>> {
    _deinterleave(samples, channels)
}

pub fn into_deinterleave<T: Copy + Default>(
    samples: Vec<T>,
    channels: usize,
) -> Result<Vec<Vec<T>>, Box<dyn Error>> {
    _deinterleave(&samples, channels)
}
