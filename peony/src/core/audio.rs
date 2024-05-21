use cpal::Sample as CpalSample;
use rodio::{Sample as RodioSample, Source};
use std::{fs::File, io::BufReader, time::Duration};

mod loader;
mod sample;

use loader::Loader;
use sample::CpalSampleTraits;

////////////////////////////////////////////  Signal  ////////////////////////////////////////////

pub struct Signal<S>
where
    S: CpalSample,
{
    samples: Vec<S>,
    channels: u16,
    sample_rate: u32,
}

impl<S> Signal<S>
where
    S: CpalSample,
{
    pub fn load(path: &str) -> SignalBuilder<S> {
        SignalBuilder {
            path: path.to_string(),
            offset: Duration::from_secs(0),
            duration: None,
            mono: false,
            sample_rate: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn total_duration(&self) -> Duration {
        Duration::from_secs_f64(
            self.samples.len() as f64 / (self.channels as f64 * self.sample_rate as f64),
        )
    }

    pub fn bits_per_sample(&self) -> usize {
        std::mem::size_of::<S>() * 8
    }

    pub fn data_type(&self) -> &'static str {
        std::any::type_name::<S>()
    }

    //  Iterator
    pub fn into_iter(self) -> SignalIter<std::vec::IntoIter<S>> {
        SignalIter {
            samples: self.samples.into_iter(),
            channels: self.channels,
            sample_rate: self.sample_rate,
        }
    }
}

////////////////////////////////////////////  SignalBuilder  ////////////////////////////////////////////

pub struct SignalBuilder<S>
where
    S: CpalSample,
{
    path: String,
    offset: Duration,
    duration: Option<Duration>,
    mono: bool,
    sample_rate: Option<u32>,
    _marker: std::marker::PhantomData<S>,
}

impl<S> SignalBuilder<S>
where
    S: CpalSampleTraits,
{
    pub fn offset(mut self, offset: Duration) -> Self {
        self.offset = offset;
        self
    }

    pub fn duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    pub fn to_mono(mut self) -> Self {
        self.mono = true;
        self
    }

    pub fn build(self) -> Result<Signal<S>, std::io::Error> {
        let file = match File::open(self.path) {
            Ok(file) => BufReader::new(file),
            Err(e) => return Err(e),
        };

        let loader = Loader::new(file);

        Ok(loader.load())
    }
}

////////////////////////////////////////////  SignalIter  ////////////////////////////////////////////

pub struct SignalIter<I>
where
    I: ExactSizeIterator,
    I::Item: CpalSample,
{
    samples: I,
    channels: u16,
    sample_rate: u32,
}

impl<I> Iterator for SignalIter<I>
where
    I: ExactSizeIterator,
    I::Item: CpalSample,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.samples.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.samples.size_hint()
    }
}

impl<I> ExactSizeIterator for SignalIter<I>
where
    I: ExactSizeIterator,
    I::Item: CpalSample,
{
}

impl<I> Source for SignalIter<I>
where
    I: ExactSizeIterator,
    I::Item: RodioSample,
{
    fn current_frame_len(&self) -> Option<usize> {
        Some(self.samples.len() / self.channels as usize)
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        Some(Duration::from_secs_f64(
            self.samples.len() as f64 / (self.channels as f64 * self.sample_rate as f64),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tests() {
        let signal: Signal<f32> = Signal::load("Shape of You.wav").build().unwrap();
        println!(
            "{}, {}, {}, {:?}",
            signal.len(),
            signal.channels,
            signal.sample_rate,
            signal.total_duration()
        );
    }
}
