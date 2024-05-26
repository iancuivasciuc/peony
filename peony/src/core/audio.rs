use std::error::Error;
use std::time::Duration;

use symphonia::core::conv::{ConvertibleSample as SymphoniaSample, FromSample};
use symphonia::core::sample::{i24, u24};

use rodio::{Sample as RodioSample, Source};

mod load;
mod samples;
mod resample;
mod util;

use load::SignalLoader;
use samples::Samples;
use resample::{ResampleType, Resampler};
use util::Util;

//////////////////////////////////////////////////  Signal  //////////////////////////////////////////////////

pub struct Signal<S>
where
    S: SymphoniaSample,
{
    pub samples: Vec<S>,
    pub channels: u16,
    pub sample_rate: u32,
}

impl<S> Signal<S>
where
    S: SymphoniaSample,
{
    //  Utility functions
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

    pub fn resample(
        &mut self,
        new_sample_rate: u32,
        resample_type: ResampleType,
    ) -> Result<(), Box<dyn Error>>
    where
        f64: symphonia::core::conv::FromSample<S>
    {
        let resampler = Resampler::new(resample_type);

        resampler.resample(self, new_sample_rate);

        Ok(())
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
        if self.index < self.signal.len() {
            self.index += 1;

            Some(<R as FromSample<S>>::from_sample(
                self.signal.samples[self.index - 1],
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
        Some(self.signal.samples.len() / self.signal.channels as usize)
    }

    fn channels(&self) -> u16 {
        self.signal.channels
    }

    fn sample_rate(&self) -> u32 {
        self.signal.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        Some(self.signal.total_duration())
    }
}

//////////////////////////////////////////////////  SignalRodioSource  //////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tests() {
        let mut signal: Signal<f32> = Signal::load(
            "Shape of You.wav",
            Duration::from_secs(15),
            Some(Duration::from_secs(10)),
        )
        .unwrap();

        signal.to_mono();

        signal.resample(22000, ResampleType::Fft).unwrap();

        println!("{}", signal.len());

        let (_stream, stream_handle) = rodio::OutputStream::try_default().unwrap();

        let source = signal.rodio_source::<f32>();

        let _ = stream_handle.play_raw(source.convert_samples());

        std::thread::sleep(std::time::Duration::from_secs(10));
    }
}
