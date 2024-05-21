use std::io::{Read, Seek, SeekFrom};

use cpal::{FromSample, Sample as CpalSample};
use hound::{Sample as HoundSample, SampleFormat, WavReader};

use super::{sample::CpalSampleTraits, Signal};

pub struct Loader<R, S>
where
    R: Read + Seek,
    S: CpalSampleTraits,
{
    data: R,
    _marker: std::marker::PhantomData<S>,
}

impl<R, S> Loader<R, S>
where
    R: Read + Seek,
    S: CpalSampleTraits,
{
    pub fn new(data: R) -> Loader<R, S> {
        Loader {
            data,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn load(self) -> Signal<S> {
        self.from_wav()
    }

    fn is_wav(&mut self) -> bool {
        let stream_position = self.data.stream_position().unwrap();

        if WavReader::new(self.data.by_ref()).is_err() {
            self.data.seek(SeekFrom::Start(stream_position)).unwrap();

            return false;
        }

        self.data.seek(SeekFrom::Start(stream_position)).unwrap();

        true
    }

    fn is_flac(&mut self) -> bool {
        todo!();
    }

    fn is_mp3(&mut self) -> bool {
        todo!();
    }

    fn from_wav(self) -> Signal<S> {
        let reader = WavReader::new(self.data).unwrap();
        let spec = reader.spec();

        let sample_rate = spec.sample_rate;
        let channels = spec.channels;

        let num_samples = reader.len();
        let mut samples: Vec<S> = Vec::with_capacity(num_samples as usize);

        match (spec.bits_per_sample, spec.sample_format) {
            (8, SampleFormat::Int) => {
                reader.into_samples::<i8>().for_each(|sample| {
                    samples.push(sample.expect("Failed to read wav samples").to_sample::<S>())
                });
            }
            (16, SampleFormat::Int) => {
                reader.into_samples::<i16>().for_each(|sample| {
                    samples.push(sample.expect("Failed to read wav samples").to_sample::<S>())
                });
            }
            (32, SampleFormat::Int) => {
                reader.into_samples::<i32>().for_each(|sample| {
                    samples.push(sample.expect("Failed to read wav samples").to_sample::<S>())
                });
            }
            (32, SampleFormat::Float) => {
                reader.into_samples::<f32>().for_each(|sample| {
                    samples.push(sample.expect("Failed to read wav samples").to_sample::<S>())
                });
            }
            (_, SampleFormat::Int) => {
                reader.into_samples::<i32>().for_each(|sample| {
                    samples.push(sample.expect("Failed to read wav samples").to_sample::<S>())
                });
            }
            (_, SampleFormat::Float) => {
                reader.into_samples::<f32>().for_each(|sample| {
                    samples.push(sample.expect("Failed to read wav samples").to_sample::<S>())
                });
            }
        }

        Signal {
            samples,
            channels,
            sample_rate,
        }
    }
}
