use std::error::Error;
use std::time::Duration;

use symphonia::core::conv::{ConvertibleSample as SymphoniaSample, FromSample};

use rodio::{Sample as RodioSample, Source};

mod load;
// mod resample;
mod samples;
mod util;

use load::SignalLoader;
// use resample::{ResampleType, Resampler};
use samples::Samples;

//////////////////////////////////////////////////  Signal  //////////////////////////////////////////////////

pub struct Signal<S>
where
    S: SymphoniaSample,
{
    pub samples: Vec<Vec<S>>,
    pub sample_rate: u32,
}

impl<S> Signal<S>
where
    S: SymphoniaSample,
{
    //  Utility functions
    pub fn len(&self) -> usize {
        self.samples[0].len()
    }

    pub fn channels(&self) -> usize {
        self.samples.len()
    }

    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(
            self.len() as f64 / self.sample_rate as f64,
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

    // pub fn rodio_source<R>(self) -> SignalRodioSource<S, R>
    // where
    //     R: RodioSample + FromSample<S>,
    // {
    //     SignalRodioSource {
    //         signal: self,
    //         index: 0,
    //         _marker: std::marker::PhantomData,
    //     }
    // }

    // pub fn resample(
    //     &mut self,
    //     new_sample_rate: u32,
    //     resample_type: ResampleType,
    // ) -> Result<(), Box<dyn Error>>
    // where
    //     f64: symphonia::core::conv::FromSample<S>,
    // {
    //     let resampler = Resampler::new(resample_type);

    //     resampler.resample(self, new_sample_rate)?;

    //     Ok(())
    // }

    pub fn tone(frequency: f64, sample_rate: u32, duration: Duration) -> Signal<S> {
        let ts = 1.0 / sample_rate as f64;

        let length = (duration.as_secs_f64() * sample_rate as f64) as u64;

        let samples = (0..length)
            .map(|n| S::from_sample((2.0 * std::f64::consts::PI * frequency * n as f64 * ts).cos()))
            .collect();

        Signal {
            samples: vec![samples],
            sample_rate,
        }
    }

    pub fn chirp(
        freq1: f64,
        freq2: f64,
        sample_rate: u32,
        duration: Duration,
        linear: bool,
    ) -> Signal<S> {
        let ts = 1.0 / sample_rate as f64;

        let length = (duration.as_secs_f64() * sample_rate as f64) as u64;

        let samples = if linear {
            (0..length)
                .map(|n| {
                    S::from_sample(
                        (2.0 * std::f64::consts::PI
                            * (freq1 + (freq2 - freq1) * n as f64 * ts / duration.as_secs_f64())
                            * n as f64
                            * ts)
                            .cos(),
                    )
                })
                .collect()
        } else {
            (0..length)
                .map(|n| {
                    S::from_sample(
                        (2.0 * std::f64::consts::PI
                            * (freq1
                                * (freq2 / freq1).powf(n as f64 * ts / duration.as_secs_f64())
                                * n as f64
                                * ts))
                            .cos(),
                    )
                })
                .collect()
        };

        Signal {
            samples: vec![samples],
            sample_rate,
        }
    }
}

//////////////////////////////////////////////////  SignalRodioSource  //////////////////////////////////////////////////

// pub struct SignalRodioSource<S, R>
// where
//     S: SymphoniaSample,
//     R: RodioSample + FromSample<S>,
// {
//     pub signal: Signal<S>,
//     pub index: usize,
//     pub _marker: std::marker::PhantomData<R>,
// }

// impl<S, R> SignalRodioSource<S, R>
// where
//     S: SymphoniaSample,
//     R: RodioSample + FromSample<S>,
// {
//     pub fn inner(self) -> Signal<S> {
//         self.signal
//     }
// }

// impl<S, R> Iterator for SignalRodioSource<S, R>
// where
//     S: SymphoniaSample,
//     R: RodioSample + FromSample<S>,
// {
//     type Item = R;

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index < self.signal.len() {
//             self.index += 1;

//             Some(<R as FromSample<S>>::from_sample(
//                 self.signal.samples[self.index - 1],
//             ))
//         } else {
//             None
//         }
//     }

//     fn size_hint(&self) -> (usize, Option<usize>) {
//         (
//             self.index - self.signal.len(),
//             Some(self.index - self.signal.len()),
//         )
//     }
// }

// impl<S, R> ExactSizeIterator for SignalRodioSource<S, R>
// where
//     S: SymphoniaSample,
//     R: RodioSample + FromSample<S>,
// {
// }

// impl<S, R> Source for SignalRodioSource<S, R>
// where
//     S: SymphoniaSample,
//     R: RodioSample + FromSample<S>,
// {
//     fn current_frame_len(&self) -> Option<usize> {
//         Some(self.signal.samples.len() / self.signal.channels as usize)
//     }

//     fn channels(&self) -> u16 {
//         self.signal.channels
//     }

//     fn sample_rate(&self) -> u32 {
//         self.signal.sample_rate
//     }

//     fn total_duration(&self) -> Option<Duration> {
//         Some(self.signal.duration())
//     }
// }

//////////////////////////////////////////////////  SignalRodioSource  //////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use rodio::Sink;

    use super::*;

    #[test]
    fn load_test() {
        let signal: Signal<f32> = Signal::load(
            "Shape of You.wav",
            Duration::from_secs(0),
            Some(Duration::from_secs(100)),
        )
        .unwrap();

        println!("{}, {}, {}, {:?}", signal.len(), signal.channels(), signal.sample_rate, signal.duration());

        for ch in 0..signal.channels() {
            for index in 0..10 {
                print!("{:.4} ", signal.samples[ch][index]);
            }
            println!()
        }

        // signal.to_mono();

        // signal.resample(60000, ResampleType::SincVeryHighQuality).unwrap();

        // let signal: Signal<f32> =
        //     Signal::chirp(440.0, 880.0, 22050, Duration::from_secs(10), false);

        // for i in 0..10 {
        //     print!("{} ", signal.samples[i]);
        // }
        // println!();

        // println!(
        //     "Length: {}, Duration: {:?}",
        //     signal.len(),
        //     signal.duration()
        // );

        // let source = signal.rodio_source::<f32>();

        // let (_stream, stream_handle) = rodio::OutputStream::try_default().unwrap();
        // let sink = Sink::try_new(&stream_handle).unwrap();
        // sink.append(source);
        // sink.sleep_until_end()
    }
}
