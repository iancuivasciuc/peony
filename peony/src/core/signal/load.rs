use std::error::Error;
use std::ffi::OsStr;
use std::path::Path;
use std::time::Duration;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::conv::ConvertibleSample as SymphoniaSample;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatReader, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;
use symphonia::core::units::Time;
use symphonia::default::{get_codecs, get_probe};

use super::Signal;
use crate::core::sample::Sample;
use crate::core::util::into_deinterleave;

//////////////////////////////////////////////////  SignalLoader  //////////////////////////////////////////////////

pub(crate) struct SignalLoader<S>
where
    S: Sample + SymphoniaSample,
{
    pub path: String,
    pub offset: Duration,
    pub duration: Option<Duration>,
    _marker: std::marker::PhantomData<S>,
}

impl<S> SignalLoader<S>
where
    S: Sample + SymphoniaSample,
{
    pub fn new(path: &str, offset: Duration, duration: Option<Duration>) -> Self {
        SignalLoader {
            path: path.to_string(),
            offset,
            duration,
            _marker: std::marker::PhantomData,
        }
    }

    fn _n_frames(&self, all_frames: u64, sample_rate: u32) -> u64 {
        let mut frames = all_frames - (self.offset.as_secs_f64() * sample_rate as f64) as u64;

        if let Some(duration) = self.duration {
            frames = std::cmp::min(frames, (duration.as_secs_f64() * sample_rate as f64) as u64);
        }

        frames
    }

    fn _decode_format(
        &self,
        mut format: Box<dyn FormatReader>,
    ) -> Result<Signal<S>, Box<dyn Error>> {
        let track = format.default_track().unwrap();

        let mut decoder = get_codecs().make(&track.codec_params, &Default::default())?;

        let channels = track
            .codec_params
            .channels
            .ok_or("unknown channels")?
            .count();
        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or("unknown sample rate")?;

        let frames = self._n_frames(
            track.codec_params.n_frames.ok_or("unknown frames")?,
            sample_rate,
        );
        let frames_per_packet = track.codec_params.max_frames_per_packet.unwrap_or_default();

        // let frames_first_packet = frames_per_packet
        //     - ((self.offset.as_secs_f64() * sample_rate as f64) as u64 % frames_per_packet);
        // let frames_first_packet = std::cmp::min(frames_first_packet, frames);

        // let mut frames_to_read = frames - frames_first_packet;
        let mut frames_to_read = frames;

        let mut samples: Vec<S> = Vec::with_capacity((frames * channels as u64) as usize);
        let mut sample_buffer = None;

        format
            .seek(
                SeekMode::Accurate,
                SeekTo::Time {
                    time: Time::from(self.offset),
                    track_id: Some(track.id),
                },
            )
            .unwrap();

        while let Ok(packet) = format.next_packet() {
            match decoder.decode(&packet) {
                Ok(audio_buffer) => {
                    if sample_buffer.is_none() {
                        let spec = *audio_buffer.spec();
                        let duration = audio_buffer.capacity() as u64;

                        sample_buffer = Some(SampleBuffer::<S>::new(duration, spec));
                    }

                    if let Some(buffer) = &mut sample_buffer {
                        buffer.copy_interleaved_ref(audio_buffer);

                        let slice = buffer.samples();

                        // if samples.is_empty() {
                            // samples.extend_from_slice(
                            //     &slice[slice.len()
                            //         - (frames_first_packet * channels as u64) as usize..],
                            // );
                        // } else 
                        if frames_to_read <= (slice.len() as u64 / channels as u64) {
                            samples.extend_from_slice(
                                &slice[..(frames_to_read * channels as u64) as usize],
                            );

                            break;
                        } else {
                            samples.extend_from_slice(slice);

                            frames_to_read -= slice.len() as u64 / channels as u64;
                        }
                    }
                }
                Err(SymphoniaError::DecodeError(_)) => (),
                Err(_) => break,
            }
        }

        Ok(Signal {
            samples: into_deinterleave(samples, channels)?,
            sample_rate,
        })
    }

    pub fn load(&self) -> Result<Signal<S>, Box<dyn Error>> {
        let format = _file_format(&self.path)?;

        self._decode_format(format)
    }
}

fn _file_format(path: &str) -> Result<Box<dyn FormatReader>, Box<dyn Error>> {
    let path = Path::new(path);
    let file = std::fs::File::open(path)?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(OsStr::to_str) {
        hint.with_extension(extension);
    }

    let probed = get_probe().format(&hint, mss, &Default::default(), &Default::default())?;

    let format = probed.format;

    Ok(format)
}

pub fn get_file_duration(path: &str) -> Result<Duration, Box<dyn Error>> {
    let format = _file_format(path)?;

    let track = format.default_track().unwrap();

    let n_frames = match track.codec_params.n_frames {
        Some(n_frames) => n_frames,
        None => return Err("No duration found".into()),
    };
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("No sample rate found")?;

    Ok(Duration::from_secs_f64(
        n_frames as f64 / sample_rate as f64,
    ))
}

pub fn get_file_sample_rate(path: &str) -> Result<u32, Box<dyn Error>> {
    let format = _file_format(path)?;

    let track = format.default_track().unwrap();

    track
        .codec_params
        .sample_rate
        .ok_or("No sample rate found".into())
}
