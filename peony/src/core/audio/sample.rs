use cpal::{FromSample, Sample as CpalSample, I24, I48, U24, U48};

pub trait CpalSampleTraits:
    CpalSample
    + FromSample<u8>
    + FromSample<u16>
    + FromSample<U24>
    + FromSample<u32>
    + FromSample<U48>
    + FromSample<u64>
    + FromSample<i8>
    + FromSample<i16>
    + FromSample<I24>
    + FromSample<i32>
    + FromSample<I48>
    + FromSample<i64>
    + FromSample<f32>
    + FromSample<f64>
{
}

impl<T> CpalSampleTraits for T where
    T: CpalSample
        + FromSample<u8>
        + FromSample<u16>
        + FromSample<U24>
        + FromSample<u32>
        + FromSample<U48>
        + FromSample<u64>
        + FromSample<i8>
        + FromSample<i16>
        + FromSample<I24>
        + FromSample<i32>
        + FromSample<I48>
        + FromSample<i64>
        + FromSample<f32>
        + FromSample<f64>
{
}
