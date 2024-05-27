use symphonia::core::conv::FromSample;
use symphonia::core::sample::{i24, u24};

use super::Signal;

//////////////////////////////////////////////////  Samples  //////////////////////////////////////////////////

pub trait Samples {
    // type Sample: SymphoniaSample;

    fn to_mono(&mut self);
}

macro_rules! impl_samples {
    ( $($t1:ty, $t2:ty),*) => {
        $(
            impl Samples for Signal<$t1> {
                fn to_mono(&mut self) {
                    if self.channels == 1 {
                        return;
                    }

                    let channels: usize = self.channels as usize;
                    let mut sum: $t2 = Default::default();

                    for index in 0..self.len() {
                        sum += <$t2>::from_sample(self.samples[index]);

                        if index % channels == channels - 1 {
                            self.samples[index / channels] = <$t1>::from_sample(sum / channels as $t2);
                            sum = Default::default();
                        }
                    }

                    self.samples.truncate(self.samples.len() / channels);
                    self.samples.shrink_to_fit();

                    self.channels = 1;
                }
            }
        )*
    };
}

//  Unsigned
impl_samples!(u8, f32);
impl_samples!(u16, f32);
impl_samples!(u24, f32);
impl_samples!(u32, f32);

//  Signed
impl_samples!(i8, f32);
impl_samples!(i16, f32);
impl_samples!(i24, f32);
impl_samples!(i32, f32);

//  Float
impl_samples!(f32, f32);
impl_samples!(f64, f64);
