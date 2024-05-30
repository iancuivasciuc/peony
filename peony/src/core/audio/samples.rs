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
                    if self.channels() == 1 {
                        return;
                    }

                    for index in 0..self.len() {
                        let mut sum: $t2 = Default::default();

                        for ch in 0..self.channels() {
                            sum += <$t2>::from_sample(self.samples[ch][index]);
                        }

                        self.samples[0][index] = <$t1>::from_sample(sum / self.channels() as $t2);
                    }

                    self.samples.truncate(1);
                    self.samples.shrink_to_fit();
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
