use symphonia::core::conv::ConvertibleSample as SymphoniaSample;
use symphonia::core::sample::{i24, u24};

//////////////////////////////////////////////////  Samples  //////////////////////////////////////////////////

// pub trait Samples {
//     // type Sample: SymphoniaSample;

//     fn to_mono(&mut self);
// }

// pub trait Sample: SymphoniaSample + {
//     type ProcessType: Sample;
// }

// macro_rules! impl_sample {
//     ( $($t:ty, $proc:ty),*) => {
//         $(
//             impl Sample for $t {
//                 type ProcessType: = $proc;
//             }
//         )*
//     }
// }

// //  Unsigned
// impl_sample!(u8, f32);
// impl_sample!(u16, f32);
// impl_sample!(u24, f32);
// impl_sample!(u32, f32);

// //  Signed
// impl_sample!(i8, f32);
// impl_sample!(i16, f32);
// impl_sample!(i24, f32);
// impl_sample!(i32, i32);

// //  Float
// impl_sample!(f32, f32);
// impl_sample!(f64, f64);
