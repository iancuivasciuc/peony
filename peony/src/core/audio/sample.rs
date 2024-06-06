use num_traits::{Bounded, Float, FloatConst, Num, NumCast, ToPrimitive};

//////////////////////////////////////////////////  Sample  //////////////////////////////////////////////////
pub trait Sample: Num + Copy + NumCast + PartialOrd + Bounded {
    type MonoType: Sample;

    #[inline(always)]
    fn from_expect<T: ToPrimitive>(n: T) -> Self {
        Self::from(n).expect("Conversion should not fail")
    }
}

impl Sample for u8 {
    type MonoType = f32;
}
impl Sample for u16 {
    type MonoType = f32;
}
impl Sample for u32 {
    type MonoType = f32;
}
impl Sample for u64 {
    type MonoType = f64;
}
impl Sample for i8 {
    type MonoType = f32;
}
impl Sample for i16 {
    type MonoType = f32;
}
impl Sample for i32 {
    type MonoType = f32;
}
impl Sample for i64 {
    type MonoType = f64;
}
impl Sample for f32 {
    type MonoType = Self;
}
impl Sample for f64 {
    type MonoType = Self;
}

//////////////////////////////////////////////////  IntSample  //////////////////////////////////////////////////

pub trait IntSample: Sample {
    const IS_SIGNED: bool;

    #[inline(always)]
    fn max_bins() -> usize {
        (Self::max_value().to_i32().unwrap() - Self::min_value().to_i32().unwrap()) as usize
    }
}

impl IntSample for u8 {
    const IS_SIGNED: bool = false;
}
impl IntSample for u16 {
    const IS_SIGNED: bool = false;
}
impl IntSample for u32 {
    const IS_SIGNED: bool = false;
}
impl IntSample for u64 {
    const IS_SIGNED: bool = false;
}
impl IntSample for i8 {
    const IS_SIGNED: bool = true;
}
impl IntSample for i16 {
    const IS_SIGNED: bool = true;
}
impl IntSample for i32 {
    const IS_SIGNED: bool = true;
}
impl IntSample for i64 {
    const IS_SIGNED: bool = true;
}

//////////////////////////////////////////////////  FloatSample  //////////////////////////////////////////////////
pub trait FloatSample: Sample + Float + FloatConst {}

impl FloatSample for f32 {}
impl FloatSample for f64 {}
