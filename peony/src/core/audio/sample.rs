use num_traits::{Float, FloatConst, FromPrimitive};

//////////////////////////////////////////////////  FloatSample  //////////////////////////////////////////////////
pub trait Sample: Float + FloatConst + FromPrimitive {}

impl Sample for f32 {}
impl Sample for f64 {}
