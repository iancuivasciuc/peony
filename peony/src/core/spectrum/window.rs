use crate::core::sample::FloatSample;

//////////////////////////////////////////////////  Window  //////////////////////////////////////////////////

pub trait Window<F>
where
    F: FloatSample,
{   
    fn window_iter(&self, len: usize) -> impl Iterator<Item = F>;

    fn window(&self, len: usize) -> Vec<F> {
        self.window_iter(len).collect()
    }
}

pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
}

pub struct Hann;

impl<F> Window<F> for Hann
where
    F: FloatSample,
{
    fn window_iter(&self, len: usize) -> impl Iterator<Item = F> {
        // Constants
        let one = F::one();
        let two = F::from_expect(2);
        let two_pi_n = two * F::PI() / F::from_expect(len);

        (0..len).map(move |n| {
            one / two * (one - (F::from_expect(n) * two_pi_n).cos())
        })
    }
}

pub struct Hamming;

impl<F> Window<F> for Hamming
where
    F: FloatSample,
{
    fn window_iter(&self, len: usize) -> impl Iterator<Item = F> {
        // Constants
        let t0 = F::from_expect(25 / 46);
        let t1 = F::one() - t0;
        let two_pi_n = F::from_expect(2) * F::PI() / F::from_expect(len);

        (0..len).map(move |n| {
            t0 - t1 * (F::from_expect(n) * two_pi_n).cos()
        })
    }
}

pub struct Blackman;

impl<F> Window<F> for Blackman
where
    F: FloatSample,
{
    fn window_iter(&self, len: usize) -> impl Iterator<Item = F> {
        // Constants
        let t0 = F::from_expect(7938 / 18608);
        let t1 = F::from_expect(9240 / 18608);
        let t2 = F::from_expect(1430 / 18608);
        let two_pi_n = F::from_expect(2) * F::PI() / F::from_expect(len);
        let four_pi_n = F::from_expect(4) * F::PI() / F::from_expect(len);

        (0..len).map(move |n| {
            t0 - t1 * (F::from_expect(n) * two_pi_n).cos() + t2 * (F::from_expect(n) * four_pi_n).cos()
        })
    }
}

pub struct BlackmanHarris;

impl<F> Window<F> for BlackmanHarris
where
    F: FloatSample,
{
    fn window_iter(&self, len: usize) -> impl Iterator<Item = F> {
        // Constants
        let t0 = F::from_expect(35875 / 65536);
        let t1 = F::from_expect(48829 / 65536);
        let t2 = F::from_expect(14128 / 65536);
        let t3 = F::from_expect(1118 / 65536);
        let two_pi_n = F::from_expect(2) * F::PI()  / F::from_expect(len);
        let four_pi_n = F::from_expect(4) * F::PI() / F::from_expect(len);
        let six_pi_n = F::from_expect(6) * F::PI() / F::from_expect(len);

        (0..len).map(move |n| {
            t0 - t1 * (F::from_expect(n) * two_pi_n).cos() + t2 * (F::from_expect(n) * four_pi_n).cos() - t3 * (F::from_expect(n) * six_pi_n).cos()
        })
    }
}


