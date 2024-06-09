use crate::core::sample::FloatSample;

//////////////////////////////////////////////////  Window  //////////////////////////////////////////////////

#[derive(Clone, Copy)]
pub enum WindowType {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
}

pub struct Window<F>
where
    F: FloatSample,
{
    pub len: usize,
    pub window_type: WindowType,
    _marker: std::marker::PhantomData<F>,
}

impl<F> Window<F>
where
    F: FloatSample + 'static,
{
    pub fn new(len: usize, window_type: WindowType) -> Self {
        Window {
            len,
            window_type,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn window_iter(&self) -> Box<dyn Iterator<Item = F>> {
        match self.window_type {
            WindowType::Rectangular => self._rectangular_iter(),
            WindowType::Hann => self._hann_iter(),
            WindowType::Hamming => self._hamming_iter(),
            WindowType::Blackman => self._blackman_iter(),
            WindowType::BlackmanHarris => self._blackman_harris_iter(),
        }
    }

    pub fn window(&self) -> Vec<F> {
        self.window_iter().collect()
    }

    fn _rectangular_iter(&self) -> Box<dyn Iterator<Item = F>> {
        let one = F::one();

        Box::new((0..self.len).map(move |_| one))
    }

    fn _hann_iter(&self) -> Box<dyn Iterator<Item = F>> {
        // Constants
        let one = F::one();
        let two = F::from_expect(2);
        let two_pi_n = two * F::PI() / F::from_expect(self.len);

        Box::new(
            (0..self.len).map(move |n| one / two * (one - (F::from_expect(n) * two_pi_n).cos())),
        )
    }

    fn _hamming_iter(&self) -> Box<dyn Iterator<Item = F>> {
        // Constants
        let t0 = F::from_expect(25 / 46);
        let t1 = F::one() - t0;
        let two_pi_n = F::from_expect(2) * F::PI() / F::from_expect(self.len);

        Box::new((0..self.len).map(move |n| t0 - t1 * (F::from_expect(n) * two_pi_n).cos()))
    }

    fn _blackman_iter(&self) -> Box<dyn Iterator<Item = F>> {
        // Constants
        let t0 = F::from_expect(7938 / 18608);
        let t1 = F::from_expect(9240 / 18608);
        let t2 = F::from_expect(1430 / 18608);
        let two_pi_n = F::from_expect(2) * F::PI() / F::from_expect(self.len);
        let four_pi_n = F::from_expect(4) * F::PI() / F::from_expect(self.len);

        Box::new((0..self.len).map(move |n| {
            t0 - t1 * (F::from_expect(n) * two_pi_n).cos()
                + t2 * (F::from_expect(n) * four_pi_n).cos()
        }))
    }

    fn _blackman_harris_iter(&self) -> Box<dyn Iterator<Item = F>> {
        // Constants
        let t0 = F::from_expect(35875 / 65536);
        let t1 = F::from_expect(48829 / 65536);
        let t2 = F::from_expect(14128 / 65536);
        let t3 = F::from_expect(1118 / 65536);
        let two_pi_n = F::from_expect(2) * F::PI() / F::from_expect(self.len);
        let four_pi_n = F::from_expect(4) * F::PI() / F::from_expect(self.len);
        let six_pi_n = F::from_expect(6) * F::PI() / F::from_expect(self.len);

        Box::new((0..self.len).map(move |n| {
            t0 - t1 * (F::from_expect(n) * two_pi_n).cos()
                + t2 * (F::from_expect(n) * four_pi_n).cos()
                - t3 * (F::from_expect(n) * six_pi_n).cos()
        }))
    }
}
