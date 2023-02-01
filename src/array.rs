pub trait Zero {
    fn zero() -> Self;
}

macro_rules! zero_impl {
    ($t:ty, $v:expr) => {
        impl Zero for $t {
            #[inline]
            fn zero() -> $t {
                $v
            }
        }
    };
}

zero_impl!(f32, 0.0);
zero_impl!(f64, 0.0);

pub type Array<S, D> = ArrayData<Vec<S>, D>;

pub type Matrix<S> = Array<S, [usize; 2]>;

pub type Matrix3D<S> = Array<S, [usize; 3]>;

pub struct OwnedPtr<P> {
    ptr: *const P,
    len: usize,
    cap: usize,
}

pub struct ArrayData<S, D>
where
    D: Dimension,
{
    data: S,
    dim: D,
}

impl<S, D> Array<S, D>
where
    D: Dimension,
{
    fn new() {}

    fn from_elem(d: D, s: S) -> Self
    where
        S: Clone,
    {
        let s = vec![s; d.size()];
        Self { data: s, dim: d }
    }

    fn zeros(d: D) -> Self
    where
        S: Clone + Zero,
    {
        Self::from_elem(d, S::zero())
    }
}

pub trait Dimension {
    fn slice(&self) -> &[usize];

    fn size(&self) -> usize;

    fn dim(&self) -> usize;
}

macro_rules! dimension_impl_array {
    ($n:expr) => {
        impl Dimension for [usize; $n] {
            fn slice(&self) -> &[usize] {
                self
            }

            fn size(&self) -> usize {
                self.iter().fold(1, |s, &a| s * a as usize)
            }

            fn dim(&self) -> usize {
                self.len()
            }
        }
    };
}

dimension_impl_array!(1);
dimension_impl_array!(2);
dimension_impl_array!(3);
dimension_impl_array!(4);
dimension_impl_array!(5);
dimension_impl_array!(6);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuple() {
        let x = (1, 2, 3);
        // let a = Array::<f64, _>::zeros((1, 2));
        let x = Matrix::<f64>::zeros([2, 3]);
        println!("{:?}", x.data);
    }
}
