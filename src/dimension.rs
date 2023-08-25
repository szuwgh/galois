use std::iter;

pub(crate) fn zip<I, J>(i: I, j: J) -> iter::Zip<I::IntoIter, J::IntoIter>
where
    I: IntoIterator,
    J: IntoIterator,
{
    i.into_iter().zip(j)
}

pub struct Axis(pub usize);

impl Axis {
    pub fn index(&self) -> usize {
        self.0
    }
}

#[inline(always)]
pub fn stride_offset(n: usize, stride: usize) -> isize {
    (n * stride) as isize
}

pub trait Dimension: Clone {
    fn as_slice(&self) -> &[usize];

    fn as_slice_mut(&mut self) -> &mut [usize];

    fn size(&self) -> usize;

    fn dim(&self) -> usize;

    fn strides(&self) -> Self;

    fn zero() -> Self;

    #[inline]
    fn first_index(&self) -> Option<Self> {
        for ax in self.as_slice().iter() {
            if *ax == 0 {
                return None;
            }
        }
        Some(Self::zero())
    }

    #[inline]
    fn next_for(&self, index: Self) -> Option<Self> {
        let mut index = index;
        let mut done = false;
        for (&dim, ix) in zip(self.as_slice(), index.as_slice_mut()).rev() {
            *ix += 1;
            if *ix == dim {
                *ix = 0;
            } else {
                done = true;
                break;
            }
        }
        if done {
            Some(index)
        } else {
            None
        }
    }

    #[inline]
    fn stride_offset(index: &Self, strides: &Self) -> isize {
        let mut offset = 0;
        for (&i, &s) in index.as_slice().iter().zip(strides.as_slice().iter()) {
            offset += stride_offset(i, s);
        }
        offset
    }

    type AxisDimension: Dimension;
    fn select_axis(&self, a: Axis) -> Self::AxisDimension;
}

impl Dimension for [usize; 1] {
    type AxisDimension = [usize; 1];
    fn select_axis(&self, a: Axis) -> Self::AxisDimension {
        [0; 1]
    }

    fn as_slice(&self) -> &[usize] {
        self
    }

    fn as_slice_mut(&mut self) -> &mut [usize] {
        self
    }

    fn zero() -> Self {
        [0usize; 1]
    }

    fn size(&self) -> usize {
        self.iter().fold(1, |s, &a| s * a as usize)
    }

    // [a, b, c] => strides [b * c, c, 1]
    fn strides(&self) -> [usize; 1] {
        let mut x = [0; 1];
        let s = self.as_slice().iter().rev();
        let mut prod = 1;
        let mut temp = 1;
        for (m, dim) in x.iter_mut().rev().zip(s) {
            prod *= temp;
            *m = prod;
            temp = *dim;
        }
        x
    }

    fn dim(&self) -> usize {
        self.len()
    }
}

#[derive(Clone)]
pub struct DynDim {
    dim: Box<[usize]>,
}

impl DynDim {
    fn from(v: Vec<usize>) -> DynDim {
        Self {
            dim: v.into_boxed_slice(),
        }
    }
}

impl Dimension for DynDim {
    type AxisDimension = DynDim;
    fn select_axis(&self, a: Axis) -> Self::AxisDimension {
        let mut dst = vec![0; self.dim() - 1];
        let src = self.as_slice();
        dst[..a.index()].copy_from_slice(&src[..a.index()]);
        dst[a.index()..].copy_from_slice(&src[a.index() + 1..]);
        DynDim::from(dst)
    }

    fn as_slice(&self) -> &[usize] {
        &self.dim
    }

    fn as_slice_mut(&mut self) -> &mut [usize] {
        &mut self.dim
    }

    fn size(&self) -> usize {
        self.as_slice().iter().fold(1, |s, &a| s * a as usize)
    }

    fn zero() -> Self {
        DynDim::from(Vec::new())
    }

    // [a, b, c] => strides [b * c, c, 1]
    fn strides(&self) -> DynDim {
        let mut x = vec![0; self.dim()];
        let s = self.as_slice().iter().rev();
        let mut prod = 1;
        let mut temp = 1;
        for (m, dim) in x.iter_mut().rev().zip(s) {
            prod *= temp;
            *m = prod;
            temp = *dim;
        }
        DynDim::from(x)
    }

    fn dim(&self) -> usize {
        self.dim.len()
    }
}

macro_rules! dimension_impl_array {
    ($n:expr) => {
        impl Dimension for [usize; $n] {
            type AxisDimension = [usize; $n - 1];
            fn select_axis(&self, a: Axis) -> Self::AxisDimension {
                let mut dst = [0; $n - 1];
                let src = self.as_slice();
                dst[..a.index()].copy_from_slice(&src[..a.index()]);
                dst[a.index()..].copy_from_slice(&src[a.index() + 1..]);
                dst
            }

            fn as_slice(&self) -> &[usize] {
                self
            }

            fn as_slice_mut(&mut self) -> &mut [usize] {
                self
            }

            fn size(&self) -> usize {
                self.iter().fold(1, |s, &a| s * a as usize)
            }

            fn zero() -> Self {
                [0usize; $n]
            }

            // [a, b, c] => strides [b * c, c, 1]
            fn strides(&self) -> [usize; $n] {
                let mut x = [0; $n];
                let s = self.as_slice().iter().rev();
                let mut prod = 1;
                let mut temp = 1;
                for (m, dim) in x.iter_mut().rev().zip(s) {
                    prod *= temp;
                    *m = prod;
                    temp = *dim;
                }
                x
            }

            fn dim(&self) -> usize {
                self.len()
            }
        }
    };
}

dimension_impl_array!(2);
dimension_impl_array!(3);
dimension_impl_array!(4);
dimension_impl_array!(5);
dimension_impl_array!(6);
dimension_impl_array!(7);
dimension_impl_array!(8);
dimension_impl_array!(9);
dimension_impl_array!(10);
dimension_impl_array!(11);

#[cfg(test)]
mod tests {
    use super::*;

    #[test] //[3, 2]
    fn test_select_axis() {
        let d = [4usize, 3, 2, 1];
        let d2 = d.select_axis(Axis(0));
        println!("{:?}", d2);
    }

    #[test]
    fn test_dyn_dim_select_axis() {
        let d = DynDim::from(vec![4usize, 3, 2, 1]);
        let d2 = d.select_axis(Axis(0));
        println!("{:?}", d2.as_slice());
    }
}
