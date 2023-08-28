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

#[derive(Clone, Debug)]
pub struct Shape(Box<[usize]>);

impl Shape {
    pub fn from(v: Vec<usize>) -> Shape {
        Shape(v.into_boxed_slice())
    }

    pub fn from_slice(v: &[usize]) -> Shape {
        Shape::from(v.to_vec())
    }
}

impl Shape {
    pub fn select_axis(&self, a: Axis) -> Shape {
        let mut dst = vec![0; self.dim() - 1];
        let src = self.as_slice();
        dst[..a.index()].copy_from_slice(&src[..a.index()]);
        dst[a.index()..].copy_from_slice(&src[a.index() + 1..]);
        Shape::from(dst)
    }

    pub fn as_slice(&self) -> &[usize] {
        self.0.as_ref()
    }

    pub fn as_slice_mut(&mut self) -> &mut [usize] {
        self.0.as_mut()
    }

    pub fn size(&self) -> usize {
        self.as_slice().iter().fold(1, |s, &a| s * a as usize)
    }

    pub fn zero() -> Self {
        Shape::from(Vec::new())
    }

    // [a, b, c] => strides [b * c, c, 1]
    pub fn strides(&self) -> Shape {
        let mut x = vec![0; self.dim()];
        let s = self.as_slice().iter().rev();
        let mut prod = 1;
        let mut temp = 1;
        for (m, dim) in x.iter_mut().rev().zip(s) {
            prod *= temp;
            *m = prod;
            temp = *dim;
        }
        Shape::from(x)
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub(crate) fn first_index(&self) -> Option<Self> {
        for ax in self.as_slice().iter() {
            if *ax == 0 {
                return None;
            }
        }
        Some(Self::zero())
    }

    #[inline]
    pub(crate) fn next_for(&self, index: Self) -> Option<Self> {
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
    pub(crate) fn stride_offset(index: &Self, strides: &Self) -> isize {
        let mut offset = 0;
        for (&i, &s) in index.as_slice().iter().zip(strides.as_slice().iter()) {
            offset += stride_offset(i, s);
        }
        offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] //[3, 2]
    fn test_select_axis() {
        // let d = [4usize, 3, 2, 1];
        // let d2 = d.select_axis(Axis(0));
        // println!("{:?}", d2);
    }

    #[test]
    fn test_dyn_dim_select_axis() {
        let d = Shape::from(vec![4usize, 3, 2, 1]);
        let d2 = d.select_axis(Axis(0));
        println!("{:?}", d2.as_slice());
    }
}
