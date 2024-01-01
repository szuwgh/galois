use crate::{GError, GResult};
use std::iter;
pub(crate) fn zip<I, J>(i: I, j: J) -> iter::Zip<I::IntoIter, J::IntoIter>
where
    I: IntoIterator,
    J: IntoIterator,
{
    i.into_iter().zip(j)
}

#[derive(Copy, Clone)]
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
pub struct Dim {
    pub(crate) s: Shape,
    pub(crate) stride: Box<[usize]>,
}

impl Dim {
    pub fn shape(&self) -> &Shape {
        &self.s
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn transpose(&self, d1: usize, d2: usize) -> GResult<Dim> {
        let rank = self.s.size();
        if rank <= d1 || rank <= d2 {
            return Err(GError::UnexpectedNumberOfDims {
                expected: usize::max(d1, d1),
                got: rank,
                shape: self.shape().clone(),
            });
        }
        let mut stride = self.stride().to_vec();
        let mut dims = self.shape().as_slice().to_vec();
        dims.swap(d1, d2);
        stride.swap(d1, d2);
        Ok(Self {
            s: Shape::from_vec(dims),
            stride: stride.into_boxed_slice(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Shape(pub(crate) Box<[usize]>);

impl Shape {
    pub fn from_vec(v: Vec<usize>) -> Shape {
        Shape(v.into_boxed_slice())
    }

    pub fn from_array<const N: usize>(v: [usize; N]) -> Shape {
        Shape::from_vec(v.to_vec())
    }

    pub fn from_slice(v: &[usize]) -> Shape {
        Shape::from_vec(v.to_vec())
    }

    pub fn dims4(&self) -> (usize, usize, usize, usize) {
        dims4(&self.0)
    }

    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }
}

pub fn dims4(s: &[usize]) -> (usize, usize, usize, usize) {
    assert!(s.len() >= 4);
    (s[0], s[1], s[2], s[3])
}

impl PartialEq<Shape> for Shape {
    fn eq(&self, other: &Shape) -> bool {
        // if self.dim() != other.dim() {
        //     return false;
        // }
        return self.as_slice() == other.as_slice();
    }
}

pub fn select_axis(src: &[usize], a: Axis) -> Vec<usize> {
    let mut dst = vec![0; src.len() - 1];
    dst[..a.index()].copy_from_slice(&src[..a.index()]);
    dst[a.index()..].copy_from_slice(&src[a.index() + 1..]);
    dst
}

impl Shape {
    pub fn select_axis(&self, a: Axis) -> Shape {
        let dst = select_axis(self.as_slice(), a);
        Shape::from_vec(dst)
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

    pub fn zero(s: usize) -> Self {
        Shape::from_vec(vec![0usize; s])
    }

    // [a, b, c] => strides [b * c, c, 1]
    pub fn strides(&self) -> Box<[usize]> {
        let mut x = vec![0; self.dim()];
        let s = self.as_slice().iter().rev();
        let mut prod = 1;
        let mut temp = 1;
        for (m, dim) in x.iter_mut().rev().zip(s) {
            prod *= temp;
            *m = prod;
            temp = *dim;
        }
        x.into_boxed_slice()
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub(crate) fn first_index(&self) -> Option<Self> {
        let slice = self.as_slice();
        for ax in slice.iter() {
            if *ax == 0 {
                return None;
            }
        }
        Some(Self::zero(slice.len()))
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
    pub(crate) fn stride_offset(index: &Self, strides: &[usize]) -> isize {
        let mut offset = 0;
        for (&i, &s) in index.as_slice().iter().zip(strides.iter()) {
            offset += stride_offset(i, s);
        }
        offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] //[3, 2]
    fn test_tensor_transpose() {
        // let d = [4usize, 3, 2, 1];
        // let d2 = d.select_axis(Axis(0));
        // println!("{:?}", d2);
    }

    #[test]
    fn test_dyn_dim_select_axis() {
        let d = Shape::from_vec(vec![4usize, 3, 2, 1]);
        let d2 = d.select_axis(Axis(0));
        println!("{:?}", d2.as_slice());
    }
}
