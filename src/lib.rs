mod dimension;
mod method;

use crate::dimension::{Axis, Dimension};

use rawpointer::PointerExt;
use std::fmt;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

#[macro_export]
macro_rules! tensor {
    ($([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::Matrix3D::from(vec![$([$([$($x,)*],)*],)*])
    }};
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
        $crate::Matrix::from(vec![$([$($x,)*],)*])
    }};
    ($($x:expr),* $(,)*) => {{
        $crate::Array::from(vec![$($x,)*])
    }};
}

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

pub type Tensor<A, D> = TensorData<OwnerSlice<A>, D>;

//real only nd Tensor
pub type TensorView<'a, A, D> = TensorData<ViewSlice<A>, D>;

pub type Array<A> = Tensor<A, [usize; 1]>;

pub type Matrix<A> = Tensor<A, [usize; 2]>;

pub type Matrix3D<A> = Tensor<A, [usize; 3]>;

pub fn arr2<A: Clone, const N: usize>(xs: Vec<[A; N]>) -> Matrix<A> {
    let dim = [xs.len(), N];
    todo!();
    //Tensor::from_vec(dim, xs)
    //Array2::from(xs.to_vec())
}

pub trait BaseData {
    type Elem;
    fn new(elements: Vec<Self::Elem>) -> Self;

    fn as_ptr(&self) -> NonNull<Self::Elem>;

    fn as_slice(&self) -> &[Self::Elem];

    fn len(&self) -> usize;
}

impl<'a, A, D> TensorView<'a, A, D>
where
    D: Dimension,
{
    fn new(v: ViewSlice<A>, d: D) -> TensorView<'a, A, D>
    where
        D: Dimension,
    {
        let stride = d.strides();
        Self {
            data: v,
            dim: d,
            stride: stride,
        }
    }
}

pub struct ViewSlice<P> {
    ptr: NonNull<P>,
    len: usize,
}

impl<P> BaseData for ViewSlice<P> {
    type Elem = P;

    fn new(elements: Vec<Self::Elem>) -> Self {
        todo!()
    }

    fn as_ptr(&self) -> NonNull<Self::Elem> {
        self.ptr
    }

    fn as_slice(&self) -> &[P] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const P, self.len) }
    }

    fn len(&self) -> usize {
        self.len
    }
}

pub struct OwnerSlice<P> {
    ptr: NonNull<P>,
    len: usize,
    cap: usize,
}

impl<P> BaseData for OwnerSlice<P> {
    type Elem = P;

    fn new(elements: Vec<Self::Elem>) -> Self {
        OwnerSlice::from(elements)
    }

    fn as_ptr(&self) -> NonNull<Self::Elem> {
        self.ptr
    }

    fn as_slice(&self) -> &[P] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const P, self.len) }
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl<P> OwnerSlice<P> {
    pub(crate) fn from(mut v: Vec<P>) -> Self {
        let mut v = ManuallyDrop::new(v);
        let len = v.len();
        let cap = v.capacity();
        let ptr = unsafe { NonNull::new_unchecked(v.as_mut_ptr()) };
        Self { ptr, len, cap }
    }

    pub(crate) fn as_slice_mut(&self) -> &mut [P] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

pub struct TensorData<S, D>
where
    D: Dimension,
{
    data: S,
    dim: D,
    stride: D,
}

impl<A, S, D> TensorData<S, D>
where
    S: BaseData<Elem = A>,
    D: Dimension,
{
    fn axis_index(&mut self, a: Axis, index: usize) -> TensorView<'_, A, D::AxisDimension>
    where
        D: Dimension,
    {
        let stride = self.stride.slice()[a.index()];
        let offset = index * stride;
        // println!("offset:{}", offset);
        let axis_dim = self.dim.select_axis(a);
        TensorView::new(
            ViewSlice {
                ptr: unsafe { self.data.as_ptr().offset(offset as isize) },
                len: self.data.len() - offset,
            },
            axis_dim,
        )
    }

    pub fn view(&self) -> TensorView<'_, A, D>
    where
        D: Dimension,
    {
        TensorView::new(
            ViewSlice {
                ptr: self.data.as_ptr(),
                len: self.data.len(),
            },
            self.dim.clone(),
        )
    }

    pub fn from_elem(s: A, d: D) -> Self
    where
        A: Clone,
    {
        //申请内存
        let v = vec![s; d.size()];

        Self::from_vec(d, v)
    }

    pub fn from_vec(d: D, v: Vec<A>) -> Self {
        let stride = d.strides();
        Self {
            data: BaseData::new(v),
            dim: d,
            stride: stride,
        }
    }

    pub fn zeros(d: D) -> Self
    where
        A: Clone + Zero,
    {
        Self::from_elem(A::zero(), d)
    }

    pub fn slice(&self) -> &[A] {
        self.data.as_slice()
    }

    pub fn shape(&self) -> &[usize] {
        self.dim.slice()
    }

    pub fn size(&self) -> usize {
        self.dim.size()
    }

    pub fn dim(&self) -> usize {
        self.dim.dim()
    }
}

impl<A: fmt::Debug, S, D> fmt::Debug for TensorData<S, D>
where
    S: BaseData<Elem = A>,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_Tensor(self.view(), f)
    }
}

fn format_Tensor<A: fmt::Debug, D>(
    mut Tensor: TensorView<'_, A, D>,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result
where
    D: Dimension,
{
    match Tensor.shape() {
        &[] => {}
        &[len] => {
            let v = Tensor.slice();
            f.write_str("[")?;
            f.write_str(&format!("{:?}", v[0]))?;
            for i in 1..len {
                f.write_str(",")?;
                f.write_str(&format!(" {:?}", v[i]))?;
            }
            f.write_str("]")?;
        }
        shape => {
            f.write_str("\n")?;
            f.write_str("[")?;
            for i in 0..shape[0] {
                format_Tensor(Tensor.axis_index(Axis::Row, i), f)?;
            }
            f.write_str("]")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmt() {
        // let x = (1, 2, 3);
        let a = Tensor::<f64, _>::zeros([4usize, 3, 2]);
        let v = a.slice();
        println!("v:{:?}", v);
        // let x = Matrix::<f64>::zeros([2, 3]);
        // println!("xxx");
        // shape=[4, 3, 2, 2], strides=[12, 4, 2, 1]
        //let x: [usize; 4] = [4, 3, 2, 2];
        //let z = x.strides();
        println!("{:?}", a);
    }
}
