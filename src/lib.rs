mod dimension;
mod method;
extern crate alloc;
use crate::dimension::{Axis, Dimension};
use core::ptr::{self, NonNull};
use rawpointer::PointerExt;
use std::fmt;
use std::mem::ManuallyDrop;
use std::vec::from_elem;

#[macro_export]
macro_rules! tensor {
    ($([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::Cube::from(vec![$([$([$($x,)*],)*],)*])
    }};
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
        $crate::Matrix::from(vec![$([$($x,)*],)*])
    }};
    ($($x:expr),* $(,)*) => {{
        $crate::Array::from(vec![$($x,)*])
    }};
}

#[macro_export]
macro_rules! arr {
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

//一维数组
pub type Array<A> = Tensor<A, [usize; 1]>;

//二维数组
pub type Matrix<A> = Tensor<A, [usize; 2]>;

//三维数组
pub type Cube<A> = Tensor<A, [usize; 3]>;

//real only nd Tensor
pub type TensorView<'a, A, D> = TensorData<ViewSlice<A>, D>;

pub fn mat<A: Clone, const N: usize>(xs: Vec<[A; N]>) -> Matrix<A> {
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

#[cold]
#[cfg(not(no_global_oom_handling))]
fn capacity_overflow() -> ! {
    panic!("capacity overflow");
}

pub(crate) struct SetLenOnDrop<'a> {
    len: &'a mut usize,
    local_len: usize,
}

impl<'a> SetLenOnDrop<'a> {
    #[inline]
    pub(crate) fn new(len: &'a mut usize) -> Self {
        SetLenOnDrop {
            local_len: *len,
            len,
        }
    }

    #[inline]
    pub(crate) fn increment_len(&mut self, increment: usize) {
        self.local_len += increment;
    }

    #[inline]
    pub(crate) fn current_len(&self) -> usize {
        self.local_len
    }
}

impl Drop for SetLenOnDrop<'_> {
    #[inline]
    fn drop(&mut self) {
        *self.len = self.local_len;
    }
}

pub struct RawTen<P> {
    ptr: NonNull<P>,
    len: usize,
    cap: usize,
}

impl<P> RawTen<P> {
    fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                ptr: NonNull::<P>::dangling(),
                len: 0,
                cap: 0,
            };
        }
        use alloc::alloc::{alloc, handle_alloc_error, Layout};
        let layout = match Layout::array::<P>(capacity) {
            Ok(layout) => layout,
            Err(_) => capacity_overflow(),
        };
        let ptr = if layout.size() == 0 {
            core::ptr::NonNull::<P>::dangling()
        } else {
            let ptr = unsafe { alloc(layout) } as *mut P;
            if ptr.is_null() {
                handle_alloc_error(layout)
            } else {
                unsafe { NonNull::<P>::new_unchecked(ptr) }
            }
        };
        Self {
            ptr: ptr,
            len: 0,
            cap: capacity,
        }
    }

    fn as_slice(&self) -> &[P] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const P, self.len) }
    }

    fn take_as_vec(&mut self) -> Vec<P> {
        let capacity = self.cap;
        let len = self.len;
        self.len = 0;
        self.cap = 0;
        unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), len, capacity) }
    }

    pub(crate) fn as_ptr(&self) -> *const P {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_ptr_mut(&self) -> *mut P {
        self.ptr.as_ptr()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn fill(&mut self, elem: P, n: usize)
    where
        P: Clone,
    {
        unsafe {
            let mut ptr = self.as_ptr_mut().add(self.len());
            let mut local_len = SetLenOnDrop::new(&mut self.len);

            // Write all elements except the last one
            for _ in 1..n {
                ptr::write(ptr, elem.clone());
                ptr = ptr.add(1);
                // Increment the length in every step in case next() panics
                local_len.increment_len(1);
            }

            if n > 0 {
                // We can write the last element directly without cloning needlessly
                ptr::write(ptr, elem);
                local_len.increment_len(1);
            }

            // len set by scope guard
        }
    }
}

// pub fn new(row_capacity: usize, col_capacity: usize) -> Self {
//     if core::mem::size_of::<T>() == 0 {
//         Self {
//             ptr: NonNull::<T>::dangling(),
//             row_capacity,
//             col_capacity,
//         }
//     } else {
//         let cap = row_capacity
//             .checked_mul(col_capacity)
//             .unwrap_or_else(capacity_overflow);
//         let cap_bytes = cap
//             .checked_mul(core::mem::size_of::<T>())
//             .unwrap_or_else(capacity_overflow);
//         if cap_bytes > isize::MAX as usize {
//             capacity_overflow::<()>();
//         }

//         use alloc::alloc::{alloc, handle_alloc_error, Layout};

//         let layout = Layout::from_size_align(cap_bytes, align_for::<T>())
//             .ok()
//             .unwrap_or_else(capacity_overflow);

//         let ptr = if layout.size() == 0 {
//             core::ptr::NonNull::<T>::dangling()
//         } else {
//             // SAFETY: we checked that layout has non zero size
//             let ptr = unsafe { alloc(layout) } as *mut T;
//             if ptr.is_null() {
//                 handle_alloc_error(layout)
//             } else {
//                 // SAFETY: we checked that the pointer is not null
//                 unsafe { NonNull::<T>::new_unchecked(ptr) }
//             }
//         };

//         Self {
//             ptr,
//             row_capacity,
//             col_capacity,
//         }
//     }
// }

// fn allocate_in(capacity: usize, init: AllocInit, alloc: A) -> Self {
//     // Don't allocate here because `Drop` will not deallocate when `capacity` is 0.
//     if T::IS_ZST || capacity == 0 {
//         Self::new_in(alloc)
//     } else {
//         // We avoid `unwrap_or_else` here because it bloats the amount of
//         // LLVM IR generated.
//         let layout = match Layout::array::<T>(capacity) {
//             Ok(layout) => layout,
//             Err(_) => capacity_overflow(),
//         };
//         match alloc_guard(layout.size()) {
//             Ok(_) => {}
//             Err(_) => capacity_overflow(),
//         }
//         let result = match init {
//             AllocInit::Uninitialized => alloc.allocate(layout),
//             AllocInit::Zeroed => alloc.allocate_zeroed(layout),
//         };
//         let ptr = match result {
//             Ok(ptr) => ptr,
//             Err(_) => handle_alloc_error(layout),
//         };

//         // Allocators currently return a `NonNull<[u8]>` whose length
//         // matches the size requested. If that ever changes, the capacity
//         // here should change to `ptr.len() / mem::size_of::<T>()`.
//         Self {
//             ptr: unsafe { Unique::new_unchecked(ptr.cast().as_ptr()) },
//             cap: capacity,
//             alloc,
//         }
//     }
//}

pub struct OwnerSlice<P> {
    ptr: NonNull<P>,
    len: usize,
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
    pub(crate) fn from(v: Vec<P>) -> Self {
        let mut v = ManuallyDrop::new(v);
        let len = v.len();
        // let cap = v.capacity();
        let ptr = unsafe { NonNull::new_unchecked(v.as_mut_ptr()) };
        Self { ptr, len }
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

        Self::from_vec(v, d)
    }

    pub fn from_vec(v: Vec<A>, d: D) -> Self {
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
        format_tensor(self.view(), f)
    }
}

fn format_tensor<A: fmt::Debug, D>(
    mut tensor: TensorView<'_, A, D>,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result
where
    D: Dimension,
{
    match tensor.shape() {
        &[] => {}
        &[len] => {
            let v = tensor.slice();
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
                format_tensor(tensor.axis_index(Axis::Row, i), f)?;
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
    fn test_rawten() {
        let mut t = RawTen::<u32>::with_capacity(10);
        t.fill(3, 10);
        println!("t:{:?}", t.as_slice());
    }

    #[test]
    fn test_fmt() {
        // let x = (1, 2, 3);
        // let b = Vec::new();
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
