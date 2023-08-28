mod broadcast;
mod error;
mod method;
mod shape;
mod zip;

extern crate alloc;
use crate::shape::{Axis, Shape};
use core::ptr::{self, NonNull};
use rawpointer::PointerExt;
use std::fmt;
use std::mem::forget;

// #[macro_export]
// macro_rules! tensor {
//     ($([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
//         $crate::Cube::from(vec![$([$([$($x,)*],)*],)*])
//     }};
//     ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
//         $crate::Matrix::from(vec![$([$($x,)*],)*])
//     }};
//     ($($x:expr),* $(,)*) => {{
//         $crate::Array::from(vec![$($x,)*])
//     }};
// }

// #[macro_export]
// macro_rules! arr {
//     ($($x:expr),* $(,)*) => {{
//         $crate::Array::from(vec![$($x,)*])
//     }};
// }

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

pub fn arr<A: Clone>(xs: &[A]) -> Tensor<A> {
    Tensor::arr(xs.to_vec())
}

pub fn mat<A: Clone, const N: usize>(xs: &[[A; N]]) -> Tensor<A> {
    Tensor::mat(xs.to_vec())
}

pub fn cube<A: Clone, const N: usize, const M: usize>(xs: &[[[A; N]; M]]) -> Tensor<A> {
    Tensor::cube(xs.to_vec())
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

#[derive(Clone)]
struct RawTen<P> {
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

    fn as_slice_mut(&mut self) -> &mut [P] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut P, self.len) }
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

    fn from_raw_parts(ptr: *mut P, length: usize, capacity: usize) -> Self {
        unsafe {
            Self {
                ptr: NonNull::new_unchecked(ptr),
                len: length,
                cap: capacity,
            }
        }
    }

    fn fill(&mut self, elem: P, n: usize)
    where
        P: Clone,
    {
        unsafe {
            let mut ptr = self.as_ptr_mut().add(self.len());
            let mut local_len = SetLenOnDrop::new(&mut self.len);
            for _ in 1..n {
                ptr::write(ptr, elem.clone());
                ptr = ptr.add(1);

                local_len.increment_len(1);
            }
            if n > 0 {
                ptr::write(ptr, elem);
                local_len.increment_len(1);
            }
        }
    }
}

use std::marker::PhantomData;

pub struct TensorIter<'a, A> {
    ptr: NonNull<A>,
    dim: Shape,
    strides: Shape,
    index: Option<Shape>,
    _marker: PhantomData<&'a A>,
}

impl<'a, A> Iterator for TensorIter<'a, A> {
    type Item = &'a A;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        let offset = Shape::stride_offset(&index, &self.strides);
        self.index = self.dim.next_for(index);
        let ptr = unsafe { self.ptr.offset(offset) }.as_ptr();
        unsafe { Some(&*ptr) }
    }
}

#[derive(Clone)]
pub struct TenRef<A> {
    ptr: NonNull<A>,
    len: usize,
    cap: usize,
    dim: Shape,
    stride: Shape,
}

impl<A> TenRef<A> {
    ///
    ///```
    /// let a = arr2(&[[1., 2. ],    // ... axis 0, row 0
    ///                [3., 4. ],    // --- axis 0, row 1
    ///                [5., 6. ]]);  // ... axis 0, row 2
    /// //               .   \
    /// //                .   axis 1, column 1
    /// //                 axis 1, column 0
    /// ```
    ///```
    ///
    ///
    /// data =[[a00, a01],
    ///        [a10, a11]]
    /// 所以axis=0时，沿着第0个下标变化的方向进行操作，也就是a00->a10, a01->a11，也就是纵坐标的方向，axis=1时也类似。
    ///
    ///
    /// array([[[[3, 5, 5, 0],
    ///     [0, 1, 2, 4],
    ///     [0, 5, 0, 5]],    # D0000  ->   D0023
    ///
    ///     [[5, 5, 0, 0],
    ///      [2, 1, 5, 0],
    ///      [1, 0, 0, 1]]],  # D0100  ->   D0123

    ///   [[[0, 5, 1, 2],
    ///     [4, 4, 2, 2],
    ///     [3, 5, 0, 1]],    # D1000  ->   D1023

    ///    [[5, 1, 2, 1],
    ///     [2, 2, 3, 5],
    ///     [5, 3, 3, 3]]],   # D1100  ->   D1123

    ///   [[[2, 4, 1, 4],
    ///     [1, 4, 1, 4],
    ///     [4, 5, 0, 2]],    # D2000  ->   D2023

    ///    [[2, 5, 5, 1],
    ///     [5, 3, 0, 2],
    ///     [4, 0, 1, 3]]],   # D2100  ->   D2123

    ///   [[[1, 3, 4, 5],
    ///     [0, 2, 5, 4],
    ///     [2, 3, 5, 3]],    # D3000  ->   D3023

    ///    [[2, 2, 2, 2],
    ///     [3, 2, 1, 3],
    /// 所以axis=0时，沿着第0个下标变化的方向进行操作，也就是a00->a10, a01->a11，也就是纵坐标的方向，axis=1时也类似。
    /// 所以axis=0时，沿着第0个下标变化的方向进行操作
    /// 四维的求sum的例子
    /// 当axis=0时，numpy验证第0维的方向来求和，也就是第一个元素值=D0000+D1000+D2000+D3000=3+0+2+1=6,第二个元素=D0001+D1001+D2001+D3001=5+5+4+3=17
    /// 当axis=0时，numpy验证第0维的方向来求和，也就是第一个元素值=D0000+D1000+D2000+D3000=3+0+2+1=6,第二个元素=D0001+D1001+D2001+D3001=5+5+4+3=17，同理可得最后的结果如下：
    /// 当axis=3时，numpy验证第3维的方向来求和，也就是第一个元素值=D0000+D0001+D0002+D0003=3+5+5+0=13,第二个元素=D0010+D0011+D0012+D0013=0+1+2+4=7，同理可得最后的结果如下
    /// ```
    ///
    ///
    fn axis_index(&self, a: Axis, index: usize) -> TenRef<A> {
        let stride = self.stride.as_slice()[a.index()];
        let offset = index * stride;
        let axis_dim = self.dim.select_axis(a);
        let s = axis_dim.strides();
        TenRef {
            ptr: unsafe { self.ptr.offset(offset as isize) },
            len: self.len - offset,
            cap: self.cap,
            dim: axis_dim,
            stride: s,
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.dim.as_slice()
    }

    pub fn size(&self) -> usize {
        self.dim.size()
    }

    pub fn dim(&self) -> usize {
        self.dim.dim()
    }

    pub fn iter(&self) -> TensorIter<'_, A> {
        TensorIter {
            ptr: self.ptr,
            dim: self.dim.clone(),
            strides: self.stride.clone(),
            index: self.dim.first_index(),
            _marker: PhantomData,
        }
    }

    fn as_slice(&self) -> &[A] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const A, self.len) }
    }

    fn as_slice_mut(&mut self) -> &mut [A] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut A, self.len) }
    }

    pub fn broadcast<E>(&self, d: E) {}

    pub fn zip<F>(&mut self, rhs: &TenRef<A>, mut f: F)
    where
        F: FnMut(&mut A, &A),
    {

        // let slicel = self.as_slice_mut();
        // let slicer = rhs.as_slice();
        // for (s, r) in slicel.iter_mut().zip(slicer) {
        //     f(s, r);
        // }
    }
}

#[derive(Clone)]
pub struct Tensor<A> {
    data: RawTen<A>,
    dim: Shape,
    stride: Shape,
}

impl<A> Tensor<A> {
    fn arr(mut xs: Vec<A>) -> Self {
        let dim = [xs.len()];
        let shape = Shape::from_slice(&dim);
        let ptr = xs.as_mut_ptr();
        let cap = shape.size();
        forget(xs);
        let data = RawTen::from_raw_parts(ptr as *mut A, cap, cap);
        Tensor::from_raw_data(data, shape)
    }

    fn mat<const N: usize>(mut xs: Vec<[A; N]>) -> Self {
        let dim = [xs.len(), N];
        let shape = Shape::from_slice(&dim);
        let ptr = xs.as_mut_ptr();
        let cap = shape.size();
        forget(xs);
        let data = RawTen::from_raw_parts(ptr as *mut A, cap, cap);
        Tensor::from_raw_data(data, shape)
    }

    fn cube<const M: usize, const N: usize>(mut xs: Vec<[[A; N]; M]>) -> Self {
        let dim = [xs.len(), N, M];
        let shape = Shape::from_slice(&dim);
        let ptr = xs.as_mut_ptr();
        let cap = shape.size();
        forget(xs);
        let data = RawTen::from_raw_parts(ptr as *mut A, cap, cap);
        Tensor::from_raw_data(data, shape)
    }

    fn from_raw_data(data: RawTen<A>, s: Shape) -> Self {
        let stride = s.strides();
        Self {
            data: data,
            dim: s,
            stride: stride,
        }
    }

    pub fn as_ref<'a>(&'a self) -> TenRef<A> {
        TenRef {
            ptr: self.data.ptr,
            len: self.data.len,
            cap: self.data.cap,
            dim: self.dim.clone(),
            stride: self.stride.clone(),
        }
    }

    pub fn from_elem(a: A, s: Shape) -> Self
    where
        A: Clone,
    {
        let mut v = RawTen::with_capacity(s.size());
        v.fill(a, s.size());
        let stride = s.strides();
        Self {
            data: v,
            dim: s,
            stride: stride,
        }
    }

    pub fn with_shape(v: Vec<A>, d: Shape) -> Self {
        Self::from_vec(v, d)
    }

    pub fn from_vec(v: Vec<A>, d: Shape) -> Self {
        let stride = d.strides();
        Self {
            data: RawTen::from_raw_parts(v.as_ptr() as *mut A, v.len(), v.capacity()),
            dim: d,
            stride: stride,
        }
    }

    pub fn as_slice(&self) -> &[A] {
        self.data.as_slice()
    }

    pub fn as_slice_mut(&mut self) -> &mut [A] {
        self.data.as_slice_mut()
    }

    pub fn zeros(d: Shape) -> Self
    where
        A: Clone + Zero,
    {
        Self::from_elem(A::zero(), d)
    }

    pub fn shape(&self) -> &[usize] {
        self.dim.as_slice()
    }

    pub fn size(&self) -> usize {
        self.dim.size()
    }

    pub fn dim(&self) -> usize {
        self.dim.dim()
    }
}

impl<T> Drop for RawTen<T> {
    fn drop(&mut self) {
        use alloc::alloc::{dealloc, Layout};
        let alloc_size = self.cap * core::mem::size_of::<T>();
        if alloc_size != 0 {
            unsafe {
                dealloc(
                    self.ptr.as_ptr() as *mut u8,
                    Layout::from_size_align_unchecked(alloc_size, core::mem::align_of::<T>()),
                );
            }
        }
    }
}

impl<A: fmt::Debug> fmt::Debug for TenRef<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor(self, 0, f)
    }
}

impl<A: fmt::Debug> fmt::Debug for Tensor<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor(&self.as_ref(), 0, f)
    }
}

fn format_tensor<A: fmt::Debug>(
    tensor: &TenRef<A>,
    depth: usize,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    match tensor.shape() {
        &[] => {}
        &[len] => {
            let v = tensor.as_slice();
            f.write_str("[")?;
            f.write_str(&format!("{:?}", v[0]))?;
            for i in 1..len {
                f.write_str(",")?;
                f.write_str(&format!(" {:?}", v[i]))?;
            }
            f.write_str("]")?;
        }
        shape => {
            let blank_lines = "\n".repeat(shape.len() - 2);
            let indent = " ".repeat(depth + 1);
            let separator = format!(",\n{}{}", blank_lines, indent);
            f.write_str("[")?;

            for i in 0..shape[0] {
                f.write_str(&separator)?;
                format_tensor(&tensor.axis_index(Axis(0), i), depth, f)?;
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
    fn test_vec() {
        let mut t = [1, 2, 3];
        let mut t1 = [1, 2, 3];
        let a: Vec<u32> = t.iter().zip(t1).map(|(x1, x2)| x1 + x2).collect();
        println!("a:{:?}", a);
    }

    #[test]
    fn test_rawten() {
        let mut t = RawTen::<u32>::with_capacity(10);
        t.fill(3, 10);
        println!("t:{:?}", t.take_as_vec());
    }

    #[test]
    fn test_mat() {
        let m = mat(&[[1.0, 2.0], [3.0, 4.0]]);
        let v = m.as_slice();
        let s = &m.stride;
        {
            let t = m.as_ref();
        }
        println!("s:{:?}", s);
        println!("v:{:?}", m);
    }

    #[test]
    fn test_cude() {
        let m = cube(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);
        let v = m.as_slice();
        for v in m.as_ref().iter() {
            println!("t:{:?}", v);
        }
        println!("v:{:?}", m);
    }

    #[test]
    fn test_fmt() {
        let a = Tensor::<f64>::from_elem(1.0, Shape::from_slice(&[4usize, 3, 2]));
        let v = a.as_slice();
        let t = a.as_ref();
        println!("v:{:?}", v);
    }
}
