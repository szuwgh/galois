mod dimension;
mod method;
mod zip;
extern crate alloc;
use crate::dimension::{Axis, Dimension, DynDim};
use core::ptr::{self, NonNull};
use rawpointer::PointerExt;
use std::fmt;
use std::mem::forget;
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

//一维数组
pub type Array<A> = TensorData<A, [usize; 1]>;

//二维数组
pub type Matrix<A> = TensorData<A, [usize; 2]>;

//三维数组
pub type Cube<A> = TensorData<A, [usize; 3]>;

//四维维数组
pub type Ten4<A> = TensorData<A, [usize; 4]>;

//动态数组
pub type Tensor<A> = TensorData<A, DynDim>;

pub fn mat<A: Clone, const N: usize>(xs: &[[A; N]]) -> Matrix<A> {
    Matrix::from(xs.to_vec())
}

impl<A, const N: usize> From<Vec<[A; N]>> for Matrix<A> {
    fn from(mut xs: Vec<[A; N]>) -> Self {
        let dim = [xs.len(), N];
        let ptr = xs.as_mut_ptr();
        let cap = dim.size();
        forget(xs);
        let data = RawTen::from_raw_parts(ptr as *mut A, cap, cap);
        Matrix::from_raw_data(data, dim)
    }
}

pub fn cube<A: Clone, const N: usize, const M: usize>(xs: &[[[A; N]; M]]) -> Cube<A> {
    Cube::from(xs.to_vec())
}

impl<A, const N: usize, const M: usize> From<Vec<[[A; N]; M]>> for Cube<A> {
    fn from(mut xs: Vec<[[A; N]; M]>) -> Self {
        let dim = [xs.len(), N, M];
        let ptr = xs.as_mut_ptr();
        let cap = dim.size();
        forget(xs);
        let data = RawTen::from_raw_parts(ptr as *mut A, cap, cap);
        Cube::from_raw_data(data, dim)
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

pub struct TensorIter<A, D>
where
    D: Dimension,
{
    r: TenRef<A, D>,
    index: usize,
}

impl<A, D> TensorIter<A, D> where D: Dimension {}

impl<A, D> Iterator for TensorIter<A, D>
where
    D: Dimension,
{
    type Item = A;
    fn next(&mut self) -> Option<Self::Item> {
        // self.r.ptr.offset(i);
        todo!();
    }
}

pub struct TenRef<A, D>
where
    D: Dimension,
{
    ptr: NonNull<A>,
    len: usize,
    dim: D,
    stride: D,
}

impl<A, D> TenRef<A, D>
where
    D: Dimension,
{
    ///
    ///```
    /// let a = arr2(&[[1., 2. ],    // ... axis 0, row 0
    ///                [3., 4. ],    // --- axis 0, row 1
    ///                [5., 6. ]]);  // ... axis 0, row 2
    /// //               .   \
    /// //                .   axis 1, column 1
    /// //                 axis 1, column 0
    /// assert!(
    ///     a.index_axis(Axis(0), 1) == ArrayView::from(&[3., 4.]) &&
    ///     a.index_axis(Axis(1), 1) == ArrayView::from(&[2., 4., 6.])
    /// );
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
    fn axis_index(&mut self, a: Axis, index: usize) -> TenRef<A, D::AxisDimension>
    where
        D: Dimension,
    {
        let stride = self.stride.slice()[a.index()];
        let offset = index * stride;
        let axis_dim = self.dim.select_axis(a);
        let s = axis_dim.strides();
        TenRef {
            ptr: unsafe { self.ptr.offset(offset as isize) },
            len: self.len - offset,
            dim: axis_dim,
            stride: s,
        }
    }

    pub fn slice(&self) -> &[A] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const A, self.len) }
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

// impl<A, D> TenRef<A, D> {}

pub struct TensorData<A, D>
where
    D: Dimension,
{
    data: RawTen<A>,
    dim: D,
    stride: D,
}

impl<A, D> TensorData<A, D>
where
    D: Dimension,
{
    fn from_raw_data(data: RawTen<A>, d: D) -> Self {
        let stride = d.strides();
        Self {
            data: data,
            dim: d,
            stride: stride,
        }
    }

    pub fn as_ref<'a>(&'a self) -> TenRef<A, D>
    where
        D: Dimension,
    {
        TenRef {
            ptr: self.data.ptr,
            len: self.data.len,
            dim: self.dim.clone(),
            stride: self.stride.clone(),
        }
    }

    pub fn from_elem(s: A, d: D) -> Self
    where
        A: Clone,
    {
        //申请内存
        let mut v = RawTen::with_capacity(d.size());
        v.fill(s, d.size());
        let stride = d.strides();
        Self {
            data: v,
            dim: d,
            stride: stride,
        }
    }

    pub fn from_vec(v: Vec<A>, d: D) -> Self {
        let stride = d.strides();
        Self {
            data: RawTen::from_raw_parts(v.as_ptr() as *mut A, v.len(), v.capacity()),
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

impl<A: fmt::Debug, D> fmt::Debug for TensorData<A, D>
where
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor(self.as_ref(), 0, f)
    }
}

fn format_tensor<A: fmt::Debug, D>(
    mut tensor: TenRef<A, D>,
    depth: usize,
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
            let blank_lines = "\n".repeat(shape.len() - 2);
            let indent = " ".repeat(depth + 1);
            let separator = format!(",\n{}{}", blank_lines, indent);
            f.write_str("[")?;

            for i in 0..shape[0] {
                f.write_str(&separator)?;
                format_tensor(tensor.axis_index(Axis(0), i), depth, f)?;
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
        let v = m.slice();
        let t = m.as_ref();
        println!("v:{:?}", m);
    }

    #[test]
    fn test_cude() {
        let m = cube(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);
        let v = m.slice();
        let t = m.as_ref();
        println!("v:{:?}", m);
    }

    #[test]
    fn test_fmt() {
        // let x = (1, 2, 3);
        // let b = Vec::new();
        // b.as_ref()

        let a = TensorData::<f64, _>::from_elem(1.0, [4usize, 3, 2]);
        let v = a.slice();
        let t = a.as_ref();
        println!("v:{:?}", v);
        // let x = Matrix::<f64>::zeros([2, 3]);
        // println!("xxx");
        // shape=[4, 3, 2, 2], strides=[12, 4, 2, 1]
        //let x: [usize; 4] = [4, 3, 2, 2];
        //let z = x.strides();
        // println!("{:?}", t);
    }
}
