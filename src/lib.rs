mod broadcast;
mod error;
mod method;
mod op;
pub mod shape;
mod simd;
mod zip;
extern crate alloc;
pub use crate::shape::{Axis, Shape};
use crate::zip::Zip;
use core::ptr::{self, NonNull};

use rawpointer::PointerExt;
use std::fmt;
use std::mem::forget;
mod tensor;

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

pub fn arr<A: Clone>(xs: &[A]) -> DTensor<A> {
    DTensor::arr(xs.to_vec())
}

pub fn mat<A: Clone, const N: usize>(xs: &[[A; N]]) -> DTensor<A> {
    DTensor::mat(xs.to_vec())
}

pub fn cube<A: Clone, const N: usize, const M: usize>(xs: &[[[A; N]; M]]) -> DTensor<A> {
    DTensor::cube(xs.to_vec())
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
enum RawData<P> {
    Own(RawPtr<P>),
    Ref(RawRef<P>),
}

#[derive(Clone)]
struct RawRef<P> {
    ptr: NonNull<P>,
    len: usize,
}

impl<P> RawRef<P> {
    fn as_slice(&self) -> &[P] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const P, self.len) }
    }

    fn as_slice_mut(&mut self) -> &mut [P] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut P, self.len) }
    }
}

#[derive(Clone)]
struct RawPtr<P> {
    ptr: NonNull<P>,
    len: usize,
    cap: usize,
}

impl<P> RawData<P> {
    pub(crate) fn as_ref(&self) -> RawData<P> {
        return match self {
            RawData::Own(v) => RawData::Ref(RawRef {
                ptr: v.ptr,
                len: v.len,
            }),
            RawData::Ref(v) => RawData::Ref(RawRef {
                ptr: v.ptr,
                len: v.len,
            }),
        };
    }

    pub(crate) fn as_ptr(&self) -> NonNull<P> {
        return match self {
            RawData::Own(v) => v.ptr,
            RawData::Ref(v) => v.ptr,
        };
    }

    pub(crate) fn len(&self) -> usize {
        return match self {
            RawData::Own(v) => v.len,
            RawData::Ref(v) => v.len,
        };
    }

    fn as_slice(&self) -> &[P] {
        return match self {
            RawData::Own(v) => v.as_slice(),
            RawData::Ref(v) => v.as_slice(),
        };
    }

    fn as_slice_mut(&mut self) -> &mut [P] {
        return match self {
            RawData::Own(v) => v.as_slice_mut(),
            RawData::Ref(v) => v.as_slice_mut(),
        };
    }
}

impl<P> RawPtr<P> {
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

pub struct DTensorIter<'a, A> {
    ptr: NonNull<A>,
    dim: Shape,
    strides: Shape,
    index: Option<Shape>,
    _marker: PhantomData<&'a A>,
}

impl<'a, A> Iterator for DTensorIter<'a, A> {
    type Item = &'a mut A;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        let offset = Shape::stride_offset(&index, &self.strides);
        self.index = self.dim.next_for(index);
        let ptr = unsafe { self.ptr.offset(offset) }.as_ptr();
        unsafe { Some(&mut *ptr) }
    }
}

impl<'a, A> DTensorIter<'a, A> {
    fn zip(self, t: DTensorIter<'a, A>) -> Zip<'a, A>
    where
        Self: Sized,
    {
        Zip::new(self, t)
    }
}

impl<A> DTensor<A> {
    ///
    fn axis_index(&self, a: Axis, index: usize) -> DTensor<A> {
        let stride = self.stride.as_slice()[a.index()];
        let offset = index * stride;

        let axis_dim = self.dim.select_axis(a);
        let s = axis_dim.strides();

        let raw_ref = RawRef {
            ptr: unsafe { self.data.as_ptr().offset(offset as isize) },
            len: self.data.len() - offset,
        };
        DTensor {
            data: MemoryDevice::Cpu(RawData::Ref(raw_ref)),
            dim: axis_dim,
            stride: s,
        }
    }

    pub fn iter(&self) -> DTensorIter<'_, A> {
        DTensorIter {
            ptr: self.data.as_ptr(),
            dim: self.dim.clone(),
            strides: self.stride.clone(),
            index: self.dim.first_index(),
            _marker: PhantomData,
        }
    }

    pub fn into_iter<'a>(self) -> DTensorIter<'a, A> {
        DTensorIter {
            ptr: self.data.as_ptr(),
            dim: self.dim.clone(),
            strides: self.stride.clone(),
            index: self.dim.first_index(),
            _marker: PhantomData,
        }
    }
}

#[derive(Clone)]
enum MemoryDevice<A> {
    Cpu(RawData<A>),
    Gpu(), //todo!
}

impl<A> MemoryDevice<A> {
    pub(crate) fn as_ptr(&self) -> NonNull<A> {
        return match self {
            MemoryDevice::Cpu(v) => v.as_ptr(),
            MemoryDevice::Gpu() => {
                todo!()
            }
        };
    }

    pub(crate) fn as_ref(&self) -> MemoryDevice<A> {
        return match self {
            MemoryDevice::Cpu(v) => MemoryDevice::Cpu(v.as_ref()),
            MemoryDevice::Gpu() => {
                todo!()
            }
        };
    }

    pub(crate) fn len(&self) -> usize {
        return match self {
            MemoryDevice::Cpu(v) => v.len(),
            MemoryDevice::Gpu() => {
                todo!()
            }
        };
    }

    fn as_slice(&self) -> &[A] {
        return match self {
            MemoryDevice::Cpu(v) => v.as_slice(),
            MemoryDevice::Gpu() => {
                todo!()
            }
        };
    }

    fn as_slice_mut(&mut self) -> &mut [A] {
        return match self {
            MemoryDevice::Cpu(v) => v.as_slice_mut(),
            MemoryDevice::Gpu() => {
                todo!()
            }
        };
    }
}

#[derive(Clone)]
pub struct DTensor<A> {
    data: MemoryDevice<A>,
    dim: Shape,
    stride: Shape,
}

impl<A> DTensor<A> {
    fn arr(mut xs: Vec<A>) -> Self {
        let dim = [xs.len()];
        let shape = Shape::from_array(dim);
        let ptr = xs.as_mut_ptr();
        let cap = shape.size();
        forget(xs);
        let data = RawPtr::from_raw_parts(ptr as *mut A, cap, cap);
        DTensor::from_raw(data, shape)
    }

    fn mat<const N: usize>(mut xs: Vec<[A; N]>) -> Self {
        let dim = [xs.len(), N];
        let shape = Shape::from_array(dim);
        let ptr = xs.as_mut_ptr();
        let cap = shape.size();
        forget(xs);
        let data = RawPtr::from_raw_parts(ptr as *mut A, cap, cap);
        DTensor::from_raw(data, shape)
    }

    fn cube<const M: usize, const N: usize>(mut xs: Vec<[[A; N]; M]>) -> Self {
        let dim = [xs.len(), N, M];
        let shape = Shape::from_array(dim);
        let ptr = xs.as_mut_ptr();
        let cap = shape.size();
        forget(xs);
        let data = RawPtr::from_raw_parts(ptr as *mut A, cap, cap);
        DTensor::from_raw(data, shape)
    }

    fn from_raw(data: RawPtr<A>, s: Shape) -> Self {
        let stride = s.strides();
        Self {
            data: MemoryDevice::Cpu(RawData::Own(data)),
            dim: s,
            stride: stride,
        }
    }

    pub fn from_raw_data(ptr: *mut A, length: usize, s: Shape) -> Self {
        let data = RawPtr::from_raw_parts(ptr, length, length);
        DTensor::from_raw(data, s)
    }

    pub fn as_ref(&self) -> DTensor<A> {
        DTensor {
            data: self.data.as_ref(),
            dim: self.dim.clone(),
            stride: self.stride.clone(),
        }
    }

    pub fn from_elem(a: A, s: Shape) -> Self
    where
        A: Clone,
    {
        let mut v = RawPtr::with_capacity(s.size());
        v.fill(a, s.size());
        let stride = s.strides();
        Self {
            data: MemoryDevice::Cpu(RawData::Own(v)),
            dim: s,
            stride: stride,
        }
    }

    pub fn reshape(&self, d: Shape) -> DTensor<A>
    where
        A: Clone,
    {
        if self.dim.size() != d.size() {
            panic!(
                "ndarray: incompatible shapes in reshape, attempted from: {:?}, to: {:?}",
                self.dim.as_slice(),
                d.as_slice()
            )
        }
        let stride = d.strides();
        DTensor {
            data: self.data.as_ref(),
            dim: d,
            stride: stride,
        }
    }

    pub fn with_shape(v: Vec<A>, d: Shape) -> Self {
        Self::from_vec(v, d)
    }

    fn from_vec(v: Vec<A>, d: Shape) -> Self {
        let stride = d.strides();
        let t = Self {
            data: MemoryDevice::Cpu(RawData::Own(RawPtr::from_raw_parts(
                v.as_ptr() as *mut A,
                v.len(),
                v.capacity(),
            ))),
            dim: d,
            stride: stride,
        };
        forget(v);
        t
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

impl<T> Drop for RawData<T> {
    fn drop(&mut self) {
        match self {
            RawData::Own(v) => drop(v),
            RawData::Ref(_) => {}
        }
    }
}

impl<T> Drop for RawPtr<T> {
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

impl<A: fmt::Debug> fmt::Debug for DTensor<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor(self, 0, f)
    }
}

fn format_tensor<A: fmt::Debug>(
    tensor: &DTensor<A>,
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
            format_tensor(&tensor.axis_index(Axis(0), 0), depth, f)?;
            for i in 1..shape[0] {
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
        let mut t = RawPtr::<u32>::with_capacity(10);
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
        let a = DTensor::<f64>::from_elem(1.0, Shape::from_array([4usize, 3, 2]));
        let v = a.as_slice();
        let t = a.as_ref();
        println!("v:{:?}", a);
    }

    #[test]
    fn test_reshape() {
        let t1 = arr(&[1, 2, 3, 4, 5, 6]);
        println!("t1:{:?}", t1);

        let t2 = t1.reshape(Shape::from_array([3, 2]));
        println!("t2:{:?}", t2);
    }
}
