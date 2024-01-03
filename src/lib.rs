#![feature(concat_idents)]
mod broadcast;
mod error;
mod method;
mod op;
pub mod shape;
mod simd;
mod zip;
extern crate alloc;

use crate::error::{GError, GResult};
use crate::shape::Dim;
pub use crate::shape::{Axis, Shape};
use crate::zip::Zip;
use core::ptr::{self, NonNull};
use rawpointer::PointerExt;
use std::fmt;
use std::mem::forget;
mod tensor;
use half::f16;
pub use tensor::DType;
pub use tensor::Tensor;
pub use tensor::TensorValue;
pub trait TensorType:
    std::cmp::PartialOrd + fmt::Debug + PartialEq + Copy + num_traits::NumAssign + Sync + Send
{
    #[inline(always)]
    unsafe fn vec_dot(lhs: *const Self, rhs: *const Self, res: *mut Self, len: usize) {
        *res = Self::zero();
        for i in 0..len {
            *res += *lhs.add(i) * *rhs.add(i)
        }
    }
}

pub trait Similarity {
    fn ip() -> usize;
    fn l2() -> usize;
    fn cosine() -> usize;
    fn hamming() -> usize;
    fn jaccard() -> usize;
    fn tanimoto() -> usize;
}

pub trait Zero: Clone {
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

zero_impl!(u8, 0);
zero_impl!(u16, 0);
zero_impl!(u32, 0);
zero_impl!(u64, 0);
zero_impl!(i8, 0);
zero_impl!(i16, 0);
zero_impl!(i32, 0);
zero_impl!(i64, 0);
zero_impl!(f16, f16::from_f32(0.0));
zero_impl!(f32, 0.0);
zero_impl!(f64, 0.0);

pub fn arr<A: TensorType>(xs: &[A]) -> DTensor<A> {
    DTensor::arr(xs.to_vec())
}

pub fn mat<A: TensorType, const N: usize>(xs: &[[A; N]]) -> DTensor<A> {
    DTensor::mat(xs.to_vec())
}

pub fn cube<A: TensorType, const N: usize, const M: usize>(xs: &[[[A; N]; M]]) -> DTensor<A> {
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

    fn take_as_vec(self) -> Vec<P> {
        let capacity = self.cap;
        let len = self.len;
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
    strides: Box<[usize]>,
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

impl<'a, A> DTensorIter<'a, A>
where
    A: TensorType,
{
    fn zip(self, t: DTensorIter<'a, A>) -> Zip<'a, A>
    where
        Self: Sized,
    {
        Zip::new(self, t)
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
pub struct DTensor<A>
where
    A: TensorType,
{
    data: MemoryDevice<A>,
    dim: Dim,
}

impl<A: TensorType> AsRef<DTensor<A>> for DTensor<A> {
    fn as_ref(&self) -> &DTensor<A> {
        &self
    }
}

impl<A> DTensor<A>
where
    A: TensorType,
{
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
            dim: Dim { s, stride },
        }
    }

    pub fn from_raw_data(ptr: *mut A, length: usize, s: Shape) -> Self {
        let data = RawPtr::from_raw_parts(ptr, length, length);
        DTensor::from_raw(data, s)
    }

    pub fn view(&self) -> DTensor<A> {
        DTensor {
            data: self.data.as_ref(),
            dim: self.dim.clone(),
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
            dim: Dim { s, stride },
        }
    }

    pub fn reshape(&self, s: Shape) -> DTensor<A>
    where
        A: Clone,
    {
        if self.dim.shape().size() != s.size() {
            panic!(
                "ndarray: incompatible shapes in reshape, attempted from: {:?}, to: {:?}",
                self.dim.shape().as_slice(),
                s.as_slice()
            )
        }
        let stride = s.strides();
        DTensor {
            data: self.data.as_ref(),
            dim: Dim { s, stride },
        }
    }

    pub fn with_shape(v: Vec<A>, d: Shape) -> Self {
        Self::from_vec(v, d)
    }

    fn from_vec(v: Vec<A>, s: Shape) -> Self {
        let stride = s.strides();
        let t = Self {
            data: MemoryDevice::Cpu(RawData::Own(RawPtr::from_raw_parts(
                v.as_ptr() as *mut A,
                v.len(),
                v.capacity(),
            ))),
            dim: Dim { s, stride },
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

    pub fn zeros(d: Shape) -> Self {
        Self::from_elem(A::zero(), d)
    }

    pub fn shape(&self) -> &Shape {
        &self.dim.shape()
    }

    pub fn size(&self) -> usize {
        self.dim.shape().dim()
    }

    pub fn dim(&self) -> &Dim {
        &self.dim
    }

    fn transpose(self, d1: usize, d2: usize) -> GResult<DTensor<A>> {
        if d1 > self.size() {
            return Err(GError::DimOutOfRange {
                shape: self.dim().shape().clone(),
                dim: d1,
                op: "transpose",
            });
        }
        if d2 > self.size() {
            return Err(GError::DimOutOfRange {
                shape: self.dim().shape().clone(),
                dim: d2,
                op: "transpose",
            });
        }
        let new_dim = self.dim.transpose(d1, d2)?;
        Ok(DTensor {
            data: self.data,
            dim: new_dim,
        })
    }

    fn cat<T: AsRef<DTensor<A>>>(ts: &[T], dim: usize) -> GResult<DTensor<A>> {
        if dim == 0 {
            Self::cat0(ts)
        } else {
            let args: Vec<DTensor<A>> = ts
                .iter()
                .map(|a| a.as_ref().view().transpose(0, dim))
                .collect::<GResult<Vec<DTensor<A>>>>()?;
            let cat = Self::cat0(&args)?;
            cat.transpose(0, dim)
        }
    }

    fn cat0<T: AsRef<DTensor<A>>>(ts: &[T]) -> GResult<DTensor<A>> {
        let t0 = ts[0].as_ref();
        let mut cat_dims = t0.shape().as_slice().to_vec();
        cat_dims[0] = 0;
        let mut offsets = vec![0usize];
        for (t_i, arg) in ts.iter().enumerate() {
            let t = arg.as_ref();
            for (dim_idx, (v1, v2)) in t0
                .shape()
                .as_slice()
                .iter()
                .zip(t.shape().as_slice().iter())
                .enumerate()
            {
                if dim_idx == 0 {
                    cat_dims[0] += v2;
                }
            }
            let next_offset = offsets.last().unwrap() + t.shape().elem_count();
            offsets.push(next_offset);
        }
        println!("cat_dims:{:?}", cat_dims);
        println!("offsets:{:?}", offsets);
        let shape: Shape = Shape::from_vec(cat_dims);
        let mut new_tensor = Self::zeros(shape);
        for (arg, &offset) in ts.iter().zip(offsets.iter()) {
            let t = arg.as_ref();
            copy_strided_src(t.as_slice(), new_tensor.as_slice_mut(), offset, t.dim());
        }
        Ok(new_tensor)
    }

    fn axis_index(&self, a: Axis, index: usize) -> DTensor<A> {
        let axis = a.index();
        let stride = self.dim.stride[axis];
        let offset = index * stride;
        let axis_dim = self.dim.shape().select_axis(a);
        let s = shape::select_axis(&self.dim.stride, a);
        let raw_ref = RawRef {
            ptr: unsafe { self.data.as_ptr().offset(offset as isize) },
            len: self.data.len() - offset,
        };
        DTensor {
            data: MemoryDevice::Cpu(RawData::Ref(raw_ref)),
            dim: Dim {
                s: axis_dim,
                stride: s.into_boxed_slice(),
            },
        }
    }

    fn axis_index_inner(&self, a: Axis, index: usize) -> DTensor<A> {
        let axis = a.index();
        let stride = self.dim.stride[axis];
        let offset = index * stride;
        let axis_dim = self.dim.shape().select_axis(a);
        let s = shape::select_axis(&self.dim.stride, a);
        let raw_ref = RawRef {
            ptr: unsafe { self.data.as_ptr().offset(offset as isize) },
            len: self.data.len() - offset,
        };
        DTensor {
            data: MemoryDevice::Cpu(RawData::Ref(raw_ref)),
            dim: Dim {
                s: axis_dim,
                stride: s.into_boxed_slice(),
            },
        }
    }

    pub fn iter(&self) -> DTensorIter<'_, A> {
        DTensorIter {
            ptr: self.data.as_ptr(),
            dim: self.dim.s.clone(),
            strides: self.dim.stride.clone(),
            index: self.dim.shape().first_index(),
            _marker: PhantomData,
        }
    }

    pub fn into_iter<'a>(self) -> DTensorIter<'a, A> {
        DTensorIter {
            ptr: self.data.as_ptr(),
            dim: self.dim.s.clone(),
            strides: self.dim.stride.clone(),
            index: self.dim.s.first_index(),
            _marker: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct StridedIndex<'a> {
    next_storage_index: Option<usize>,
    multi_index: Vec<usize>,
    dims: &'a [usize],
    stride: &'a [usize],
}

impl<'a> StridedIndex<'a> {
    pub(crate) fn new(dims: &'a [usize], stride: &'a [usize], start_offset: usize) -> Self {
        let elem_count: usize = dims.iter().product();
        let next_storage_index = if elem_count == 0 {
            None
        } else {
            // This applies to the scalar case.
            Some(start_offset)
        };
        StridedIndex {
            next_storage_index,
            multi_index: vec![0; dims.len()],
            dims,
            stride,
        }
    }
}

impl<'a> Iterator for StridedIndex<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let storage_index = match self.next_storage_index {
            None => return None,
            Some(storage_index) => storage_index,
        };
        let mut updated = false;
        let mut next_storage_index = storage_index;
        for ((multi_i, max_i), stride_i) in self
            .multi_index
            .iter_mut()
            .zip(self.dims.iter())
            .zip(self.stride.iter())
            .rev()
        {
            let next_i = *multi_i + 1;
            if next_i < *max_i {
                *multi_i = next_i;
                updated = true;
                next_storage_index += stride_i;
                break;
            } else {
                next_storage_index -= *multi_i * stride_i;
                *multi_i = 0
            }
        }
        self.next_storage_index = if updated {
            Some(next_storage_index)
        } else {
            None
        };
        Some(storage_index)
    }
}

#[derive(Debug)]
pub enum StridedBlocks<'a> {
    SingleBlock {
        start_offset: usize,
        len: usize,
    },
    MultipleBlocks {
        block_start_index: StridedIndex<'a>,
        block_len: usize,
    },
}

fn copy_strided_src<T: Copy>(src: &[T], dst: &mut [T], dst_offset: usize, src_d: &Dim) {
    match src_d.strided_blocks() {
        crate::StridedBlocks::SingleBlock { start_offset, len } => {
            let to_copy = (dst.len() - dst_offset).min(len);
            dst[dst_offset..dst_offset + to_copy]
                .copy_from_slice(&src[start_offset..start_offset + to_copy])
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len: 1,
        } => {
            for (dst_index, src_index) in block_start_index.enumerate() {
                let dst_index = dst_index + dst_offset;
                if dst_index >= dst.len() {
                    break;
                }
                dst[dst_index] = src[src_index]
            }
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            println!("MultipleBlocks");
            let mut dst_index = dst_offset;
            for src_index in block_start_index {
                let next_dst_index = dst_index + block_len;
                if dst_index >= dst.len() {
                    break;
                }
                let to_copy = usize::min(block_len, dst.len() - dst_index);
                dst[dst_index..dst_index + to_copy]
                    .copy_from_slice(&src[src_index..src_index + to_copy]);
                dst_index = next_dst_index
            }
        }
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

impl<A: TensorType> fmt::Debug for DTensor<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor(self.view(), 0, f)
    }
}

fn format_tensor<A: TensorType>(
    tensor: DTensor<A>,
    depth: usize,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    match tensor.shape().as_slice() {
        &[] => {}
        &[len] => {
            f.write_str("[")?;
            for (i, v) in tensor.iter().enumerate() {
                if i > 0 {
                    f.write_str(",")?;
                }
                f.write_str(&format!(" {:?}", *v))?;
            }
            f.write_str("]")?;
        }
        shape => {
            let blank_lines = "\n".repeat(shape.len() - 2);
            let indent = " ".repeat(depth + 1);
            let separator = format!(",\n{}{}", blank_lines, indent);
            f.write_str("[")?;
            format_tensor(tensor.axis_index(Axis(0), 0), depth, f)?;
            for i in 1..shape[0] {
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
        let mut t = RawPtr::<u32>::with_capacity(10);
        t.fill(3, 10);
        println!("t:{:?}", t.take_as_vec());
    }

    #[test]
    fn test_mat() {
        let m = mat(&[[1.0, 2.0], [3.0, 4.0]]);
        let v = m.as_slice();
        let s = &m.dim.stride;
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
        println!("v:{:?}", m);
        println!("shape:{:?}", m.dim().shape().as_slice());
        println!("stride:{:?}", m.dim().stride());

        for v in m.iter() {
            println!("v:{:?}", *v);
        }

        let m1 = m.transpose(0, 1).unwrap();

        for v in m1.iter() {
            println!("v1:{:?}", *v);
        }
        let vv = m1.axis_index(Axis(0), 0);
        println!("vv:{:?}", vv);
        println!("m1:{:?}", m1);
        println!("shape:{:?}", m1.dim().shape().as_slice());
        println!("stride:{:?}", m1.dim().stride());
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

    #[test]
    fn test_transpose() {
        let m = cube(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);
        let m1 = m.transpose(1, 0).unwrap();
        println!("m1:{:?}", m1);
    }

    #[test]
    fn test_cat() {
        let m1 = mat(&[[1.0, 2.0], [3.0, 4.0]]);

        let m2 = mat(&[[1.0, 2.0], [3.0, 4.0]]);

        let m3 = DTensor::cat(&[&m1, &m2], 0).unwrap();

        println!("m3:{:?}", m3);
        for m in m3.iter() {
            println!("m:{:?}", *m);
        }

        let m4: DTensor<f64> = DTensor::cat(&[&m1, &m2], 1).unwrap();

        println!("m4:{:?}", m4);
        println!("shape:{:?}", m4.dim());
        println!("slice:{:?}", m4.as_slice());

        let m5 = mat(&[[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]]);
        println!("m5:{:?}", m5);
        println!("shape:{:?}", m5.dim());
        println!("slice:{:?}", m5.as_slice());
    }
}
