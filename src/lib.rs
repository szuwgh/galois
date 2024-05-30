#![feature(concat_idents)]
mod broadcast;
pub mod error;
pub mod op;
pub mod shape;
mod simd;

mod zip;
extern crate alloc;

use crate::error::{GError, GResult};
use crate::shape::Dim;
pub use crate::shape::{Axis, Shape};
use crate::zip::Zip;
use core::ptr::{self, NonNull};

use op::UnaryOp;
use rawpointer::PointerExt;
use shape::ShapeIter;
use std::fmt;
use std::mem::forget;
mod tensor;
use half::f16;
pub use tensor::DType;

pub trait ToUsize {
    fn as_usize(&self) -> usize;
}

pub trait FromF32 {
    fn from_f32(a: f32) -> Self;
}

pub trait FromF64 {
    fn from_f64(a: f64) -> Self;
}

pub trait TensorType:
    std::cmp::PartialOrd
    + fmt::Debug
    + PartialEq
    + Copy
    + num_traits::NumAssign
    + Sync
    + Send
    + ToUsize
    + FromF32
    + FromF64
    + UnaryOp
    + 'static
{
    const DTYPE: DType;
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

pub fn arr<A: TensorType>(xs: &[A]) -> Tensor {
    Tensor::arr(xs.to_vec())
}

pub fn mat<A: TensorType, const N: usize>(xs: &[[A; N]]) -> Tensor {
    Tensor::mat(xs.to_vec())
}

pub fn cube<A: TensorType, const N: usize, const M: usize>(xs: &[[[A; N]; M]]) -> Tensor {
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

trait Nd: TensorType {
    fn to_cpu_device(a: Self, size: usize) -> CpuDevice {
        let mut v = RawPtr::<Self>::with_capacity(size);
        v.fill(a, size);

        // match Self::DTYPE {
        //     DType::F16 => CpuDevice::F16(RawData::Own(v)),
        //     DType::F32 => {
        //         let mut v = RawPtr::<f32>::with_capacity(size);
        //         v.fill(a as f32, size);
        //         CpuDevice::F32(RawData::Own(v))
        //     }
        //     DType::F64 => {
        //         let mut v = RawPtr::<f64>::with_capacity(size);
        //         v.fill(a as f32, size);
        //         CpuDevice::F64(RawData::Own(data))
        //     }
        //     _ => {
        //         todo!()
        //     }
        // }
    }

    fn to_gpu_device();
}

trait NdArray {
    fn to_cpu_device(self) -> CpuDevice;

    fn to_gpu_device();
}

impl<A: TensorType> NdArray for Vec<A> {
    fn to_cpu_device(self) -> CpuDevice {
        let device = match A::DTYPE {
            DType::F16 => {
                let data =
                    RawPtr::from_raw_parts(self.as_ptr() as *mut f16, self.len(), self.capacity());
                CpuDevice::F16(RawData::Own(data))
            }
            DType::F32 => {
                let data =
                    RawPtr::from_raw_parts(self.as_ptr() as *mut f32, self.len(), self.capacity());
                CpuDevice::F32(RawData::Own(data))
            }
            DType::F64 => {
                let data =
                    RawPtr::from_raw_parts(self.as_ptr() as *mut f64, self.len(), self.capacity());
                CpuDevice::F64(RawData::Own(data))
            }
            _ => {
                todo!()
            }
        };
        forget(self);
        return device;
    }

    fn to_gpu_device() {
        todo!()
    }
}

impl<A: TensorType> NdArray for &[A] {
    fn to_cpu_device(self) -> CpuDevice {
        match A::DTYPE {
            DType::F16 => {
                let data = RawRef::from_raw_parts(self.as_ptr() as *mut f16, self.len());
                CpuDevice::F16(RawData::Ref(data))
            }
            DType::F32 => {
                let data = RawRef::from_raw_parts(self.as_ptr() as *mut f32, self.len());
                CpuDevice::F32(RawData::Ref(data))
            }
            DType::F64 => {
                let data = RawRef::from_raw_parts(self.as_ptr() as *mut f64, self.len());
                CpuDevice::F64(RawData::Ref(data))
            }
            _ => {
                todo!()
            }
        }
    }

    fn to_gpu_device() {
        todo!()
    }
}

impl<A: TensorType, const N: usize> NdArray for Vec<[A; N]> {
    fn to_cpu_device(self) -> CpuDevice {
        let device = match A::DTYPE {
            DType::F16 => {
                let data =
                    RawPtr::from_raw_parts(self.as_ptr() as *mut f16, self.len(), self.capacity());
                CpuDevice::F16(RawData::Own(data))
            }
            DType::F32 => {
                let data =
                    RawPtr::from_raw_parts(self.as_ptr() as *mut f32, self.len(), self.capacity());
                CpuDevice::F32(RawData::Own(data))
            }
            DType::F64 => {
                let data =
                    RawPtr::from_raw_parts(self.as_ptr() as *mut f64, self.len(), self.capacity());
                CpuDevice::F64(RawData::Own(data))
            }
            _ => {
                todo!()
            }
        };
        forget(self);
        return device;
    }

    fn to_gpu_device() {
        todo!()
    }
}

impl<A: TensorType, const N: usize, const M: usize> NdArray for Vec<[[A; N]; M]> {
    fn to_cpu_device(self) -> CpuDevice {
        let device = match A::DTYPE {
            DType::F16 => {
                let data =
                    RawPtr::from_raw_parts(self.as_ptr() as *mut f16, self.len(), self.capacity());
                CpuDevice::F16(RawData::Own(data))
            }
            DType::F32 => {
                let data =
                    RawPtr::from_raw_parts(self.as_ptr() as *mut f32, self.len(), self.capacity());
                CpuDevice::F32(RawData::Own(data))
            }
            DType::F64 => {
                let data =
                    RawPtr::from_raw_parts(self.as_ptr() as *mut f64, self.len(), self.capacity());
                CpuDevice::F64(RawData::Own(data))
            }
            _ => {
                todo!()
            }
        };
        forget(self);
        return device;
    }

    fn to_gpu_device() {
        todo!()
    }
}

#[derive(Clone)]
enum RawData<P> {
    Own(RawPtr<P>),
    Ref(RawRef<P>),
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

    pub(crate) fn offset(&self, i: usize) -> RawData<P> {
        return match self {
            RawData::Own(v) => RawData::Ref(RawRef {
                ptr: unsafe { v.ptr.offset(i as isize) },
                len: v.len - i,
            }),
            RawData::Ref(v) => RawData::Ref(RawRef {
                ptr: unsafe { v.ptr.offset(i as isize) },
                len: v.len - i,
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

    fn to_rawdata(self) -> RawData<P> {
        RawData::Ref(self)
    }

    fn from_raw_parts(ptr: *mut P, length: usize) -> Self {
        unsafe {
            Self {
                ptr: NonNull::new_unchecked(ptr),
                len: length,
            }
        }
    }
}

struct RawPtr<P> {
    ptr: NonNull<P>,
    len: usize,
    cap: usize,
}

impl<A: TensorType> NdArray for RawPtr<A> {
    fn to_cpu_device(self) -> CpuDevice {
        let device = match A::DTYPE {
            DType::F16 => {
                let data =
                    RawPtr::from_raw_parts(self.as_ptr() as *mut f16, self.len(), self.cap());
                CpuDevice::F16(RawData::Own(data))
            }
            DType::F32 => {
                let data =
                    RawPtr::from_raw_parts(self.as_ptr() as *mut f32, self.len(), self.cap());
                CpuDevice::F32(RawData::Own(data))
            }
            DType::F64 => {
                let data =
                    RawPtr::from_raw_parts(self.as_ptr() as *mut f64, self.len(), self.cap());
                CpuDevice::F64(RawData::Own(data))
            }
            _ => {
                todo!()
            }
        };
        forget(self);
        return device;
    }

    fn to_gpu_device() {
        todo!()
    }
}

impl<P> Clone for RawPtr<P> {
    fn clone(&self) -> Self {
        use alloc::alloc::{alloc, handle_alloc_error, Layout};
        let layout = match Layout::array::<P>(self.cap) {
            Ok(layout) => layout,
            Err(_) => capacity_overflow(),
        };
        let dst = if layout.size() == 0 {
            core::ptr::NonNull::<P>::dangling()
        } else {
            let ptr = unsafe { alloc(layout) } as *mut P;
            if ptr.is_null() {
                handle_alloc_error(layout)
            } else {
                unsafe { NonNull::<P>::new_unchecked(ptr) }
            }
        };
        unsafe {
            ptr::copy(self.ptr.as_ptr(), dst.as_ptr(), self.cap);
        }
        Self {
            ptr: dst,
            len: self.len,
            cap: self.cap,
        }
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

    pub(crate) fn as_ptr(&self) -> *const P {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_ptr_mut(&self) -> *mut P {
        self.ptr.as_ptr()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn cap(&self) -> usize {
        self.cap
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

    fn to_rawdata(self) -> RawData<P> {
        RawData::Own(self)
    }
}

use std::marker::PhantomData;

pub struct TensorIter<'a> {
    device: &'a Device, //NonNull<A>,
    strides: &'a [usize],
    shape_iter: ShapeIter<'a>,
    // _marker: PhantomData<&'a>,
}

impl<'a, A> Iterator for TensorIter<'a, A> {
    type Item = &'a mut A;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.shape_iter.next()?;
        let offset = Shape::stride_offset(index.as_slice(), self.strides);
        let ptr = unsafe { self.ptr.offset(offset) }.as_ptr();
        unsafe { Some(&mut *ptr) }
    }
}

impl<'a, A> TensorIter<'a, A>
where
    A: TensorType,
{
    fn zip2(self, t: TensorIter<'a, A>) -> Zip<'a, A>
    where
        Self: Sized,
    {
        Zip::new(self, t)
    }
}

#[derive(Clone)]
pub enum CpuDevice {
    F16(RawData<f16>),
    F32(RawData<f32>),
    F64(RawData<f64>),
}

impl CpuDevice {
    pub(crate) fn offset(&self, i: usize) -> CpuDevice {
        return match self {
            CpuDevice::F16(v) => CpuDevice::F16(v.offset(i)),
            CpuDevice::F32(v) => CpuDevice::F32(v.offset(i)),
            CpuDevice::F64(v) => CpuDevice::F64(v.offset(i)),
        };
    }

    pub(crate) fn as_ref(&self) -> CpuDevice {
        return match self {
            CpuDevice::F16(v) => CpuDevice::F16(v.as_ref()),
            CpuDevice::F32(v) => CpuDevice::F32(v.as_ref()),
            CpuDevice::F64(v) => CpuDevice::F64(v.as_ref()),
        };
    }

    pub(crate) fn len(&self) -> usize {
        return match self {
            CpuDevice::F16(v) => v.len(),
            CpuDevice::F32(v) => v.len(),
            CpuDevice::F64(v) => v.len(),
        };
    }
}

#[derive(Clone)]
enum Device {
    Cpu(CpuDevice),
    Gpu(), //todo!
}

impl Device {
    // pub(crate) fn as_ptr(&self) -> NonNull<A> {
    //     return match self {
    //         Storate::Cpu(v) => v.as_ptr(),
    //         Storate::Gpu() => {
    //             todo!()
    //         }
    //     };
    // }

    pub(crate) fn offset(&self, i: usize) -> Device {
        return match self {
            Device::Cpu(v) => Device::Cpu(v.offset(i)),
            Device::Gpu() => {
                todo!()
            }
        };
    }

    pub(crate) fn as_ref(&self) -> Device {
        return match self {
            Device::Cpu(v) => Device::Cpu(v.as_ref()),
            Device::Gpu() => {
                todo!()
            }
        };
    }

    pub(crate) fn len(&self) -> usize {
        return match self {
            Device::Cpu(v) => v.len(),
            Device::Gpu() => {
                todo!()
            }
        };
    }

    // fn as_slice(&self) -> &[A] {
    //     return match self {
    //         Device::Cpu(v) => v.as_slice(),
    //         Device::Gpu() => {
    //             todo!()
    //         }
    //     };
    // }

    // fn as_slice_mut(&mut self) -> &mut [A] {
    //     return match self {
    //         Device::Cpu(v) => v.as_slice_mut(),
    //         Device::Gpu() => {
    //             todo!()
    //         }
    //     };
    // }
}

#[derive(Clone)]
pub struct Tensor {
    data: Device,
    dim: Dim,
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        &self
    }
}

impl Tensor {
    fn arr<A: TensorType>(xs: Vec<A>) -> Self {
        let dim = [xs.len()];
        let shape = Shape::from_array(dim);
        let data = xs.to_cpu_device();
        Tensor::from_device(Device::Cpu(data), shape)
    }

    fn mat<A: TensorType, const N: usize>(xs: Vec<[A; N]>) -> Self {
        let dim = [xs.len(), N];
        let shape = Shape::from_array(dim);
        let data = xs.to_cpu_device();
        Tensor::from_device(Device::Cpu(data), shape)
    }

    fn cube<A: TensorType, const M: usize, const N: usize>(mut xs: Vec<[[A; N]; M]>) -> Self {
        let dim = [xs.len(), M, N];
        let shape = Shape::from_array(dim);
        let data = xs.to_cpu_device();
        Tensor::from_device(Device::Cpu(data), shape)
    }

    fn from_device(data: Device, s: Shape) -> Self {
        let stride = s.strides();
        Self {
            data: data,
            dim: Dim { s, stride },
        }
    }

    // fn from_device<A>(data: RawPtr<A>, s: Shape) -> Self {
    //     let stride = s.strides();
    //     Self {
    //         data: Device::Cpu(RawData::Own(data)),
    //         dim: Dim { s, stride },
    //     }
    // }

    // pub fn from_bytes(buf: &[u8], shape: &[usize]) {}

    // fn from_raw_bytes(buf: &[u8]) {}

    // pub fn from_raw_data<A>(ptr: *mut A, length: usize, s: Shape) -> Self {
    //     let data = RawPtr::from_raw_parts(ptr, length, length);
    //     Tensor::from_raw(data, s)
    // }

    pub fn view(&self) -> Tensor {
        Tensor {
            data: self.data.as_ref(),
            dim: self.dim.clone(),
        }
    }

    pub fn from_elem<A: TensorType>(a: A, s: Shape) -> Self
    where
        A: Clone,
    {
        let size = s.size();
        let mut v = RawPtr::<A>::with_capacity(size);
        v.fill(a, size);
        Tensor::from_device(Device::Cpu(v.to_cpu_device()), s)
    }

    // pub fn with_shape<A: TensorType>(v: Vec<A>, d: Shape) -> Self {
    //     Self::from_vec(v, d)
    // }

    // pub fn with_shape_fn<F>(d: Shape, mut f: F) -> Self
    // where
    //     F: FnMut(&Shape) -> A,
    // {
    //     let iter = d.iter();
    //     let size = d.elem_count();
    //     let mut result = Vec::with_capacity(size);
    //     let mut out_ptr = result.as_mut_ptr();
    //     let mut len = 0;
    //     iter.fold((), |(), elt| unsafe {
    //         ptr::write(out_ptr, f(elt));
    //         len += 1;
    //         result.set_len(len);
    //         out_ptr = out_ptr.offset(1);
    //     });
    //     Self::with_shape(result, d)
    // }

    fn from_vec<A: TensorType>(v: Vec<A>, s: Shape) -> Self {
        let data = v.to_cpu_device();
        Tensor::from_device(Device::Cpu(data), s)
    }

    // pub fn as_slice<A>(&self) -> &[A] {
    //     self.data.as_slice()
    // }

    // pub fn as_slice_mut(&mut self) -> &mut [A] {
    //     self.data.as_slice_mut()
    // }

    pub fn zeros<A: TensorType>(d: Shape) -> Self {
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

    pub fn single_dim(&self, dim: usize) -> GResult<usize> {
        let s = self.shape().as_slice();
        if dim > s.len() {
            return Err(GError::DimOutOfRange {
                shape: self.shape().clone(),
                dim: dim,
                op: "single_dim",
            });
        }
        Ok(s[dim])
    }

    pub fn elem_count(&self) -> usize {
        self.shape().elem_count()
    }

    pub fn is_contiguous(&self) -> bool {
        self.dim.is_contiguous()
    }

    pub fn sqrt(&self) -> Self {
        match &self.data {
            Device::Cpu(d) => {
                let v = self.as_slice().iter().map(|x| x._sqrt()).collect();
                Tensor::from_vec(v, self.dim().shape().clone())
            }
            Device::Gpu() => {
                todo!()
            }
        }
    }

    pub fn into_sqrt(mut self) -> Self {
        self.as_slice_mut().iter_mut().for_each(|v| *v = v._sqrt());
        self
    }

    pub fn chunk(&self, chunks: usize, dim: usize) -> GResult<Vec<Self>> {
        let size = self.dim().shape().as_slice()[dim];
        if size < chunks {
            (0..size).map(|i| self.narrow(dim, i, 1)).collect()
        } else {
            let chunk_size = size / chunks;
            let cnt_additional = size % chunks;
            let mut tensors = vec![];
            let mut sum_chunk_size = 0;
            for i in 0..chunks {
                let chunk_size = if i < cnt_additional {
                    chunk_size + 1
                } else {
                    chunk_size
                };
                let tensor = self.narrow(dim, sum_chunk_size, chunk_size)?;
                tensors.push(tensor);
                sum_chunk_size += chunk_size
            }
            Ok(tensors)
        }
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> GResult<Self> {
        let dims = self.shape().as_slice();
        let err = |msg| {
            Err::<(), _>(GError::NarrowInvalidArgs {
                shape: self.shape().clone(),
                dim,
                start,
                len,
                msg,
            })
        };
        if start > dims[dim] {
            err("start > dim_len")?
        }
        if start.saturating_add(len) > dims[dim] {
            err("start + len > dim_len")?
        }
        if start == 0 && dims[dim] == len {
            Ok(self.clone())
        } else {
            let new_dim = self.dim.narrow(dim, start, len)?;
            let offset = self.dim().stride()[dim] * start;
            Ok(Tensor {
                data: self.data.offset(offset),
                dim: new_dim,
            })
        }
    }

    pub fn broadcast_with(&self, s: &Shape) -> GResult<Tensor> {
        let broadcast_dim = self.dim().broadcast_with(s)?;
        Ok(Tensor {
            data: self.data.as_ref(),
            dim: broadcast_dim,
        })
    }

    // pub fn matmul(&self, rhs: &Tensor) -> GResult<Tensor> {
    //     let lhs = self;
    //     let (l_shape, r_shape) = broadcasting_matmul_op::<A>(lhs.shape(), rhs.shape())?;
    //     let l_broadcast = l_shape == *lhs.shape();
    //     let r_broadcast = r_shape == *rhs.shape();
    //     match (l_broadcast, r_broadcast) {
    //         (true, true) => lhs._matmul(rhs),
    //         (false, true) => lhs.broadcast_with(&l_shape)?._matmul(rhs),
    //         (true, false) => lhs._matmul(&rhs.broadcast_with(&r_shape)?),
    //         (false, false) => lhs
    //             .broadcast_with(&l_shape)?
    //             ._matmul(&rhs.broadcast_with(&r_shape)?),
    //     }
    // }

    pub fn _matmul(&self, rhs: &Tensor) -> GResult<Tensor> {
        let l_dim = self.shape().as_slice();
        let r_dim: &[usize] = rhs.shape().as_slice();
        let dim = l_dim.len();
        if dim < 2 || r_dim.len() != dim {
            return Err(GError::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            });
        }
        let m = l_dim[dim - 2];
        let k = l_dim[dim - 1];
        let k2 = r_dim[dim - 2];
        let n = r_dim[dim - 1];
        let mut c_dim = l_dim[..dim - 2].to_vec();
        c_dim.extend(&[m, n]);
        let c_shape = Shape::from_vec(c_dim);
        let batching: usize = l_dim[..dim - 2].iter().product();
        let batching_b: usize = r_dim[..dim - 2].iter().product();
        if k != k2 || batching != batching_b {
            return Err(GError::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            });
        }

        let c = MatMul(batching, m, n, k).compute(
            self.as_slice(),
            self.dim(),
            rhs.as_slice(),
            rhs.dim(),
        )?;
        Ok(Self::from_vec(c, c_shape))
    }

    pub fn into_reshape(self, s: Shape) -> Tensor {
        if self.dim.shape().elem_count() != s.elem_count() {
            panic!(
                "ndarray: incompatible shapes in reshape, attempted from: {:?}, to: {:?}",
                self.dim.shape().as_slice(),
                s.as_slice()
            )
        }
        if self.dim().is_contiguous() {
            let stride = s.strides();
            Self {
                data: self.data,
                dim: Dim { s, stride },
            }
        } else {
            let mut new_tensor = Self::zeros(s);
            copy_strided_src(self.as_slice(), new_tensor.as_slice_mut(), 0, self.dim());
            new_tensor
        }
    }

    pub fn reshape(&self, s: Shape) -> Tensor {
        if self.dim.shape().elem_count() != s.elem_count() {
            panic!(
                "ndarray: incompatible shapes in reshape, attempted from: {:?}, to: {:?}",
                self.dim.shape().as_slice(),
                s.as_slice()
            )
        }
        if self.dim().is_contiguous() {
            let stride = s.strides();
            Self {
                data: self.data.as_ref(),
                dim: Dim { s, stride },
            }
        } else {
            let mut new_tensor = Self::zeros(s);
            copy_strided_src(self.as_slice(), new_tensor.as_slice_mut(), 0, self.dim());
            new_tensor
        }
    }

    fn into_transpose(self, d1: usize, d2: usize) -> GResult<Tensor> {
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
        Ok(Tensor {
            data: self.data,
            dim: new_dim,
        })
    }

    pub fn pad<A: TensorType>(
        &self,
        dim: usize,
        left: usize,
        right: usize,
        elem: A,
    ) -> GResult<Tensor> {
        if left == 0 && right == 0 {
            Ok(self.clone())
        } else if left == 0 {
            let mut dims = self.dim().dims().to_vec();
            dims[dim] = right;
            let right = Tensor::<A>::from_elem(elem, Shape::from_vec(dims));
            Tensor::cat(&[self, &right], dim)
        } else if right == 0 {
            let mut dims = self.dim().dims().to_vec();
            dims[dim] = left;
            let left = Tensor::<A>::from_elem(elem, Shape::from_vec(dims));
            Tensor::cat(&[&left, &self], dim)
        } else {
            let mut dims = self.dim().dims().to_vec();
            dims[dim] = left;
            let left = Tensor::<A>::from_elem(elem, Shape::from_slice(dims.as_slice()));
            dims[dim] = right;
            let right = Tensor::<A>::from_elem(elem, Shape::from_vec(dims));
            Tensor::cat(&[&left, &self, &right], dim)
        }
    }

    pub fn cat<T: AsRef<Tensor>>(ts: &[T], dim: usize) -> GResult<Tensor> {
        if dim == 0 {
            Self::cat0(ts)
        } else {
            let args: Vec<Tensor> = ts
                .iter()
                .map(|a| a.as_ref().view().into_transpose(0, dim))
                .collect::<GResult<Vec<Tensor>>>()?;
            let cat = Self::cat0(&args)?;
            cat.into_transpose(0, dim)
        }
    }

    fn cat0<T: AsRef<Tensor>>(ts: &[T]) -> GResult<Tensor> {
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
        let shape: Shape = Shape::from_vec(cat_dims);
        let mut new_tensor = Self::zeros(shape);
        for (arg, &offset) in ts.iter().zip(offsets.iter()) {
            let t = arg.as_ref();
            copy_strided_src(t.as_slice(), new_tensor.as_slice_mut(), offset, t.dim());
        }
        Ok(new_tensor)
    }

    fn axis_index(&self, a: Axis, index: usize) -> Tensor {
        let axis = a.index();
        let stride = self.dim.stride[axis];
        let offset = index * stride;
        let axis_dim = self.dim.shape().select_axis(a);
        let s = shape::select_axis(&self.dim.stride, a);
        let raw_ref = RawRef {
            ptr: unsafe { self.data.as_ptr().offset(offset as isize) },
            len: self.data.len() - offset,
        };
        Tensor {
            data: Device::Cpu(RawData::Ref(raw_ref)),
            dim: Dim {
                s: axis_dim,
                stride: s.into_boxed_slice(),
            },
        }
    }

    fn axis_index_inner(&self, a: Axis, index: usize) -> Tensor {
        let axis = a.index();
        let stride = self.dim.stride[axis];
        let offset = index * stride;
        let axis_dim = self.dim.shape().select_axis(a);
        let s = shape::select_axis(&self.dim.stride, a);
        let raw_ref = RawRef {
            ptr: unsafe { self.data.as_ptr().offset(offset as isize) },
            len: self.data.len() - offset,
        };
        Tensor {
            data: Device::Cpu(RawData::Ref(raw_ref)),
            dim: Dim {
                s: axis_dim,
                stride: s.into_boxed_slice(),
            },
        }
    }

    pub fn iter(&self) -> TensorIter<'_, A> {
        TensorIter {
            ptr: self.data.as_ptr(),
            shape_iter: self.dim.s.iter(),
            // dim: self.dim.s.clone(),
            strides: self.dim.stride(),
            // index: self.dim.shape().first_index(),
            _marker: PhantomData,
        }
    }

    // pub fn into_iter<'a>(self) -> TensorIter<'a, A> {
    //     TensorIter {
    //         ptr: self.data.as_ptr(),
    //         shape_iter: self.dim.s.iter(),
    //         strides: self.dim.stride.clone(),
    //         //  index: self.dim.s.first_index(),
    //         _marker: PhantomData,
    //     }
    // }
}

struct MatMul(usize, usize, usize, usize);

impl MatMul {
    fn compute<T: TensorType + 'static>(
        &self,
        lhs: &[T],
        lhs_l: &Dim,
        rhs: &[T],
        rhs_l: &Dim,
    ) -> GResult<Vec<T>> {
        use gemm::{gemm, Parallelism};

        // match T::DTYPE {
        //     DType::F16 | DType::F32 | DType::F64 => {}
        //     _ => Err(Error::UnsupportedDTypeForOp(T::DTYPE, "matmul").bt())?,
        // }

        let (b, m, n, k) = (self.0, self.1, self.2, self.3);
        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let rank = lhs_stride.len();
        let lhs_cs = lhs_stride[rank - 1];
        let lhs_rs = lhs_stride[rank - 2];

        let rhs_cs = rhs_stride[rank - 1];
        let rhs_rs = rhs_stride[rank - 2];

        let a_skip: usize = match lhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => m * k,
            _ => {
                return Err(GError::MatMulUnexpectedStriding {
                    lhs_l: lhs_l.clone(),
                    rhs_l: rhs_l.clone(),
                    bmnk: (b, m, n, k),
                    msg: "non-contiguous lhs",
                });
            }
        };
        let b_skip: usize = match rhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
            [stride] => stride,
            [] => n * k,
            _ => {
                return Err(GError::MatMulUnexpectedStriding {
                    lhs_l: lhs_l.clone(),
                    rhs_l: rhs_l.clone(),
                    bmnk: (b, m, n, k),
                    msg: "non-contiguous lhs",
                });
            }
        };
        let c_skip: usize = m * n;
        let dst_shape: Shape = Shape::from_array([m, n]);
        let dst_strides = dst_shape.stride_contiguous();
        let dst_rs = dst_strides[0];
        let dst_cs = dst_strides[1];
        let mut dst = vec![T::zero(); b * m * n];
        let num_threads = num_cpus::get();
        let parallelism = if num_threads > 1 {
            Parallelism::Rayon(num_threads)
        } else {
            Parallelism::None
        };
        for step in 0..b {
            let lhs_p = &lhs[step * a_skip..];
            let rhs_p = &rhs[step * b_skip..];
            let dst_p = &mut dst[step * c_skip..];
            unsafe {
                gemm(
                    /* m: usize = */ m,
                    /* n: usize = */ n,
                    /* k: usize = */ k,
                    /* dst: *mut T = */ dst_p.as_mut_ptr(),
                    /* dst_cs: isize = */ dst_cs as isize,
                    /* dst_rs: isize = */ dst_rs as isize,
                    /* read_dst: bool = */ false,
                    /* lhs: *const T = */ lhs_p.as_ptr(),
                    /* lhs_cs: isize = */ lhs_cs as isize,
                    /* lhs_rs: isize = */ lhs_rs as isize,
                    /* rhs: *const T = */ rhs_p.as_ptr(),
                    /* rhs_cs: isize = */ rhs_cs as isize,
                    /* rhs_rs: isize = */ rhs_rs as isize,
                    /* alpha: T = */ T::zero(),
                    /* beta: T = */ T::one(),
                    /* conj_dst: bool = */ false,
                    /* conj_lhs: bool = */ false,
                    /* conj_rhs: bool = */ false,
                    parallelism,
                )
            }
        }
        Ok(dst)
    }
}

#[derive(Debug)]
pub struct StridedIndex<'a> {
    next_Device_index: Option<usize>,
    multi_index: Vec<usize>,
    dims: &'a [usize],
    stride: &'a [usize],
}

impl<'a> StridedIndex<'a> {
    pub(crate) fn new(dims: &'a [usize], stride: &'a [usize], start_offset: usize) -> Self {
        let elem_count: usize = dims.iter().product();
        let next_Device_index = if elem_count == 0 {
            None
        } else {
            // This applies to the scalar case.
            Some(start_offset)
        };
        StridedIndex {
            next_Device_index,
            multi_index: vec![0; dims.len()],
            dims,
            stride,
        }
    }
}

impl<'a> Iterator for StridedIndex<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let Device_index = match self.next_Device_index {
            None => return None,
            Some(Device_index) => Device_index,
        };
        let mut updated = false;
        let mut next_Device_index = Device_index;
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
                next_Device_index += stride_i;
                break;
            } else {
                next_Device_index -= *multi_i * stride_i;
                *multi_i = 0
            }
        }
        self.next_Device_index = if updated {
            Some(next_Device_index)
        } else {
            None
        };
        Some(Device_index)
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

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor(self.view(), 0, f)
    }
}

fn format_tensor<A: TensorType>(
    tensor: Tensor,
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
        // println!("t:{:?}", t.take_as_vec());
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

        let m1 = m.into_transpose(0, 1).unwrap();

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
        let a = Tensor::<f64>::from_elem(1.0, Shape::from_array([4usize, 3, 2]));
        let v = a.as_slice();
        let t = a.as_ref();
        println!("v:{:?}", a);
    }

    #[test]
    fn test_reshape() {
        let t1 = cube(&[
            [
                [1.0, 2.0, 1.0, 2.0],
                [3.0, 4.0, 3.0, 4.0],
                [3.0, 4.0, 3.0, 4.0],
            ],
            [
                [5.0, 6.0, 5.0, 6.0],
                [7.0, 8.0, 7.0, 8.0],
                [7.0, 8.0, 7.0, 8.0],
            ],
        ]);
        println!("t1:{:?}", t1.shape());

        let t2 = t1.reshape(Shape::from_array([2, 3, 2, 2]));
        println!("t2:{:?}", t2);
    }

    #[test]
    fn test_transpose() {
        let m = cube(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);
        let m1 = m.into_transpose(1, 0).unwrap();
        println!("m1:{:?}", m1);
    }

    #[test]
    fn test_cat() {
        let m1 = mat(&[[1.0, 2.0], [3.0, 4.0]]);

        let m2 = mat(&[[1.0, 2.0], [3.0, 4.0]]);

        let m3 = Tensor::cat(&[&m1, &m2], 0).unwrap();

        println!("m3:{:?}", m3);
        for m in m3.iter() {
            println!("m:{:?}", *m);
        }

        let m4: Tensor<f64> = Tensor::cat(&[&m1, &m2], 1).unwrap();

        println!("m4:{:?}", m4);
        println!("shape:{:?}", m4.dim());
        println!("slice:{:?}", m4.as_slice());

        let m5 = mat(&[[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]]);
        println!("m5:{:?}", m5);
        println!("shape:{:?}", m5.dim());
        println!("slice:{:?}", m5.as_slice());

        let m6 = cube(&[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]);
        let m7 = cube(&[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]);
        let m8: Tensor<f64> = Tensor::cat(&[&m6, &m7], 2).unwrap();
        println!("m8:{:?}", m8);
        println!("shape:{:?}", m8.dim());
        println!("slice:{:?}", m8.as_slice());
    }

    #[test]
    fn test_pad() {
        let m1 = cube(&[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]);
        let d = m1.pad(2, 1, 1, 0.0).unwrap().pad(3, 0, 0, 0.0).unwrap();
        for x in d.iter() {
            println!("{:?}", *x);
        }
        println!("{:?}", d);
    }

    #[test]
    fn test_matmul() {
        let m1 = cube(&[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]);
        let m2 = cube(&[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]);
        let d = m1._matmul(&m2).unwrap();
        println!("{:?}", d);

        let m1 = cube(&[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]);
        let m2 = mat(&[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
        let d = m1.matmul(&m2).unwrap();
        println!("{:?}", d);
    }

    // [[1, 2, 3],
    //  [2, 4, 6],
    //   [3, 6, 9]])
    #[test]
    fn test_with_shape_fn() {
        let m1 = Tensor::with_shape_fn(Shape::from_array([3, 3]), |s| {
            let (i, j) = s.dims2();
            ((1 + i) * (1 + j)) as u32
        });
        println!("{:?}", m1);
    }

    #[test]
    fn test_sqrt() {
        let m1 = cube(&[[[4.0, 16.0], [3.0, 4.0]], [[4.0, 9.0], [36.0, 81.0]]]);
        let s = m1.sqrt();
        println!("{:?}", m1);
        println!("{:?}", s);
    }

    #[test]
    fn test_narrow() -> GResult<()> {
        let a = mat(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let b = a.narrow(0, 0, 2)?;
        println!("b:{:?}", b);

        let c = a.narrow(1, 1, 2)?;
        println!("c:{:?}", c);
        Ok(())
    }

    #[test]
    fn test_chunk() -> GResult<()> {
        let a = mat(&[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]);
        let b = a.chunk(4, 0)?;
        for i in b {
            println!("b:{:?}", i);
        }

        let c = a.chunk(4, 1)?;
        for i in c {
            println!("c:{:?}", i);
        }
        Ok(())
    }

    #[test]
    fn test_clone() -> GResult<()> {
        let mut a = mat(&[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]);
        let b = a.clone();
        a.as_slice_mut()[1] = 100;
        println!("{:?}", a);
        println!("{:?}", b);
        Ok(())
    }
}
