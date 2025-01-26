#![feature(concat_idents)]
#![feature(non_null_convenience)]
#![feature(portable_simd)]
#![feature(slice_as_chunks)]

use crate::ggml_quants::QK_K;
pub mod cuda;
#[macro_use]
pub mod macros; // 导入宏模块
pub mod error;
pub mod ggml_quants;
pub mod kernels;
use crate::cuda::CudaDevice;use crate::cuda::CudaStorageSliceView;
pub mod op;
pub mod shape;
use std::marker::PhantomData;
use std::ops;
use std::path::PathBuf;
mod simd;
pub mod similarity;
use crate::op::GLOBAL_CPU_DEVICE_CACHE;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::simd::f32x32;
use std::simd::num::SimdFloat;
mod zip;
use crate::cuda::CudaStorageView;
extern crate alloc;
use crate::cuda::CudaStorage;
use crate::cuda::CudaStorageSlice;
use crate::error::{GError, GResult};
use crate::ggml_quants::BlockQ4_0;
use crate::ggml_quants::QK8_0;
use crate::shape::Dim;
use crate::shape::Layout;
use crate::shape::MAX_DIM;
pub use crate::shape::{Axis, Shape};
use crate::zip::Zip;
use core::ptr::{self, NonNull};

use ggml_quants::{BlockQ6K, BlockQ8_0, QK4_0};
use half::f16;
use num_traits::{FromPrimitive, ToPrimitive};
use shape::ShapeIter;
use std::fmt;
use std::mem::forget;



pub type F16 = half::f16;


const STEP: usize = 32;

pub trait TensorProto: Sized {
    type Sto: StorageProto;

    fn dim(&self) -> &Dim;

    fn dim_mut(&mut self) -> &mut Dim;

    fn size(&self) -> usize {
        self.dim().shape().len()
    }

    fn dtype(&self) -> GGmlType;

    fn storage<'a>(&'a self) -> &'a Self::Sto;

    fn storage_mut<'a>(&'a mut self) -> &'a mut Self::Sto;

    fn device(&self) -> &Device;

    fn view_tensor(& self,
        offset:usize,
        n_dims: usize,
        dtype: GGmlType,
        shape: Shape)-> TensorView<'_> {  
            let v = self.storage().offset(offset);
            let stride = shape.ggml_stride(dtype);
           TensorView {
            dtype,
            data: v,
            dim:   Dim { n_dims: n_dims, shape:shape, stride: stride },
            device: self.device().clone(),
           }
    }

    unsafe fn from_bytes(
        v: &[u8],
        n_dims: usize,
        s: Shape,
        dtype: GGmlType,
        dev: &Device,
    ) -> GResult<Self>;

    fn set_value<A: TensorType>(&mut self, value: A) {
        let row = unsafe { self.as_slice_mut::<A>() };
        row.iter_mut().for_each(|e| *e = value);
    }

    fn as_bytes(&self) -> &[u8] {
        self.storage().as_bytes()
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.storage_mut().as_bytes_mut()
    }

    unsafe fn as_slice<T>(&self) -> &[T] {
        let bytes = self.as_bytes();
        assert!(bytes.len() as usize % GS_TYPE_SIZE[self.dtype() as usize] == 0);
        let len = bytes.len() / GS_TYPE_SIZE[self.dtype() as usize]; //* GS_BLCK_SIZE[self.dtype as usize] /
        std::slice::from_raw_parts(bytes.as_ptr() as *const T, len)
    }

    unsafe fn as_slice_mut<T>(&mut self) -> &mut [T] {
        let dtype = self.dtype() as usize;
        let bytes = self.as_bytes_mut();
        assert!(bytes.len() as usize % GS_TYPE_SIZE[dtype as usize] == 0);
        let len = bytes.len() / GS_TYPE_SIZE[dtype]; //* GS_BLCK_SIZE[dtype]
                                                     // println!(
                                                     //     "bytes len:{},len:{},GS_TYPE_SIZE:{}",
                                                     //     bytes.len(),
                                                     //     len,
                                                     //     GS_TYPE_SIZE[dtype]
                                                     // );
        std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len)
    }

    fn shape_layout_mut(&mut self) -> &mut Layout {
        self.dim_mut().shape_layout_mut()
    }

    fn shape_layout(&self) -> &Layout {
        self.dim().shape_layout()
    }

    fn stride_layout_mut(&mut self) -> &mut Layout {
        self.dim_mut().stride_layout_mut()
    }

    fn stride_layout(&self) -> &Layout {
        self.dim().stride_layout()
    }

    fn ret_stride(&mut self, stride: Layout) {
        self.dim_mut().ret_stride(stride)
    }

    fn transpose(&mut self, d1: usize, d2: usize) -> GResult<TensorView<'_>> {
        if d1 > self.size() {
            return Err(GError::DimOutOfRange {
                shape: self.dim().shape.clone(),
                dim: d1,
                op: "transpose",
            }
            .into());
        }
        if d2 > self.size() {
            return Err(GError::DimOutOfRange {
                shape: self.dim().shape.clone(),
                dim: d2,
                op: "transpose",
            }
            .into());
        }
        let new_dim = self.dim_mut().transpose(d1, d2)?;
        Ok(TensorView {
            dtype: self.dtype(),
            data: self.storage().view(),
            dim: new_dim,
            device: self.device().clone(),
        })
    }

    fn view<'a: 'b, 'b>(&'a self) -> TensorView<'b> {
        TensorView {
            dtype: self.dtype().clone(),
            data: self.storage().view(),
            dim: self.dim().clone(),
            device: self.device().clone(),
        }
    }

    fn ggml_shape(&self) -> &Layout {
        self.dim().shape_layout()
    }

    fn shape(&self) -> &[usize] {
        &self.dim().shape()
    }

    fn stride(&self) -> &[usize] {
        &self.dim().stride()
    }

    fn dim_3(&self) -> usize {
        self.dim().dim_3()
    }

    fn dim_2(&self) -> usize {
        self.dim().dim_2()
    }

    fn dim_1(&self) -> usize {
        self.dim().dim_1()
    }

    fn dim_0(&self) -> usize {
        self.dim().dim_0()
    }

    fn dim1(&self) -> usize {
        self.dim().dim1()
    }

    fn dim2(&self) -> (usize, usize) {
        self.dim().dim2()
    }

    fn dim3(&self) -> (usize, usize, usize) {
        self.dim().dim3()
    }

    fn dim4(&self) -> (usize, usize, usize, usize) {
        self.dim().dim4()
    }

    fn stride4(&self) -> (usize, usize, usize, usize) {
        self.dim().stride_4d()
    }

    fn n_dims(&self) -> usize {
        self.dim().n_dims()
    }

    fn elem_count(&self) -> usize {
        self.dim().elem_count()
    }

    fn is_contiguous(&self) -> bool {
        self.dim().is_contiguous()
    }

    fn is_vector(&self) -> bool {
        self.dim().is_vector()
    }

    fn ggml_is_contiguous(&self) -> bool {
        self.dim().ggml_is_contiguous()
    }

    fn elem_size(&self) -> usize {
        GS_TYPE_SIZE[self.dtype() as usize]
    }
}

pub trait StorageProto {
    fn as_bytes_mut(&mut self) -> &mut [u8];

    fn view(&self) -> StorageView<'_>;

    fn as_bytes(&self) -> &[u8];

    fn offset<'a>(&'a self, i: usize) -> StorageView<'a>;
}

#[derive(Clone)]
pub enum Device {
    Cpu,
    Gpu(CudaDevice),
}

impl Device {
    pub fn new_cuda(ordinal: usize) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum GGmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // // GGML_TYPE_Q4_2 = 4, support has been removed
    // // GGML_TYPE_Q4_3 (5) support has been removed
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    // k-quantizations
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    TypeCount = 19,
}

impl GGmlType {
    pub fn from_usize(u: usize) -> Self {
        match u {
            0 => GGmlType::F32,
            1 => GGmlType::F16,
            2 => GGmlType::Q4_0,
            3 => GGmlType::Q4_1,
            // 4 => GGmlType::Q4_2,
            // 5 => GGmlType::Q4_3,
            6 => GGmlType::Q5_0,
            7 => GGmlType::Q5_1,

            8 => GGmlType::Q8_0,
            9 => GGmlType::Q8_1,
            10 => GGmlType::Q2K,
            11 => GGmlType::Q3K,
            12 => GGmlType::Q4K,
            13 => GGmlType::Q5K,
            14 => GGmlType::Q6K,
            15 => GGmlType::Q8K,
            16 => GGmlType::I8,
            17 => GGmlType::I16,
            18 => GGmlType::I32,
            // 19 => GGmlType::F32,
            //1 => GGmlType::F32,
            _ => todo!(),
        }
    }
}

pub const GS_TYPE_SIZE: [usize; GGmlType::TypeCount as usize] = [
    std::mem::size_of::<f32>(),       //F32
    std::mem::size_of::<F16>(),       //F16
    std::mem::size_of::<BlockQ4_0>(), //Q4_0
    0,                                //Q4_1
    0,                                //Q4_2
    0,                                //Q4_3
    0,                                //Q5_0
    0,                                //Q5_1
    std::mem::size_of::<BlockQ8_0>(), //Q8_0
    0,                                //Q8_1
    0,                                //Q2K
    0,                                //Q3K
    0,                                //Q4K
    0,                                //Q5K
    std::mem::size_of::<BlockQ6K>(),  //Q6K
    0,                                //Q8K
    0,                                //I8
    0,                                //I16
    std::mem::size_of::<i32>(),       //i32
];

pub const GS_BLCK_SIZE: [usize; GGmlType::TypeCount as usize] = [
    1,     // F32
    1,     //F16
    QK4_0, //Q4_0
    QK4_0, //Q4_1
    1,     //Q4_2
    1,     //Q4_3
    1,     //Q5_0
    1,     //Q5_1
    QK8_0, //Q8_0
    1,     //Q8_1
    1,     //Q2K
    1,     //Q3K
    1,     //Q4K
    1,     //Q5K
    QK_K,  //Q6K
    1,     //Q8K
    1,     //I8
    1,     //I16
    1,     //i32
];

#[macro_export]
macro_rules! impl_tousize {
    ($($e:ident),*) => {
        $(impl ToUsize for $e {
            fn as_usize(&self) -> usize {
                *self as usize
            }
        })*
    };
}

#[macro_export]
macro_rules! impl_fromf32 {
    ($($e:ident),*) => {
        $(impl FromF32 for $e {
            fn from_f32(a: f32) -> Self {
                a as $e
            }
        })*
    };
}

#[macro_export]
macro_rules! impl_fromf64 {
    ($($e:ident),*) => {
        $(impl FromF64 for $e {
            fn from_f64(a: f64) -> Self {
                a as $e
            }
        })*
    };
}

impl FromF32 for f16 {
    fn from_f32(a: f32) -> Self {
        f16::from_f32(a)
    }
}

impl FromF64 for f16 {
    fn from_f64(a: f64) -> Self {
        f16::from_f64(a)
    }
}

impl_tousize!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_fromf32!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_fromf64!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
//impl_no_unary_op!(u8, u16, u32, u64, i8, i16, i32, i64);

impl TensorType for f16 {
    const DTYPE: GGmlType = GGmlType::F16;

    fn to_f32(&self) -> f32 {
        ToPrimitive::to_f32(self).unwrap_or(0.0)
    }

    fn to_f16(&self) -> f16 {
        *self
    }

    fn to_i32(&self) -> i32 {
        ToPrimitive::to_i32(self).unwrap_or(0)
    }

    fn to_usize(&self) -> usize {
        ToPrimitive::to_usize(self).unwrap_or(0)
    }

    fn from_f32(x: f32) -> Self {
        Self::from_f32(x)
    }

    fn from_x<X: TensorType>(x: X) -> Self {
        x.to_f16()
    }

    fn reduce_sum(chunk: &[Self]) -> Self {
        todo!()
    }

    fn vec_silu(n: usize, y: &mut [Self], x: &[Self]) {
        todo!()
    }

    fn vec_scale(n: usize, y: &mut [Self], v: Self) {
        todo!()
    }

    fn softmax(nc: usize, dp: &mut [Self], sp: &[Self]) {
        todo!()
    }

    fn rms_norm(ne00: usize, x: &[Self], y: &mut [Self], eps: f32) {
        todo!()
    }

    fn vec_add(inp1: &[Self], inp2: &[Self], dst: &mut [Self]) {
        todo!()
    }

    fn vec_mul(inp1: &[Self], inp2: &[Self], dst: &mut [Self]) {
        todo!()
    }
}

impl TensorType for f32 {
    const DTYPE: GGmlType = GGmlType::F32;

    fn to_f32(&self) -> f32 {
        *self
    }

    fn to_f16(&self) -> f16 {
        f16::from_f32(*self)
    }

    fn to_i32(&self) -> i32 {
        *self as i32
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }

    fn from_f32(x: f32) -> Self {
        x
    }

    fn from_x<X: TensorType>(x: X) -> Self {
        x.to_f32()
    }

    #[inline]
    fn reduce_sum(chunk: &[Self]) -> Self {
        let mut v = f32x32::from_slice(chunk);
        v *= v;
        v.reduce_sum()
    }

    #[inline]
    fn vec_silu(n: usize, y: &mut [Self], x: &[Self]) {
        for i in 0..n {
            y[i] = GLOBAL_CPU_DEVICE_CACHE
                .get_silu_cache(f16::from_f32(x[i]).to_bits() as usize)
                .to_f32()
        }
    }
    #[inline]
    fn vec_scale(n: usize, y: &mut [Self], v: Self) {
        let n32 = n & !(STEP - 1);
        let scale = std::simd::f32x32::splat(v);
        y[..n32].chunks_exact_mut(STEP).for_each(|d| {
            let va = std::simd::f32x32::from_slice(d);
            let va = va * scale;
            va.copy_to_slice(d);
        });
        for i in n32..n {
            y[i] *= v;
        }
    }

    fn softmax(nc: usize, dp: &mut [Self], sp: &[Self]) {
        let mut max = -std::f32::INFINITY;
        for i in 0..nc {
            max = max.max(sp[i]) //f32::max(max, );
        }

        let mut sum: f32 = 0.0;

        //  let ss: u16 = 0;
        for i in 0..nc {
            if sp[i] == -std::f32::INFINITY {
                dp[i] = 0.0;
            } else {
                //const float val = (S[i] == -INFINITY) ? 0.0 : exp(S[i] - max);
                let s = (sp[i] - max).to_f16();
                // let ss: u16 = unsafe { std::mem::transmute(s) };
                let val = GLOBAL_CPU_DEVICE_CACHE
                    .get_exp_cache(s.to_bits() as usize)
                    .to_f32();
                sum += val;
                dp[i] = val;
            }
        }

        assert!(sum > 0.0f32);

        sum = 1.0 / sum;
        Self::vec_scale(nc, dp, sum);
    }

    fn rms_norm(ne00: usize, x: &[Self], y: &mut [Self], eps: f32) {
        let n32 = ne00 & !(STEP - 1);
        let mut mean = x[..n32]
            .chunks_exact(STEP)
            .map(|chunk| Self::reduce_sum(chunk))
            .sum::<f32>();
        mean /= ne00 as f32;
        y[..ne00].copy_from_slice(&x[..ne00]);
        let scale = 1.0 / (mean + eps).sqrt();
        Self::vec_scale(ne00, y, scale);
    }

    fn vec_add(inp1: &[Self], inp2: &[Self], dst: &mut [Self]) {
        dst.par_chunks_exact_mut(4)
            .zip(inp1.par_chunks_exact(4))
            .zip(inp2.par_chunks_exact(4))
            .for_each(|((d, ia), ib)| {
                let va = std::simd::f32x4::from_slice(ia);
                let vb = std::simd::f32x4::from_slice(ib);
                let vc = va + vb;
                vc.copy_to_slice(d);
            });
        let dst_length = dst.len();
        // 处理剩余部分
        // let remainder_a = &inp1[inp1.len() / 4 * 4..];
        // let remainder_b = &inp2[inp2.len() / 4 * 4..];
        // let remainder_result = &mut dst[dst_length / 4 * 4..];
        // for i in 0..remainder_a.len() {
        //     remainder_result[i] = remainder_a[i] + remainder_b[i];
        // }

        dst[dst_length / 4 * 4..]
            .iter_mut()
            .zip(
                inp1[inp1.len() / 4 * 4..]
                    .iter()
                    .zip(inp2[inp2.len() / 4 * 4..].iter()),
            )
            .for_each(|(res, (a, b))| {
                *res = *a + *b;
            });
    }

    fn vec_mul(inp1: &[f32], inp2: &[f32], dst: &mut [f32]) {
        dst.par_chunks_exact_mut(4)
            .zip(inp1.par_chunks_exact(4))
            .zip(inp2.par_chunks_exact(4))
            .for_each(|((d, ia), ib)| {
                let va = std::simd::f32x4::from_slice(ia);
                let vb = std::simd::f32x4::from_slice(ib);
                let vc = va * vb;
                vc.copy_to_slice(d);
            });
        let dst_length = dst.len();
        // 处理剩余部分
        dst[dst_length / 4 * 4..]
            .iter_mut()
            .zip(
                inp1[inp1.len() / 4 * 4..]
                    .iter()
                    .zip(inp2[inp2.len() / 4 * 4..].iter()),
            )
            .for_each(|(res, (a, b))| {
                *res = *a * *b;
            });
    }
}

impl TensorType for i32 {
    const DTYPE: GGmlType = GGmlType::I32;

    fn to_f32(&self) -> f32 {
        *self as f32
    }

    fn to_f16(&self) -> f16 {
        f16::from_i32(*self).unwrap_or(f16::zero())
    }

    fn to_i32(&self) -> i32 {
        *self
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }

    fn from_f32(x: f32) -> Self {
        x as i32
    }

    fn from_x<X: TensorType>(x: X) -> Self {
        x.to_i32()
    }

    fn reduce_sum(chunk: &[Self]) -> Self {
        todo!()
    }

    fn vec_silu(n: usize, y: &mut [Self], x: &[Self]) {
        todo!()
    }
    fn vec_scale(n: usize, y: &mut [Self], v: Self) {
        todo!()
    }

    fn softmax(nc: usize, dp: &mut [Self], sp: &[Self]) {
        todo!()
    }

    fn rms_norm(ne00: usize, x: &[Self], y: &mut [Self], eps: f32) {
        todo!()
    }

    fn vec_add(inp1: &[Self], inp2: &[Self], dst: &mut [Self]) {
        todo!()
    }

    fn vec_mul(inp1: &[Self], inp2: &[Self], dst: &mut [Self]) {
        todo!()
    }
}

impl TensorType for BlockQ4_0 {
    const DTYPE: GGmlType = GGmlType::Q4_0;

    fn to_f32(&self) -> f32 {
        todo!()
    }

    fn from_f32(x: f32) -> Self {
        todo!()
    }

    fn from_x<X: TensorType>(x: X) -> Self {
        todo!()
    }

    fn to_f16(&self) -> f16 {
        todo!()
    }

    fn to_i32(&self) -> i32 {
        todo!()
    }

    fn to_usize(&self) -> usize {
        todo!()
    }

    fn reduce_sum(chunk: &[Self]) -> Self {
        todo!()
    }

    fn vec_silu(n: usize, y: &mut [Self], x: &[Self]) {
        todo!()
    }

    fn vec_scale(n: usize, y: &mut [Self], v: Self) {
        todo!()
    }

    fn softmax(nc: usize, dp: &mut [Self], sp: &[Self]) {
        todo!()
    }

    fn rms_norm(ne00: usize, x: &[Self], y: &mut [Self], eps: f32) {
        todo!()
    }

    fn vec_add(inp1: &[Self], inp2: &[Self], dst: &mut [Self]) {
        todo!()
    }

    fn vec_mul(inp1: &[Self], inp2: &[Self], dst: &mut [Self]) {
        todo!()
    }
}

impl TensorType for BlockQ6K {
    const DTYPE: GGmlType = GGmlType::Q6K;

    fn to_f32(&self) -> f32 {
        todo!()
    }

    fn from_f32(x: f32) -> Self {
        todo!()
    }

    fn from_x<X: TensorType>(x: X) -> Self {
        todo!()
    }

    fn to_f16(&self) -> f16 {
        todo!()
    }

    fn to_i32(&self) -> i32 {
        todo!()
    }

    fn to_usize(&self) -> usize {
        todo!()
    }

    fn reduce_sum(chunk: &[Self]) -> Self {
        todo!()
    }

    fn vec_silu(n: usize, y: &mut [Self], x: &[Self]) {
        todo!()
    }

    fn vec_scale(n: usize, y: &mut [Self], v: Self) {
        todo!()
    }

    fn softmax(nc: usize, dp: &mut [Self], sp: &[Self]) {
        todo!()
    }

    fn rms_norm(ne00: usize, x: &[Self], y: &mut [Self], eps: f32) {
        todo!()
    }

    fn vec_add(inp1: &[Self], inp2: &[Self], dst: &mut [Self]) {
        todo!()
    }

    fn vec_mul(inp1: &[Self], inp2: &[Self], dst: &mut [Self]) {
        todo!()
    }
}

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
    //std::cmp::PartialOrd
     fmt::Debug
    + PartialEq
    + Copy
   // + num_traits::NumAssign
    + Sync
    + Send
    //+ ToUsize
    // + FromF32
    // + FromF64
   // + UnaryOp
    + 'static
{
    const DTYPE: GGmlType;
    fn zeros() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }

    fn byte_size()->usize{
        GS_TYPE_SIZE[Self::DTYPE as usize]
    }

    fn blck_size()->usize{
        GS_BLCK_SIZE[Self::DTYPE as usize]
    }

    fn to_f32(&self)->f32;

    fn to_f16(&self)->f16;

    fn to_i32(&self)->i32;

    fn to_usize(&self)->usize;

    fn from_f32(x:f32)-> Self ;

    fn from_x<X:TensorType>(x:X)-> Self;

    fn reduce_sum(chunk: &[Self])->Self;

    fn vec_silu(n: usize, y: &mut [Self], x: &[Self]);

    fn vec_scale(n: usize, y: &mut [Self], v: Self);

    fn softmax(nc: usize, dp: &mut [Self], sp: &[Self]);

    fn rms_norm(ne00: usize, x: &[Self], y: &mut [Self], eps: f32) ;

    fn vec_add(inp1: &[Self], inp2: &[Self], dst: &mut [Self]);

    fn vec_mul(inp1: &[Self], inp2: &[Self], dst: &mut [Self]);

    fn is_quantized(&self)->bool{
        match Self::DTYPE { 
            GGmlType::Q4_0=>true,
            GGmlType::Q4_1=>true,
            GGmlType::Q5_0=>true,
            GGmlType::Q5_1=>true,
            GGmlType::Q8_0=>true,
            GGmlType::Q8_1=>true,
            GGmlType::Q2K=>true,
            GGmlType::Q3K=>true,
            GGmlType::Q4K=>true,
            GGmlType::Q5K=>true,
            GGmlType::Q6K =>true,
            GGmlType::Q8K=>true,  
            _=>false,
        }
    }

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

trait NdArray {
    type device;
    fn to_cpu_storage(self) -> Self::device;

    fn to_gpu_storage(self);
}

impl<A: TensorType> NdArray for Vec<A> {
    type device = CpuStorageSlice;
    fn to_cpu_storage(self) -> Self::device {
        let device = match A::DTYPE {
            GGmlType::F16 => {
                let raw = RawSlice::from_raw_parts(
                    self.as_ptr() as *mut f16,
                    self.len(),
                    self.capacity(),
                );
                CpuStorageSlice::F16(raw)
            }
            GGmlType::F32 => {
                let raw = RawSlice::from_raw_parts(
                    self.as_ptr() as *mut f32,
                    self.len(),
                    self.capacity(),
                );
                CpuStorageSlice::F32(raw)
            }
            _ => {
                todo!()
            }
        };
        forget(self);
        return device;
    }

    fn to_gpu_storage(self) {
        todo!();
    }
}

impl<'a, A: TensorType> NdArray for &'a [A] {
    type device = CpuStorageView<'a>;
    fn to_cpu_storage(self) -> Self::device {
        match A::DTYPE {
            GGmlType::F16 => {
                let raw = RawSliceView::from_raw_parts(self.as_ptr() as *mut f16, self.len());
                CpuStorageView::F16(raw)
            }
            GGmlType::F32 => {
                let raw = RawSliceView::from_raw_parts(self.as_ptr() as *mut f32, self.len());
                CpuStorageView::F32(raw)
            }
            // GGmlType::F64 => {
            //     let raw = RawRef::from_raw_parts(self.as_ptr() as *mut f64, self.len());
            //     CpuStorageSlice::F64(RawSlice::Ref(raw))
            // }
            _ => {
                todo!()
            }
        }
    }

    fn to_gpu_storage(self) {
        todo!();
    }
}

impl<A: TensorType, const N: usize> NdArray for Vec<[A; N]> {
    type device = CpuStorageSlice;
    fn to_cpu_storage(self) -> CpuStorageSlice {
        let device = match A::DTYPE {
            GGmlType::F16 => {
                let raw = RawSlice::from_raw_parts(
                    self.as_ptr() as *mut f16,
                    self.len() * N,
                    self.capacity(),
                );
                CpuStorageSlice::F16(raw)
            }
            GGmlType::F32 => {
                let raw = RawSlice::from_raw_parts(
                    self.as_ptr() as *mut f32,
                    self.len() * N,
                    self.capacity(),
                );
                CpuStorageSlice::F32(raw)
            }
            // GGmlType::F64 => {
            //     let raw = RawPtr::from_raw_parts(
            //         self.as_ptr() as *mut f64,
            //         self.len() * N,
            //         self.capacity(),
            //     );
            //     CpuStorageSlice::F64(RawSlice::Own(raw))
            // }
            _ => {
                todo!()
            }
        };
        forget(self);
        return device;
    }

    fn to_gpu_storage(self) {
        todo!();
    }
}

impl<A: TensorType, const N: usize, const M: usize> NdArray for Vec<[[A; N]; M]> {
    type device = CpuStorageSlice;
    fn to_cpu_storage(self) -> CpuStorageSlice {
        let device = match A::DTYPE {
            GGmlType::F16 => {
                let raw = RawSlice::from_raw_parts(
                    self.as_ptr() as *mut f16,
                    self.len() * N * M,
                    self.capacity(),
                );
                CpuStorageSlice::F16(raw)
            }
            GGmlType::F32 => {
                let raw = RawSlice::from_raw_parts(
                    self.as_ptr() as *mut f32,
                    self.len() * N * M,
                    self.capacity(),
                );
                CpuStorageSlice::F32(raw)
            }
            // GGmlType::F64 => {
            //     let raw = RawPtr::from_raw_parts(
            //         self.as_ptr() as *mut f64,
            //         self.len() * N * M,
            //         self.capacity(),
            //     );
            //     CpuStorageSlice::F64(RawSlice::Own(raw))
            // }
            _ => {
                todo!()
            }
        };
        forget(self);
        return device;
    }

    fn to_gpu_storage(self) {
        todo!()
    }
}

#[derive(Clone)]
pub(crate) struct RawSliceView<'a, P: TensorType> {
    ptr: NonNull<P>,
    len: usize,
    marker: PhantomData<&'a mut [P]>,
}

impl<'a, P: TensorType> RawSliceView<'a, P> {
    pub(crate) fn from_bytes(raw: &[u8]) -> Self {
        let nbytes = raw.len();
        assert_eq!(
            nbytes % GS_TYPE_SIZE[P::DTYPE as usize] / GS_BLCK_SIZE[P::DTYPE as usize],
            0,
            "Length of slice must be multiple of f32 size"
        );
        let data = RawSliceView::from_raw_parts(
            raw.as_ptr() as *mut P,
            nbytes / GS_TYPE_SIZE[P::DTYPE as usize],
        );
        data
    }

    pub(crate) fn offset(&self, i: usize) -> RawSliceView<'a, P> {
        RawSliceView {
            ptr: unsafe { self.ptr.offset(i as isize) },
            len: self.len - i,
            marker: PhantomData,
        }
    }

    pub(crate) fn as_ref(&self) -> RawSliceView<'a, P> {
        RawSliceView {
            ptr: self.ptr,
            len: self.len,
            marker: PhantomData,
        }
    }

    fn from_raw_parts(ptr: *mut P, length: usize) -> Self {
        unsafe {
            Self {
                ptr: NonNull::new_unchecked(ptr),
                len: length,
                marker: PhantomData,
            }
        }
    }

    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr.as_ptr() as *const u8,
                self.len * GS_TYPE_SIZE[P::DTYPE as usize],
            )
        }
    }

    fn as_bytes_mut(&self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.as_ptr() as *mut u8,
                self.len * GS_TYPE_SIZE[P::DTYPE as usize],
            )
        }
    }

    fn as_slice(&self) -> &[P] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const P, self.len) }
    }

    fn as_slice_mut(&mut self) -> &mut [P] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut P, self.len) }
    }
}

// #[derive(Clone)]
pub(crate) struct RawSlice<P: TensorType> {
    ptr: NonNull<P>,
    len: usize,
    cap: usize,
}

impl<P: TensorType> RawSlice<P> {
    pub(crate) fn from_raw(raw: Vec<u8>) -> Self {
        let nbytes = raw.len();
        assert_eq!(
            nbytes % GS_TYPE_SIZE[P::DTYPE as usize] / GS_BLCK_SIZE[P::DTYPE as usize],
            0,
            "Length of slice must be multiple of f32 size"
        );
        let data = RawSlice::from_raw_parts(
            raw.as_ptr() as *mut P,
            nbytes / GS_TYPE_SIZE[P::DTYPE as usize],
            raw.capacity() / GS_TYPE_SIZE[P::DTYPE as usize],
        );
        forget(raw);
        data
    }

    pub(crate) fn as_ref<'a>(&'a self) -> RawSliceView<'a, P> {
        RawSliceView {
            ptr: self.ptr,
            len: self.len,
            marker: PhantomData,
        }
    }

    pub(crate) fn offset<'a>(&'a self, i: usize) -> RawSliceView<'a, P> {
        RawSliceView {
            ptr: unsafe { self.ptr.offset(i as isize) },
            len: self.len - i,
            marker: PhantomData,
        }
    }

    // pub(crate) fn as_ptr(&self) -> NonNull<P> {
    //     return match self {
    //         RawSlice::Own(v) => v.ptr,
    //         RawSlice::Ref(v) => v.ptr,
    //     };
    // }

    // pub(crate) fn len(&self) -> usize {
    //     return match self {
    //         RawSlice::Own(v) => v.len,
    //         RawSlice::Ref(v) => v.len,
    //     };
    // }

    // fn as_bytes_mut(&mut self) -> &mut [u8] {
    //     return match self {
    //         RawSlice::Own(v) => v.as_bytes_mut(),
    //         RawSlice::Ref(v) => v.as_bytes_mut(),
    //     };
    // }

    // fn as_bytes(&self) -> &[u8] {
    //     return match self {
    //         RawSlice::Own(v) => v.as_bytes(),
    //         RawSlice::Ref(v) => v.as_bytes(),
    //     };
    // }

    // fn as_slice(&self) -> &[P] {
    //     return match self {
    //         RawSlice::Own(v) => v.as_slice(),
    //         RawSlice::Ref(v) => v.as_slice(),
    //     };
    // }

    // fn as_slice_mut(&mut self) -> &mut [P] {
    //     return match self {
    //         RawSlice::Own(v) => v.as_slice_mut(),
    //         RawSlice::Ref(v) => v.as_slice_mut(),
    //     };
    // }
}

#[derive(Clone)]
struct RawRef<P: TensorType> {
    ptr: NonNull<P>,
    len: usize,
}

impl<P: TensorType> RawRef<P> {
    fn as_slice(&self) -> &[P] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const P, self.len) }
    }

    fn as_slice_mut(&mut self) -> &mut [P] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut P, self.len) }
    }

    fn as_bytes_mut(&self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.as_ptr() as *mut u8,
                self.len * GS_TYPE_SIZE[P::DTYPE as usize],
            )
        }
    }

    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr.as_ptr() as *const u8,
                self.len * GS_TYPE_SIZE[P::DTYPE as usize],
            )
        }
    }

    pub(crate) fn as_ptr(&self) -> NonNull<P> {
        self.ptr
    }

    // fn to_RawSlice(self) -> RawSlice<P> {
    //     RawSlice::Ref(self)
    // }

    fn from_raw_parts(ptr: *mut P, length: usize) -> Self {
        unsafe {
            Self {
                ptr: NonNull::new_unchecked(ptr),
                len: length,
            }
        }
    }

    // fn nbytes(&self) -> usize {
    //     self.len * GS_TYPE_SIZE[P::DTYPE as usize] / GS_BLCK_SIZE[P::DTYPE as usize]
    // }
}

// struct RawPtr<P: TensorType> {
//     ptr: NonNull<P>,
//     len: usize,
//     cap: usize,
// }

impl<A: TensorType> NdArray for RawSlice<A> {
    type device = CpuStorageSlice;
    fn to_cpu_storage(self) -> Self::device {
        let device = match A::DTYPE {
            GGmlType::F16 => {
                let raw =
                    RawSlice::from_raw_parts(self.as_ptr() as *mut f16, self.len(), self.cap());
                CpuStorageSlice::F16(raw)
            }
            GGmlType::F32 => {
                let raw =
                    RawSlice::from_raw_parts(self.as_ptr() as *mut f32, self.len(), self.cap());
                CpuStorageSlice::F32(raw)
            }
            _ => {
                todo!()
            }
        };
        forget(self);
        return device;
    }

    fn to_gpu_storage(self) {
        todo!()
    }
}

impl<P: TensorType> Clone for RawSlice<P> {
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

impl<P: TensorType> RawSlice<P> {
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

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.as_ptr() as *mut u8,
                self.len * std::mem::size_of::<P>(),
            )
        }
    }

    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr.as_ptr() as *const u8,
                self.len * std::mem::size_of::<P>(),
            )
        }
    }

    fn nbytes(&self) -> usize {
        self.len * std::mem::size_of::<P>()
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

    // fn to_RawSlice(self) -> RawSlice<P> {
    //     RawSlice::Own(self)
    // }
}

#[derive(Debug)]
pub enum TensorItem<'a> {
    F16(&'a mut f16),
    F32(&'a mut f32),
    F64(&'a mut f64),
}

pub struct TensorIter<'a> {
    n_dims: usize,
    device: &'a Storage,
    strides: [usize; MAX_DIM],
    shape_iter: ShapeIter<'a>,
}

impl<'a> Iterator for TensorIter<'a> {
    type Item = TensorItem<'a>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.shape_iter.next(self.n_dims)?;
        let offset = Shape::stride_offset(index.dims(self.n_dims), &self.strides[..self.n_dims]);
        match self.device {
            Storage::Cpu(cpu) => match cpu {
                CpuStorageSlice::F16(v) => {
                    let ptr = unsafe { v.as_ptr().offset(offset) } as *mut f16;
                    unsafe { Some(TensorItem::F16(&mut *ptr)) }
                }
                CpuStorageSlice::F32(v) => {
                    let ptr = unsafe { v.as_ptr().offset(offset) } as *mut f32;
                    unsafe { Some(TensorItem::F32(&mut *ptr)) }
                }
                // CpuStorageSlice::F64(v) => {
                //     let ptr = unsafe { v.as_ptr().offset(offset) }.as_ptr();
                //     unsafe { Some(TensorItem::F64(&mut *ptr)) }
                // }
                _ => {
                    todo!()
                }
            },
            Storage::Gpu(_) => {
                todo!()
            }
        }
    }
}

impl<'a> TensorIter<'a> {
    fn zip2(self, t: TensorIter<'a>) -> Zip<'a>
    where
        Self: Sized,
    {
        Zip::new(self, t)
    }
}

#[derive(Clone)]
pub enum CpuStorageView<'a> {
    Q4_0(RawSliceView<'a, BlockQ4_0>),
    Q6K(RawSliceView<'a, BlockQ6K>),
    F16(RawSliceView<'a, f16>),
    F32(RawSliceView<'a, f32>),
    I32(RawSliceView<'a, i32>),
}

impl<'a> CpuStorageView<'a> {
    pub(crate) fn view(&self) -> CpuStorageView<'a> {
        return match self {
            CpuStorageView::Q4_0(v) => CpuStorageView::Q4_0(v.as_ref()),
            CpuStorageView::Q6K(v) => CpuStorageView::Q6K(v.as_ref()),
            CpuStorageView::F16(v) => CpuStorageView::F16(v.as_ref()),
            CpuStorageView::F32(v) => CpuStorageView::F32(v.as_ref()),
            CpuStorageView::I32(v) => CpuStorageView::I32(v.as_ref()),
        };
    }

    pub(crate) fn offset(&self, i: usize) -> CpuStorageView<'a> {
        return match self {
            CpuStorageView::Q4_0(v) => CpuStorageView::Q4_0(v.offset(i)),
            CpuStorageView::Q6K(v) => CpuStorageView::Q6K(v.offset(i)),
            CpuStorageView::F16(v) => CpuStorageView::F16(v.offset(i)),
            CpuStorageView::F32(v) => CpuStorageView::F32(v.offset(i)),
            CpuStorageView::I32(v) => CpuStorageView::I32(v.offset(i)),
        };
    }

    pub(crate) fn as_bytes_mut(&mut self) -> &mut [u8] {
        return match self {
            CpuStorageView::Q4_0(v) => v.as_bytes_mut(),
            CpuStorageView::Q6K(v) => v.as_bytes_mut(),
            CpuStorageView::F16(v) => v.as_bytes_mut(),
            CpuStorageView::F32(v) => v.as_bytes_mut(),
            CpuStorageView::I32(v) => v.as_bytes_mut(),
        };
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        return match self {
            CpuStorageView::Q4_0(v) => v.as_bytes(),
            CpuStorageView::Q6K(v) => v.as_bytes(),
            CpuStorageView::F16(v) => v.as_bytes(),
            CpuStorageView::F32(v) => v.as_bytes(),
            CpuStorageView::I32(v) => v.as_bytes(),
        };
    }
}

#[derive(Clone)]
pub enum CpuStorageSlice {
    Q4_0(RawSlice<BlockQ4_0>),
    Q6K(RawSlice<BlockQ6K>),
    F16(RawSlice<f16>),
    F32(RawSlice<f32>),
    I32(RawSlice<i32>),
}

impl CpuStorageSlice {
    pub(crate) fn from_bytes(raw: &[u8], dtype: GGmlType) {}

    pub(crate) fn offset<'a>(&'a self, i: usize) -> CpuStorageView<'a> {
        return match self {
            CpuStorageSlice::Q4_0(v) => CpuStorageView::Q4_0(v.offset(i)),
            CpuStorageSlice::Q6K(v) => CpuStorageView::Q6K(v.offset(i)),
            CpuStorageSlice::F16(v) => CpuStorageView::F16(v.offset(i)),
            CpuStorageSlice::F32(v) => CpuStorageView::F32(v.offset(i)),
            CpuStorageSlice::I32(v) => CpuStorageView::I32(v.offset(i)),
        };
    }

    pub(crate) fn view<'a>(&'a self) -> CpuStorageView<'a> {
        return match self {
            CpuStorageSlice::Q4_0(v) => CpuStorageView::Q4_0(v.as_ref()),
            CpuStorageSlice::Q6K(v) => CpuStorageView::Q6K(v.as_ref()),
            CpuStorageSlice::F16(v) => CpuStorageView::F16(v.as_ref()),
            CpuStorageSlice::F32(v) => CpuStorageView::F32(v.as_ref()),
            CpuStorageSlice::I32(v) => CpuStorageView::I32(v.as_ref()),
        };
    }

    pub(crate) fn as_bytes_mut(&mut self) -> &mut [u8] {
        return match self {
            CpuStorageSlice::Q4_0(v) => v.as_bytes_mut(),
            CpuStorageSlice::Q6K(v) => v.as_bytes_mut(),
            CpuStorageSlice::F16(v) => v.as_bytes_mut(),
            CpuStorageSlice::F32(v) => v.as_bytes_mut(),
            // CpuStorageSlice::F64(v) => v.as_bytes_mut(),
            CpuStorageSlice::I32(v) => v.as_bytes_mut(),
        };
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        return match self {
            CpuStorageSlice::Q4_0(v) => v.as_bytes(),
            CpuStorageSlice::Q6K(v) => v.as_bytes(),
            CpuStorageSlice::F16(v) => v.as_bytes(),
            CpuStorageSlice::F32(v) => v.as_bytes(),
            CpuStorageSlice::I32(v) => v.as_bytes(),
        };
    }

    // pub(crate) fn as_ref(&self) -> CpuStorageSlice {
    //     return match self {
    //         CpuStorageSlice::Q4_0(v) => CpuStorageSlice::Q4_0(v.as_ref()),
    //         CpuStorageSlice::Q6K(v) => CpuStorageSlice::Q6K(v.as_ref()),
    //         CpuStorageSlice::F16(v) => CpuStorageSlice::F16(v.as_ref()),
    //         CpuStorageSlice::F32(v) => CpuStorageSlice::F32(v.as_ref()),
    //         //  CpuStorageSlice::F64(v) => CpuStorageSlice::F64(v.as_ref()),
    //         CpuStorageSlice::I32(v) => CpuStorageSlice::I32(v.as_ref()),
    //     };
    // }

    pub(crate) fn len(&self) -> usize {
        return match self {
            CpuStorageSlice::Q4_0(v) => v.len(),
            CpuStorageSlice::Q6K(v) => v.len(),
            CpuStorageSlice::F16(v) => v.len(),
            CpuStorageSlice::F32(v) => v.len(),
            //CpuStorageSlice::F64(v) => v.len(),
            CpuStorageSlice::I32(v) => v.len(),
        };
    }

    pub(crate) fn copy_strided_src(&self, rhs: &mut Self, dst_offset: usize, src_l: &Dim) {
        match (self, rhs) {
            (CpuStorageSlice::F16(l), CpuStorageSlice::F16(r)) => {
                copy_strided_src(l.as_slice(), r.as_slice_mut(), dst_offset, src_l);
            }
            (CpuStorageSlice::F32(l), CpuStorageSlice::F32(r)) => {
                copy_strided_src(l.as_slice(), r.as_slice_mut(), dst_offset, src_l);
            }
            // (CpuStorageSlice::F64(l), CpuStorageSlice::F64(r)) => {
            //     copy_strided_src(l.as_slice(), r.as_slice_mut(), dst_offset, src_l);
            // }
            _ => {
                todo!()
            }
        }
    }
}

// enum StorageViewMut<'a> {
//     Cpu(CpuStorageViewMut<'a>),
//     Gpu(), //todo!
// }

impl<'a> AsRef<StorageView<'a>> for StorageView<'a> {
    fn as_ref(&self) -> &StorageView<'a> {
        &self
    }
}

impl<'a> StorageProto for StorageView<'a> {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        return match self {
            StorageView::Cpu(v) => v.as_bytes_mut(),
            StorageView::Gpu(v) => {
                todo!()
            }
        };
    }

    fn view(&self) -> StorageView<'a> {
        return match self {
            StorageView::Cpu(v) => StorageView::Cpu(v.clone()),
            StorageView::Gpu(g) => {
                StorageView::Gpu(g.offset(0))
            
            }
        };
    }

    fn as_bytes(&self) -> &[u8] {
        return match self {
            StorageView::Cpu(v) => v.as_bytes(),
            StorageView::Gpu(_) => {
                todo!()
            }
        };
    }

    fn offset(&self, i: usize) -> StorageView<'a> {
        return match self {
            StorageView::Cpu(v) => StorageView::Cpu(v.offset(i)),
            StorageView::Gpu(_) => {
                todo!()
            }
        };
    }
}

pub enum StorageView<'a> {
    Cpu(CpuStorageView<'a>),
    Gpu(CudaStorageView<'a>), //todo!
}

// impl<'a> StorageView<'a> {
//     pub(crate) fn as_bytes_mut(&mut self) -> &mut [u8] {
//         return match self {
//             StorageView::Cpu(v) => v.as_bytes_mut(),
//             StorageView::Gpu(v) => {
//                 todo!()
//             }
//         };
//     }

//     pub(crate) fn view(&self) -> StorageView<'a> {
//         return match self {
//             StorageView::Cpu(v) => StorageView::Cpu(v.clone()),
//             StorageView::Gpu(_) => {
//                 todo!()
//             }
//         };
//     }

//     pub(crate) fn as_bytes(&self) -> &[u8] {
//         return match self {
//             StorageView::Cpu(v) => v.as_bytes(),
//             StorageView::Gpu(_) => {
//                 todo!()
//             }
//         };
//     }

//     pub(crate) fn offset(&self, i: usize) -> StorageView<'a> {
//         return match self {
//             StorageView::Cpu(v) => StorageView::Cpu(v.offset(i)),
//             StorageView::Gpu(_) => {
//                 todo!()
//             }
//         };
//     }
// }

// impl<'a> AsRef<StorageView<'a>> for Storage {
//     fn as_ref(&self) -> &StorageView<'a> {
//         self
//     }
// }

pub enum Storage {
    Cpu(CpuStorageSlice),
    Gpu(CudaStorage),
}

impl StorageProto for Storage {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        return match self {
            Storage::Cpu(v) => v.as_bytes_mut(),
            Storage::Gpu(_) => {
                todo!()
            }
        };
    }

    fn view<'a>(&'a self) -> StorageView<'a> {
        return match self {
            Storage::Cpu(v) => StorageView::Cpu(v.view()),
            Storage::Gpu(v) => StorageView::Gpu(v.view()),
        };
    }

    fn as_bytes(&self) -> &[u8] {
        return match self {
            Storage::Cpu(v) => v.as_bytes(),
            Storage::Gpu(_) => {
                todo!()
            }
        };
    }

    fn offset<'a>(&'a self, i: usize) -> StorageView<'a> {
        return match self {
            Storage::Cpu(v) => StorageView::Cpu(v.offset(i)),
            Storage::Gpu(g) => {
                StorageView::Gpu(g.offset(i))
            }
        };
    }

    // pub(crate) fn len(&self) -> usize {
    //     return match self {
    //         Storage::Cpu(v) => v.len(),
    //         Storage::Gpu() => {
    //             todo!()
    //         }
    //     };
    // }

    // pub(crate) fn copy_strided_src(&self, rhs: &mut Self, dst_offset: usize, src_l: &Dim) {
    //     match (self, rhs) {
    //         (Storage::Cpu(lhs), Storage::Cpu(rhs)) => lhs.copy_strided_src(rhs, dst_offset, src_l),
    //         (Storage::Gpu(), Storage::Gpu()) => {
    //             todo!()
    //         }
    //         _ => {
    //             todo!()
    //         }
    //     }
    // }

    // pub(crate) fn sqrt(&self) -> Self {
    //     match self {
    //         Storage::Cpu(lhs) => {
    //             todo!()
    //         }
    //         Storage::Gpu() => {
    //             todo!()
    //         }
    //         _ => {
    //             todo!()
    //         }
    //     }
    // }
    // fn as_slice(&self) -> &[A] {
    //     return match self {
    //         Storage::Cpu(v) => v.as_slice(),
    //         Storage::Gpu() => {
    //             todo!()
    //         }
    //     };
    // }

    // fn as_slice_mut(&mut self) -> &mut [A] {
    //     return match self {
    //         Storage::Cpu(v) => v.as_slice_mut(),
    //         Storage::Gpu() => {
    //             todo!()
    //         }
    //     };a
    // }
}

impl<'a> TensorProto for TensorView<'a> {
    type Sto = StorageView<'a>;

    fn dtype(&self) -> GGmlType {
        self.dtype
    }

    fn storage<'b>(&'b self) -> &'b Self::Sto {
        &self.data
    }

    fn storage_mut<'b>(&'b mut self) -> &'b mut Self::Sto {
        &mut self.data
    }

    fn dim(&self) -> &Dim {
        &self.dim
    }

    fn dim_mut(&mut self) -> &mut Dim {
        &mut self.dim
    }

    fn device(&self) -> &Device {
        &self.device
    }

    unsafe fn from_bytes(
        v: &[u8],
        n_dims: usize,
        s: Shape,
        dtype: GGmlType,
        dev: &Device,
    ) -> GResult<Self> {
        match dtype {
            GGmlType::Q4_0 => Ok(TensorView::from_storage(
                StorageView::Cpu(CpuStorageView::Q4_0(RawSliceView::from_bytes(v))),
                n_dims,
                s,
                dtype,
                dev.clone(),
            )),
            GGmlType::Q6K => Ok(TensorView::from_storage(
                StorageView::Cpu(CpuStorageView::Q6K(RawSliceView::from_bytes(v))),
                n_dims,
                s,
                dtype,
                dev.clone(),
            )),
            GGmlType::F16 => Ok(TensorView::from_storage(
                StorageView::Cpu(CpuStorageView::F16(RawSliceView::from_bytes(v))),
                n_dims,
                s,
                dtype,
                dev.clone(),
            )),
            GGmlType::F32 => Ok(TensorView::from_storage(
                StorageView::Cpu(CpuStorageView::F32(RawSliceView::from_bytes(v))),
                n_dims,
                s,
                dtype,
                dev.clone(),
            )),
            // GGmlType::F64 => Tensor::from_storage(
            //     Storage::Cpu(CpuStorageSlice::F64(RawSlice::from_bytes(v))),
            //     n_dims,
            //     s,
            //     dtype,
            // ),
            GGmlType::I32 => Ok(TensorView::from_storage(
                StorageView::Cpu(CpuStorageView::I32(RawSliceView::from_bytes(v))),
                n_dims,
                s,
                dtype,
                dev.clone(),
            )),
            _ => {
                println!("dtype not found {:?}", dtype);
                todo!()
            }
        }
    }
}

impl TensorProto for Tensor {
    type Sto = Storage;

    fn dtype(&self) -> GGmlType {
        self.dtype
    }

    fn storage<'b>(&'b self) -> &'b Self::Sto {
        &self.data
    }

    fn storage_mut<'b>(&'b mut self) -> &'b mut Self::Sto {
        &mut self.data
    }

    fn dim(&self) -> &Dim {
        &self.dim
    }

    fn dim_mut(&mut self) -> &mut Dim {
        &mut self.dim
    }

    fn device(&self) -> &Device {
        &self.dev
    }

    unsafe fn from_bytes(
        v: &[u8],
        n_dims: usize,
        s: Shape,
        dtype: GGmlType,
        dev: &Device,
    ) -> GResult<Self> {
        match dtype {
            GGmlType::Q4_0 => Tensor::from_cpu_storage(
                CpuStorageSlice::Q4_0(RawSlice::from_raw(v.to_vec())),
                n_dims,
                s,
                dtype,
                dev,
            ),
            GGmlType::Q6K => Tensor::from_cpu_storage(
                CpuStorageSlice::Q6K(RawSlice::from_raw(v.to_vec())),
                n_dims,
                s,
                dtype,
                dev,
            ),
            GGmlType::F16 => Tensor::from_cpu_storage(
                CpuStorageSlice::F16(RawSlice::from_raw(v.to_vec())),
                n_dims,
                s,
                dtype,
                dev,
            ),
            GGmlType::F32 => Tensor::from_cpu_storage(
                CpuStorageSlice::F32(RawSlice::from_raw(v.to_vec())),
                n_dims,
                s,
                dtype,
                dev,
            ),
            // GGmlType::F64 => Tensor::from_storage(
            //     Storage::Cpu(CpuStorageSlice::F64(RawSlice::from_bytes(v))),
            //     n_dims,
            //     s,
            //     dtype,
            // ),
            GGmlType::I32 => Tensor::from_cpu_storage(
                CpuStorageSlice::I32(RawSlice::from_raw(v.to_vec())),
                n_dims,
                s,
                dtype,
                dev,
            ),
            _ => {
                println!("dtype not found {:?}", dtype);
                todo!()
            }
        }
    }
}

unsafe impl<'a> Send for TensorView<'a> {}

pub struct TensorView<'a> {
    dtype: GGmlType,
    data: StorageView<'a>,
    dim: Dim,
    device: Device,
}

impl<'a> TensorView<'a> {  
    fn from_storage(
        data: StorageView<'a>,
        n_dims: usize,
        shape: Shape,
        dtype: GGmlType,
        device: Device,
    ) -> Self {
        let stride = shape.ggml_stride(dtype);
        Self {
            dtype: dtype,
            data: data,
            dim: Dim {
                n_dims,
                shape,
                stride,
            },
            device,
        }
    }

    pub fn from_slice<A: TensorType>(v: &'a [A], n_dims: usize, s: Shape, dev: &Device) -> Self {
        let cpu_dev = v.to_cpu_storage();
        TensorView::from_storage(StorageView::Cpu(cpu_dev), n_dims, s, A::DTYPE, dev.clone())
    }

    pub fn to_cuda_tensor(&self, dev: &Device)-> GResult<Tensor> {  
        match dev {
            Device::Cpu => {
                todo!()
            }
            Device::Gpu(g) => {
                let view = self.data.view();
                match view {
                    StorageView::Cpu(cpu_stor) => {
                        let cuda_stor = g.from_cpu_storage_view(cpu_stor)?;
                       Ok(Tensor{
                            dtype: self.dtype,
                            data: Storage::Gpu(cuda_stor),
                            dim: self.dim.clone(),
                            dev: dev.clone()
                        }) 
                    }
                    StorageView::Gpu(_) => {
                        todo!()
                    }

                }            
            }
        }
  
    }


    pub fn to_cpu_tensor(self) -> GResult<Tensor> {
        match &self.data {
            StorageView::Cpu(_) => todo!(),
            StorageView::Gpu(g) => match g.slice() {
                CudaStorageSliceView::Q4_0(v) => {
                    let elem_count = self.dim.elem_count();
                    let mut cpu_vec = vec![BlockQ4_0::default(); elem_count];
                    g.device().device.dtoh_sync_copy_into(&v.slice(..elem_count), &mut cpu_vec)?;
                    Tensor::from_vec(
                        cpu_vec,
                        self.n_dims(),
                        Shape::from_slice(self.shape()),
                        &Device::Cpu,
                    )
                }
                CudaStorageSliceView::F16(v) => {
                    let elem_count = self.dim.elem_count();
                    let mut cpu_vec = vec![f16::zero(); elem_count];
                    let t = v.try_slice(..elem_count).unwrap();
                    g.device().device.dtoh_sync_copy_into(&t, &mut cpu_vec).unwrap();
                    Tensor::from_vec(  
                        cpu_vec,
                        self.n_dims(),
                        Shape::from_slice(self.shape()),
                        &Device::Cpu,
                    )
                }
                CudaStorageSliceView::F32(v) => {
                    let mut cpu_vec = vec![0.0f32; self.dim.elem_count()];
                    g.device().device.dtoh_sync_copy_into(v, &mut cpu_vec)?;
                    Tensor::from_vec(
                        cpu_vec,
                        self.n_dims(),
                        Shape::from_slice(self.shape()),
                        &Device::Cpu,
                    )
                }
                CudaStorageSliceView::I32(v) => {
                    let mut cpu_vec = vec![0; self.dim.elem_count()];
                    g.device().device.dtoh_sync_copy_into(v, &mut cpu_vec)?;
                    Tensor::from_vec(
                        cpu_vec,
                        self.n_dims(),
                        Shape::from_slice(self.shape()),
                        &Device::Cpu,
                    )
                }
            },
        }
    }
}

unsafe impl Send for Tensor {}

pub struct Tensor {
    dtype: GGmlType,
    data: Storage,
    dim: Dim,
    dev: Device,
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        &self
    }
}

impl Tensor {
    pub fn to_cpu_tensor(&self) -> GResult<Tensor> {
        match &self.data {
            Storage::Cpu(_) => todo!(),
            Storage::Gpu(g) => match g.slice() {
                CudaStorageSlice::Q4_0(v) => {
                    let elem_count = self.dim.elem_count();
                    let mut cpu_vec = vec![BlockQ4_0::default(); elem_count];
                    g.device().device.dtoh_sync_copy_into(&v.slice(..elem_count), &mut cpu_vec)?;
                    let cpu_slice = cpu_vec.to_cpu_storage();
                  Ok(
                    Tensor{
                        dtype: self.dtype(),
            data: Storage::Cpu(cpu_slice),
            dim: Dim {n_dims:self.n_dims(),shape:self.dim.shape.clone(),stride:self.dim.stride},
            dev: Device::Cpu,
                    }
                )
                }
                CudaStorageSlice::F16(v) => {
                    let elem_count = self.dim.elem_count();
                    let mut cpu_vec = vec![f16::zero(); elem_count];
                    g.device().device.dtoh_sync_copy_into(&v.slice(..elem_count), &mut cpu_vec)?;
                    let cpu_slice = cpu_vec.to_cpu_storage();
                  Ok(
                    Tensor{
                        dtype: self.dtype(),
                        data: Storage::Cpu(cpu_slice),
                        dim: Dim {n_dims:self.n_dims(),shape:self.dim.shape.clone(),stride:self.dim.stride},
                        dev: Device::Cpu,
                    }
                )
                }
                CudaStorageSlice::F32(v) => {
                    let elem_count = self.dim.elem_count();
                    let mut cpu_vec = vec![0.0f32; elem_count];
                    g.device().device.dtoh_sync_copy_into(&v.slice(..elem_count), &mut cpu_vec)?;
                    let cpu_slice = cpu_vec.to_cpu_storage();
                    Ok(
                      Tensor{
                        dtype: self.dtype(),
                        data: Storage::Cpu(cpu_slice),
                        dim: Dim {n_dims:self.n_dims(),shape:self.dim.shape.clone(),stride:self.dim.stride},
                        dev: Device::Cpu,
                      }
                  )
                }
                CudaStorageSlice::I32(v) => {
                    let elem_count = self.dim.elem_count();
                    let mut cpu_vec = vec![0;elem_count];
                    g.device().device.dtoh_sync_copy_into(&v.slice(..elem_count), &mut cpu_vec)?;
                    let cpu_slice = cpu_vec.to_cpu_storage();
                  Ok(
                    Tensor{
                        dtype: self.dtype(),
                        data: Storage::Cpu(cpu_slice),
                        dim: Dim {n_dims:self.n_dims(),shape:self.dim.shape.clone(),stride:self.dim.stride},
                        dev: Device::Cpu,
                    }
                )
                }
            },
        }
    }

    pub fn arr_array<A: TensorType, const N: usize>(xs: [A; N], dev: &Device) -> GResult<Self> {
        Tensor::arr(xs.to_vec(), dev)
    }

    pub fn mat_array<A: TensorType, const N: usize, const M: usize>(
        xs: [[A; N]; M],
        dev: &Device,
    ) -> GResult<Self> {
        Tensor::mat(xs.to_vec(), dev)
    }

    pub fn cube_array<A: TensorType, const N: usize, const M: usize, const K: usize>(
        xs: [[[A; N]; M]; K],
        dev: &Device,
    ) -> GResult<Self> {
        Tensor::cube(xs.to_vec(), dev)
    }

    pub fn arr_slice<A: TensorType>(xs: &[A], dev: &Device) -> GResult<Self> {
        Tensor::arr(xs.to_vec(), dev)
    }

    pub fn mat_slice<A: TensorType, const N: usize>(xs: &[[A; N]], dev: &Device) -> GResult<Self> {
        Tensor::mat(xs.to_vec(), dev)
    }

    pub fn cube_slice<A: TensorType, const N: usize, const M: usize>(
        xs: &[[[A; N]; M]],
        dev: &Device,
    ) -> GResult<Self> {
        Tensor::cube(xs.to_vec(), dev)
    }

    pub fn arr<A: TensorType>(xs: Vec<A>, dev: &Device) -> GResult<Self> {
        let dim = [xs.len()];
        let shape = Shape::from_array(dim);
        let cpu_dev = xs.to_cpu_storage();
        Tensor::from_cpu_storage(cpu_dev, 1, shape, A::DTYPE, dev)
    }

    pub fn mat<A: TensorType, const N: usize>(xs: Vec<[A; N]>, dev: &Device) -> GResult<Self> {
        let dim = [xs.len(), N];
        let shape = Shape::from_array(dim);
        let cpu_dev = xs.to_cpu_storage();
        Tensor::from_cpu_storage(cpu_dev, 2, shape, A::DTYPE, dev)
    }

    pub fn cube<A: TensorType, const M: usize, const N: usize>(
        xs: Vec<[[A; N]; M]>,
        dev: &Device,
    ) -> GResult<Self> {
        let dim = [xs.len(), M, N];
        let shape = Shape::from_array(dim);
        let cpu_dev = xs.to_cpu_storage();
        Tensor::from_cpu_storage(cpu_dev, 3, shape, A::DTYPE, dev)
    }

    fn from_cpu_storage(
        data: CpuStorageSlice,
        n_dims: usize,
        shape: Shape,
        dtype: GGmlType,
        dev: &Device,
    ) -> GResult<Self> {
        let stride = shape.ggml_stride(dtype);
        match dev {
            Device::Cpu => Ok(Self {
                dtype: dtype,
                data: Storage::Cpu(data),
                dim: Dim {
                    n_dims,
                    shape,
                    stride,
                },
                dev: dev.clone(),
            }),
            Device::Gpu(g) => {
                let cuda_stor = g.from_cpu_storage(data)?;
                Ok(Self {
                    dtype: dtype,
                    data: Storage::Gpu(cuda_stor),
                    dim: Dim {
                        n_dims,
                        shape,
                        stride,
                    },
                    dev: dev.clone(),
                })
            }
        }
    }

    // fn device(&self) -> &Storage {
    //     &self.data
    // }

    // fn device_dim(&mut self) -> (&mut Storage, &Dim) {
    //     (&mut self.data, &self.dim)
    // }

    // fn device_mut(&mut self) -> &mut Storage {
    //     &mut self.data
    // }

    fn nrows(&self) -> usize {
        let (_, d1, d2, d3) = self.dim4();
        d1 * d2 * d3
    }

    // pub fn set_value<A: TensorType>(&mut self, value: A) {
    //     let row = unsafe { self.as_slice_mut::<A>() };
    //     row.iter_mut().for_each(|e| *e = value);
    // }

    pub fn from_elem<A: TensorType>(a: A, n_dims: usize, s: Shape, dev: &Device) -> GResult<Self>
    where
        A: Clone,
    {
        let size = s.size();
        let mut v = RawSlice::<A>::with_capacity(size);
        v.fill(a, size);
        Tensor::from_cpu_storage(v.to_cpu_storage(), n_dims, s, A::DTYPE, dev)
    }

    pub fn from_vec<A: TensorType>(
        v: Vec<A>,
        n_dims: usize,
        s: Shape,
        dev: &Device,
    ) -> GResult<Self> {
        let cpu_dev = v.to_cpu_storage();
        Tensor::from_cpu_storage(cpu_dev, n_dims, s, A::DTYPE, dev)
    }

    pub fn from_raw(
        v: Vec<u8>,
        n_dims: usize,
        s: Shape,
        dtype: GGmlType,
        dev: &Device,
    ) -> GResult<Self> {
        match dtype {
            GGmlType::Q4_0 => Tensor::from_cpu_storage(
                CpuStorageSlice::Q4_0(RawSlice::from_raw(v)),
                n_dims,
                s,
                dtype,
                dev,
            ),
            GGmlType::F16 => Tensor::from_cpu_storage(
                CpuStorageSlice::F16(RawSlice::from_raw(v)),
                n_dims,
                s,
                dtype,
                dev,
            ),
            GGmlType::F32 => Tensor::from_cpu_storage(
                CpuStorageSlice::F32(RawSlice::from_raw(v)),
                n_dims,
                s,
                dtype,
                dev,
            ),
            // GGmlType::F64 => Tensor::from_storage(
            //     Storage::Cpu(CpuStorageSlice::F64(RawSlice::from_bytes(v))),
            //     n_dims,
            //     s,
            //     dtype,
            // ),
            GGmlType::I32 => Tensor::from_cpu_storage(
                CpuStorageSlice::I32(RawSlice::from_raw(v)),
                n_dims,
                s,
                dtype,
                dev,
            ),
            _ => {
                todo!()
            }
        }
    }

    pub fn nbytes(&self) -> usize {
        self.elem_count() * GS_TYPE_SIZE[self.dtype as usize] / GS_BLCK_SIZE[self.dtype as usize]
    }

    // pub fn as_bytes_mut(&mut self) -> &mut [u8] {
    //     self.device_mut().as_bytes_mut()
    // }

    // pub fn as_bytes(&self) -> &[u8] {
    //     self.storage().as_bytes()
    // }

    pub fn to_vec<T>(&self) -> Vec<T> {
        let bytes = self.as_bytes();
        assert!(bytes.len() as usize % GS_TYPE_SIZE[self.dtype as usize] == 0);
        let len = bytes.len() / GS_TYPE_SIZE[self.dtype as usize]; //* GS_BLCK_SIZE[self.dtype as usize] /
        let b = bytes.to_vec();
        let v = unsafe { Vec::from_raw_parts(b.as_ptr() as *mut T, len, len) };
        forget(b);
        v
    }

    pub fn zeros(n_dims: usize, d: Shape, dtype: GGmlType, dev: &Device) -> GResult<Self> {
        match dtype {
            GGmlType::F16 => Self::from_elem(f16::zero(), n_dims, d, dev),
            GGmlType::F32 => Self::from_elem(f32::zero(), n_dims, d, dev),
            //  GGmlType::F64 => Self::from_elem(f64::zero(), n_dims, d),
            _ => {
                todo!()
            }
        }
    }

    pub fn dtype_size(&self) -> usize {
        GS_TYPE_SIZE[self.dtype as usize] / GS_BLCK_SIZE[self.dtype as usize]
    }

    pub fn single_dim(&self, dim: usize) -> GResult<usize> {
        let s = self.dim().shape();
        if dim > s.len() {
            return Err(GError::DimOutOfRange {
                shape: self.dim.shape.clone(),
                dim: dim,
                op: "single_dim",
            }
            .into());
        }
        Ok(s[dim])
    }

    // pub fn chunk(&self, chunks: usize, dim: usize) -> GResult<Vec<Self>> {
    //     let size = self.dim().shape()[dim];
    //     if size < chunks {
    //         (0..size).map(|i| self.narrow(dim, i, 1)).collect()
    //     } else {
    //         let chunk_size = size / chunks;
    //         let cnt_additional = size % chunks;
    //         let mut tensors = vec![];
    //         let mut sum_chunk_size = 0;
    //         for i in 0..chunks {
    //             let chunk_size = if i < cnt_additional {
    //                 chunk_size + 1
    //             } else {
    //                 chunk_size
    //             };
    //             let tensor = self.narrow(dim, sum_chunk_size, chunk_size)?;
    //             tensors.push(tensor);
    //             sum_chunk_size += chunk_size
    //         }
    //         Ok(tensors)
    //     }
    // }

    // pub fn narrow(&self, dim: usize, start: usize, len: usize) -> GResult<Self> {
    //     let dims = self.dim().shape();
    //     let err = |msg| {
    //         Err::<(), _>(GError::NarrowInvalidArgs {
    //             shape: self.dim().shape.clone(),
    //             dim,
    //             start,
    //             len,
    //             msg,
    //         })
    //     };
    //     if start > dims[dim] {
    //         err("start > dim_len")?
    //     }
    //     if start.saturating_add(len) > dims[dim] {
    //         err("start + len > dim_len")?
    //     }
    //     if start == 0 && dims[dim] == len {
    //         Ok(self.clone())
    //     } else {
    //         let new_dim = self.dim.narrow(dim, start, len)?;
    //         let offset = self.dim().stride()[dim] * start;
    //         Ok(Tensor {
    //             dtype: self.dtype(),
    //             data: self.data.offset(offset),
    //             dim: new_dim,
    //         })
    //     }
    // }

    // pub fn permute(
    //     &mut self,
    //     axis0: usize,
    //     axis1: usize,
    //     axis2: usize,
    //     axis3: usize,
    // ) -> GResult<Tensor> {
    //     assert!(axis0 < MAX_DIM);
    //     assert!(axis1 < MAX_DIM);
    //     assert!(axis2 < MAX_DIM);
    //     assert!(axis3 < MAX_DIM);

    //     assert!(axis0 != axis1);
    //     assert!(axis0 != axis2);
    //     assert!(axis0 != axis3);
    //     assert!(axis1 != axis2);
    //     assert!(axis1 != axis3);
    //     assert!(axis2 != axis3);

    //     let shape = self.shape_layout();
    //     let stride = self.stride_layout();

    //     let mut ne = Layout::default();
    //     let mut nb = Layout::default();

    //     ne[axis0] = shape[0];
    //     ne[axis1] = shape[1];
    //     ne[axis2] = shape[2];
    //     ne[axis3] = shape[3];

    //     nb[axis0] = stride[0];
    //     nb[axis1] = stride[1];
    //     nb[axis2] = stride[2];
    //     nb[axis3] = stride[3];

    //     let mut result = self.view();

    //     {
    //         let shape_view = result.shape_layout_mut();
    //         shape_view[0] = ne[0];
    //         shape_view[1] = ne[1];
    //         shape_view[2] = ne[2];
    //         shape_view[3] = ne[3];
    //     }

    //     {
    //         let stride_view = result.stride_layout_mut();
    //         stride_view[0] = nb[0];
    //         stride_view[1] = nb[1];
    //         stride_view[2] = nb[2];
    //         stride_view[3] = nb[3];
    //     }

    //     Ok(result)
    // }

    fn into_transpose(mut self, d1: usize, d2: usize) -> GResult<Tensor> {
        if d1 > self.size() {
            return Err(GError::DimOutOfRange {
                shape: self.dim().shape.clone(),
                dim: d1,
                op: "transpose",
            }
            .into());
        }
        if d2 > self.size() {
            return Err(GError::DimOutOfRange {
                shape: self.dim().shape.clone(),
                dim: d2,
                op: "transpose",
            }
            .into());
        }
        let new_dim = self.dim.transpose(d1, d2)?;
        Ok(Tensor {
            dtype: self.dtype(),
            data: self.data,
            dim: new_dim,
            dev: self.dev,
        })
    }

    // pub fn pad<A: TensorType>(
    //     &self,
    //     dim: usize,
    //     left: usize,
    //     right: usize,
    //     elem: A,
    // ) -> GResult<Tensor> {
    //     if left == 0 && right == 0 {
    //         Ok(self.clone())
    //     } else if left == 0 {
    //         let mut dims = self.dim().shape().to_vec();
    //         dims[dim] = right;
    //         let right = Tensor::from_elem(elem, dims.len(), Shape::from_vec(dims));
    //         Tensor::cat(&[self, &right], dim)
    //     } else if right == 0 {
    //         let mut dims = self.dim().shape().to_vec();
    //         dims[dim] = left;
    //         let left = Tensor::from_elem(elem, dims.len(), Shape::from_vec(dims));
    //         Tensor::cat(&[&left, &self], dim)
    //     } else {
    //         let mut dims = self.dim().shape().to_vec();
    //         dims[dim] = left;
    //         let left = Tensor::from_elem(elem, dims.len(), Shape::from_slice(dims.as_slice()));
    //         dims[dim] = right;
    //         let right = Tensor::from_elem(elem, dims.len(), Shape::from_vec(dims));
    //         Tensor::cat(&[&left, &self, &right], dim)
    //     }
    // }

    // pub fn cat<T: AsRef<Tensor>>(ts: &[T], dim: usize) -> GResult<Tensor> {
    //     if dim == 0 {
    //         Self::cat0(ts)
    //     } else {
    //         let args: Vec<Tensor> = ts
    //             .iter()
    //             .map(|a| a.as_ref().view().into_transpose(0, dim))
    //             .collect::<GResult<Vec<Tensor>>>()?;
    //         let cat = Self::cat0(&args)?;
    //         cat.into_transpose(0, dim)
    //     }
    // }

    // fn cat0<T: AsRef<Tensor>>(ts: &[T]) -> GResult<Tensor> {
    //     let t0 = ts[0].as_ref();
    //     let mut cat_dims = t0.shape().to_vec();
    //     cat_dims[0] = 0;
    //     let mut offsets = vec![0usize];
    //     for (t_i, arg) in ts.iter().enumerate() {
    //         let t = arg.as_ref();
    //         for (dim_idx, (v1, v2)) in t0.shape().iter().zip(t.shape().iter()).enumerate() {
    //             if dim_idx == 0 {
    //                 cat_dims[0] += v2;
    //             }
    //         }
    //         let next_offset = offsets.last().unwrap() + t.elem_count();
    //         offsets.push(next_offset);
    //     }
    //     let n_dims = cat_dims.len();
    //     let shape: Shape = Shape::from_vec(cat_dims);
    //     let mut new_tensor = Self::zeros(n_dims, shape, t0.dtype());
    //     for (arg, &offset) in ts.iter().zip(offsets.iter()) {
    //         let t = arg.as_ref();
    //         t.storage()
    //             .copy_strided_src(new_tensor.device_mut(), offset, t.dim());
    //     }
    //     Ok(new_tensor)
    // }

    // fn axis_index(&self, a: Axis, index: usize) -> Tensor {
    //     let nd_stride = self.dim.nd_stride();
    //     let axis = a.index();
    //     let stride = nd_stride[axis];
    //     let offset = index * stride;
    //     let axis_dim = self.dim.shape.select_axis(a);
    //     let s = shape::select_axis(&nd_stride, a);
    //     // let raw_ref = RawRef {
    //     //     ptr: unsafe { self.data.as_ptr().offset(offset as isize) },
    //     //     len: self.data.len() - offset,
    //     // };

    //     let n_dims = self.dim.n_dims() - 1;
    //     Tensor {
    //         dtype: self.dtype(),
    //         data: self.storage().offset(offset),
    //         dim: Dim {
    //             n_dims: n_dims,
    //             shape: axis_dim,
    //             stride: s,
    //         },
    //     }
    // }

    // fn axis_index_inner(&self, a: Axis, index: usize) -> Tensor {
    //     let axis = a.index();
    //     let stride = self.dim.stride[axis];
    //     let offset = index * stride;
    //     let axis_dim = self.dim.shape.select_axis(a);
    //     let s = shape::select_axis(&self.dim.stride(), a);
    //     // let raw_ref = RawRef {
    //     //     ptr: unsafe { self.as_ptr().offset(offset as isize) },
    //     //     len: self.data.len() - offset,
    //     // };
    //     let n_dims = &self.dim.n_dims() - 1;
    //     Tensor {
    //         dtype: self.dtype(),
    //         data: self.storage().offset(offset), //Storage::Cpu(RawSlice::Ref(raw_ref)),
    //         dim: Dim {
    //             n_dims: n_dims,
    //             shape: axis_dim,
    //             stride: s,
    //         },
    //     }
    // }

    pub fn iter(&self) -> TensorIter<'_> {
        TensorIter {
            n_dims: self.dim().n_dims(),
            device: &self.data,
            shape_iter: self.dim.shape.iter(self.dim().n_dims()),
            strides: self.dim.nd_stride(),
        }
    }
}

#[derive(Debug)]
pub struct StridedIndex<'a> {
    next_device_index: Option<usize>,
    multi_index: Vec<usize>,
    dims: &'a [usize],
    stride: &'a [usize],
}

impl<'a> StridedIndex<'a> {
    pub(crate) fn new(dims: &'a [usize], stride: &'a [usize], start_offset: usize) -> Self {
        let elem_count: usize = dims.iter().product();
        let next_device_index = if elem_count == 0 {
            None
        } else {
            // This applies to the scalar case.
            Some(start_offset)
        };
        StridedIndex {
            next_device_index,
            multi_index: vec![0; dims.len()],
            dims,
            stride,
        }
    }
}

impl<'a> Iterator for StridedIndex<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let device_index = match self.next_device_index {
            None => return None,
            Some(device_index) => device_index,
        };
        let mut updated = false;
        let mut next_device_index = device_index;
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
                next_device_index += stride_i;
                break;
            } else {
                next_device_index -= *multi_i * stride_i;
                *multi_i = 0
            }
        }
        self.next_device_index = if updated {
            Some(next_device_index)
        } else {
            None
        };
        Some(device_index)
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

// impl<T: TensorType> Drop for RawSlice<T> {
//     fn drop(&mut self) {
//         match self {
//             RawSlice::Own(v) => drop(v),
//             RawSlice::Ref(_) => {}
//         }
//     }
// }

impl<T: TensorType> Drop for RawSlice<T> {
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
        // format_tensor(self.view(), 0, f)
        todo!()
    }
}

// fn format_tensor(tensor: Tensor, depth: usize, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//     match tensor.shape() {
//         &[] => {}
//         &[len] => {
//             f.write_str("[")?;
//             for (i, v) in tensor.iter().enumerate() {
//                 if i > 0 {
//                     f.write_str(",")?;
//                 }
//                 f.write_str(&format!(" {:?}", v))?;
//             }
//             f.write_str("]")?;
//         }
//         shape => {
//             let blank_lines = "\n".repeat(shape.len() - 2);
//             let indent = " ".repeat(depth + 1);
//             let separator = format!(",\n{}{}", blank_lines, indent);
//             f.write_str("[")?;
//             format_tensor(tensor.axis_index(Axis(0), 0), depth, f)?;
//             for i in 1..shape[0] {
//                 f.write_str(&separator)?;
//                 format_tensor(tensor.axis_index(Axis(0), i), depth, f)?;
//             }
//             f.write_str("]")?;
//         }
//     }
//     Ok(())
// }

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
        let mut t = RawSlice::<f32>::with_capacity(10);
        t.fill(3.0, 10);
        // println!("t:{:?}", t.take_as_vec());
    }

    #[test]
    fn test_mat() {
        let m = Tensor::mat_slice(&[[1.0, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        for i in m.iter() {
            println!("v:{:?}", i);
        }
        println!("shape:{:?}", m.dim().shape());
        println!("stride:{:?}", m.dim().stride());
        // //   let v = m.as_slice();
        // let s = &m.dim.stride;
        // {
        //     let t = m.as_ref();
        // }
        // println!("s:{:?}", s);
        println!("v:{:?}", m);
    }

    #[test]
    fn test_cude() {
        let m = Tensor::cube_slice(
            &[
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ],
            &Device::Cpu,
        )
        .unwrap();
        //  let v = m.as_slice();
        // println!("v:{:?}", m);
        for i in m.iter() {
            println!("v:{:?}", i);
        }
        println!("shape:{:?}", m.dim().shape());
        println!("stride:{:?}", m.dim().stride());

        // for v in m.iter() {
        //     println!("v:{:?}", v);
        // }

        // let m1 = m.into_transpose(0, 1).unwrap();

        // for v in m1.iter() {
        //     println!("v1:{:?}", v);
        // }
        // let vv = m1.axis_index(Axis(0), 0);
        // println!("vv:{:?}", vv);
        // println!("m1:{:?}", m1);
        // println!("shape:{:?}", m1.dim().shape());
        // println!("stride:{:?}", m1.dim().stride());
    }

    #[test]
    fn test_fmt() {
        let a =
            Tensor::from_elem(1.0f32, 3, Shape::from_array([4usize, 3, 2]), &Device::Cpu).unwrap();
        //let v = a.as_slice();
        let t = a.as_ref();
        println!("v:{:?}", a);
    }

    #[test]
    fn test_reshape() {
        // let t1 = cube(&[
        //     [
        //         [1.0, 2.0, 1.0, 2.0],
        //         [3.0, 4.0, 3.0, 4.0],
        //         [3.0, 4.0, 3.0, 4.0],
        //     ],
        //     [
        //         [5.0, 6.0, 5.0, 6.0],
        //         [7.0, 8.0, 7.0, 8.0],
        //         [7.0, 8.0, 7.0, 8.0],
        //     ],
        // ]);
        // println!("t1:{:?}", t1.shape());

        // let t2 = t1.reshape(Shape::from_array([2, 3, 2, 2]));
        // println!("t2:{:?}", t2);
    }

    #[test]
    fn test_transpose() {
        let m = Tensor::cube_slice(
            &[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            &Device::Cpu,
        )
        .unwrap();
        let m1 = m.into_transpose(1, 0).unwrap();
        println!("m1:{:?}", m1);
    }

    #[test]
    fn test_cat() {
        let m1 = Tensor::mat_slice(&[[1.0, 2.0], [3.0, 4.0]], &Device::Cpu);

        let m2 = Tensor::mat_slice(&[[1.0, 2.0], [3.0, 4.0]], &Device::Cpu);

        // let m3 = Tensor::cat(&[&m1, &m2], 0).unwrap();

        // println!("m3:{:?}", m3);
        // for m in m3.iter() {
        //     println!("m:{:?}", m);
        // }

        // let m4: Tensor = Tensor::cat(&[&m1, &m2], 1).unwrap();

        // println!("m4:{:?}", m4);
        // println!("shape:{:?}", m4.dim());
        //    println!("slice:{:?}", m4.as_slice());

        let m5 =
            Tensor::mat_slice(&[[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]], &Device::Cpu).unwrap();
        println!("m5:{:?}", m5);
        println!("shape:{:?}", m5.dim());
        //   println!("slice:{:?}", m5.as_slice());

        let m6 = Tensor::cube_slice(
            &[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]],
            &Device::Cpu,
        );
        let m7 = Tensor::cube_slice(
            &[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]],
            &Device::Cpu,
        );
        // let m8: Tensor = Tensor::cat(&[&m6, &m7], 2).unwrap();
        // println!("m8:{:?}", m8);
        // println!("shape:{:?}", m8.dim());
        //     println!("slice:{:?}", m8.as_slice());
    }

    // #[test]
    // fn test_pad() {
    //     let m1 = Tensor::cube_slice(&[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]);
    //     let d = m1.pad(2, 1, 1, 0.0).unwrap().pad(3, 0, 0, 0.0).unwrap();
    //     for x in d.iter() {
    //         println!("{:?}", x);
    //     }
    //     println!("{:?}", d);
    // }

    // #[test]
    // fn test_matmul() {
    //     let m1 = mat(&[[1.0f32, 2.0], [3.0f32, 4.0], [3.0f32, 4.0]]);
    //     let m2 = mat(&[[1.0f32, 2.0, 4.0], [3.0f32, 4.0, 5.0]]);
    //     let mut d = m1.matmul(&m2).unwrap();
    //     println!("{:?}", d);
    //     let v = unsafe { d.as_slice_mut::<f32>() };
    //     println!("{:?}", v);

    //     // let m1 = cube(&[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]);
    //     // let m2 = mat(&[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    //     // let d = m1.matmul(&m2).unwrap();
    //     // println!("{:?}", d);
    // }

    #[test]
    fn test_cpu_storage() {
        //  let c = CpuStorage(vec![0u8; 10]);
        //let a = vec![5u8; 10];
        //  let d = CpuStorage(a.as_slice());
    }

    #[test]
    fn test_sqrt() {
        let m1 = Tensor::cube_slice(
            &[[[4.0, 16.0], [3.0, 4.0]], [[4.0, 9.0], [36.0, 81.0]]],
            &Device::Cpu,
        )
        .unwrap();
        //let s = m1.sqrt();
        println!("{:?}", m1);
        // println!("{:?}", s);
    }

    // #[test]
    // fn test_narrow() -> GResult<()> {
    //     let a = Tensor::mat_slice(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    //     let b = a.narrow(0, 0, 2)?;
    //     println!("b:{:?}", b);

    //     let c = a.narrow(1, 1, 2)?;
    //     println!("c:{:?}", c);
    //     Ok(())
    // }

    // #[test]
    // fn test_chunk() -> GResult<()> {
    //     let a = Tensor::mat_slice(&[
    //         [0.0, 1.0, 2.0, 3.0],
    //         [4.0, 5.0, 6.0, 7.0],
    //         [8.0, 9.0, 10.0, 11.0],
    //         [12.0, 13.0, 14.0, 15.0],
    //     ]);
    //     let b = a.chunk(4, 0)?;
    //     for i in b {
    //         println!("b:{:?}", i);
    //     }

    //     let c = a.chunk(4, 1)?;
    //     for i in c {
    //         println!("c:{:?}", i);
    //     }
    //     Ok(())
    // }

    #[test]
    fn test_clone() -> GResult<()> {
        let mut a = Tensor::mat_slice(
            &[
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 1.0, 14.0, 15.0],
            ],
            &Device::Cpu,
        );
        //let b = a.clone();
        // a.as_slice_mut()[1] = 100;
        // println!("{:?}", a);
        // println!("{:?}", b);
        Ok(())
    }

    // #[test]
    // fn test_slice() -> GResult<()> {
    //     let mut a = vec![1.0, 2.0, 3.0, 4.0];
    //     let mut t = TensorView::from_slice(&a, 2, Shape::from_array([2, 2]));
    //     a[1] = 15.0;
    //     println!("t:{:?}", t);

    //     let v = t.as_bytes_mut();
    //     println!("v:{:?}", v);
    //     Ok(())
    // }
}
