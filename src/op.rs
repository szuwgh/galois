use std::ops::Neg;

use crate::error::GResult;
use crate::Device;
use crate::GError;
//use super::broadcast::{broadcasting_binary_op, general_broadcasting};
use crate::CpuDevice;
use crate::Dim;
use crate::Tensor;
use crate::TensorType;
use crate::F16;
use half::f16;
use num_traits::Float;
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv1D {
    pub(crate) b_size: usize,
    // Maybe we should have a version without l_in as this bit depends on the input and not only on
    // the weights.
    pub(crate) l_in: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) k_size: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

impl ParamsConv1D {
    pub(crate) fn l_out(&self) -> usize {
        if self.l_in == 1 {
            return 1;
        }
        (self.l_in + 2 * self.padding - self.dilation * (self.k_size - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        vec![self.b_size, self.c_out, l_out]
    }
}

fn up32(n: i32) -> i32 {
    (n + 31) & !31
}

struct Conv1D(ParamsConv1D);

impl Map2 for Conv1D {
    const OP: &'static str = "conv1d";
    fn f<T: TensorType>(
        &self,
        inp: &[T],
        inp_d: &Dim,
        k: &[T],
        k_d: &Dim,
        dst: &mut [T],
    ) -> GResult<()> {
        let p = &self.0;
        // let inp = &inp[inp_l.start_offset()..];
        // let k = &k[k_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2) = inp_d.stride_3d();
        let (k_s0, k_s1, k_s2) = k_d.stride_3d();
        let l_out = p.l_out();
        let dst_elems = p.c_out * l_out * p.b_size;
        assert!(dst.len() == dst_elems);
        // The output shape is [b_size, c_out, l_out]
        // let dst = vec![T::zero(); dst_elems];

        // TODO: Avoid making this copy if `inp` already has the appropriate layout.
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.l_in];
        for b_idx in 0..p.b_size {
            for src_l in 0..p.l_in {
                for src_c_idx in 0..p.c_in {
                    let inp_idx = b_idx * inp_s0 + src_c_idx * inp_s1 + src_l * inp_s2;
                    inp_cont[b_idx * p.l_in * p.c_in + src_l * p.c_in + src_c_idx] = inp[inp_idx]
                }
            }
        }

        for offset in 0..p.k_size {
            (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                let dst_idx = dst_c_idx * l_out;
                let k_cont = (0..p.c_in)
                    .map(|c_in_idx| k[dst_c_idx * k_s0 + c_in_idx * k_s1 + offset * k_s2])
                    .collect::<Vec<_>>();
                for b_idx in 0..p.b_size {
                    let dst_idx = dst_idx + b_idx * p.c_out * l_out;
                    for dst_l in 0..l_out {
                        let dst_idx = dst_idx + dst_l;
                        let src_l = p.stride * dst_l + offset * p.dilation;
                        if src_l < p.padding || src_l >= p.padding + p.l_in {
                            continue;
                        }
                        let src_l = src_l - p.padding;
                        let inp_cont = &inp_cont[b_idx * p.l_in * p.c_in + src_l * p.c_in..];
                        assert!(inp_cont.len() >= p.c_in);
                        assert!(k_cont.len() >= p.c_in);
                        let mut d = T::zero();
                        unsafe { T::vec_dot(inp_cont.as_ptr(), k_cont.as_ptr(), &mut d, p.c_in) }
                        let dst_p = dst.as_ptr();
                        // Safety: dst_idx are uniques per dst_c_idx which is used to parallelise
                        // the different tasks so no two threads can try to write at the same
                        // location.
                        unsafe {
                            let ptr = dst_p.add(dst_idx) as *mut T;
                            *ptr += d
                        }
                    }
                }
            })
        }
        Ok(())
    }

    fn f_f16_sf32(
        &self,
        inp: &[f32],
        inp_d: &Dim,
        kernel: &[f16],
        k_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        let (ne00, ne01, ne02) = k_d.dim3();

        let (ne10, ne11) = inp_d.dim2();
        let (nb00, nb01, nb02) = k_d.stride_3d();
        let (nb10, nb11) = inp_d.stride_2d();

        let nb1 = dst_d.stride()[1];

        let ith = 0;
        let nth: usize = 1;

        let nk = ne00;
        let nh: i32 = (nk / 2) as i32;

        let ew0 = up32(ne01 as i32) as usize;
        println!("ne10:{}", ne10);
        println!("ne00:{}, ne01:{}, ne02:{}", ne00, ne01, ne02);
        println!("nb1:{}", nb1);
        println!("ew0:{}", ew0);
        println!("inp:{}", inp.len());
        println!("kernel:{}", kernel.len());
        println!("nh:{}", nh);
        println!("dst.len:{}", dst.len());

        println!("nb00:{},nb01:{},nb02:{}", nb00, nb01, nb02);

        let mut ker_f16: Vec<f16> = vec![f16::from_f32(0.0); ne02 * ew0 * ne00];

        for i02 in 0..ne02 {
            for i01 in 0..ne01 {
                let src_start = i02 * nb02 + i01 * nb01;
                //  let src_end = src_start + ne00;
                let src_slice = &kernel[src_start..];

                let dst_start = i02 * ew0 * ne00;
                // let dst_end = dst_start + ne00 * ew0;
                let dst_slice = &mut ker_f16[dst_start..];
                for i00 in 0..ne00 {
                    dst_slice[i00 * ew0 + i01] = src_slice[i00];
                }
                // 将 src_slice 中的元素复制到 dst_slice 中
                // dst_slice.copy_from_slice(&src_slice);
            }
        }

        let kernel_data: f32 = ker_f16.iter().map(|e| e.to_f32()).sum();
        println!("kernel_data:{:?}", kernel_data);

        let mut inp_f16: Vec<f16> =
            vec![f16::from_f32(0.0); (ne10 + nh as usize) * ew0 * ne11 + ew0];
        for i11 in 0..ne11 {
            let src_chunk = &inp[i11 * nb11..];

            for i10 in 0..ne10 {
                let index = (i10 + nh as usize) * ew0 + i11;
                inp_f16[index] = f16::from_f32(src_chunk[i10]);
            }
        }

        // let inp_f16: Vec<f16> = inp.iter().map(|&val| f16::from_f32(val)).collect();

        // total rows in dst
        let nr = ne02;

        // rows per thread
        let dr = (nr + nth - 1) / nth;

        // row range for this thread
        let ir0 = dr * ith;
        let ir1 = std::cmp::min(ir0 + dr, nr);
        println!("nr:{}", nr);
        println!("dr:{}", dr);
        println!("ir0:{}", ir0);
        println!("ir1:{}", ir1);

        let mut xxx = 0.0;
        for i1 in ir0..ir1 {
            let dst_data = &mut dst[i1 * nb1..];

            for i0 in 0..ne10 {
                dst_data[i0] = 0.0;
                for k in -nh..=nh {
                    let mut v = 0.0f32;
                    let wdata1_idx = ((i1 * ew0 * ne00) as i32 + (nh + k) * ew0 as i32) as usize; //kernel
                    let wdata2_idx = ((i0 as i32 + nh + k) * ew0 as i32) as usize; //src

                    let wdata1 = &ker_f16[wdata1_idx..wdata1_idx + ew0];
                    let wdata2 = &inp_f16[wdata2_idx..wdata2_idx + ew0];
                    // println!("wdata1_idx:{}", wdata1_idx);
                    //println!("wdata2_idx:{}", wdata2_idx);
                    unsafe { f16::vec_dot_f16_f32(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0) }
                    dst_data[i0] += v;
                    xxx += v;
                }
            }
        }
        println!("xxx:{}", xxx);
        Ok(())
    }
}

trait Map2 {
    const OP: &'static str;
    fn f<T: TensorType>(
        &self,
        inp: &[T],
        inp_l: &Dim,
        k: &[T],
        k_l: &Dim,
        dst: &mut [T],
    ) -> GResult<()>;

    fn f_f16_sf32(
        &self,
        inp: &[f32],
        inp_l: &Dim,
        k: &[f16],
        k_l: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()>;

    fn map(
        &self,
        dev1: &CpuDevice,
        d1: &Dim,
        dev2: &CpuDevice,
        d2: &Dim,
        dst: &mut CpuDevice,
        d3: &Dim,
    ) -> GResult<()> {
        match (dev1, dev2, dst) {
            (CpuDevice::F16(v1), CpuDevice::F16(v2), CpuDevice::F16(d)) => {
                self.f(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut())
            }
            (CpuDevice::F32(v1), CpuDevice::F32(v2), CpuDevice::F32(d)) => {
                self.f(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut())
            }
            (CpuDevice::F32(v1), CpuDevice::F16(v2), CpuDevice::F32(d)) => {
                self.f_f16_sf32(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut(), d3)
            }
            _ => {
                todo!()
            }
        }
    }
}

pub fn conv_1d_1s(
    src: &Tensor,
    kernel: &Tensor,
    dst: &mut Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> GResult<()> {
    let (c_out, c_in_k, k_size) = kernel.dim3();
    println!("c_out:{}, c_in_k:{}, k_size:{}", c_out, c_in_k, k_size);
    let (b_size, c_in, l_in) = src.dim3();
    println!("b_size:{}, c_in:{}, l_in:{}", b_size, c_in, l_in);
    if c_in != c_in_k * groups {
        return Err(GError::Conv1dInvalidArgs {
            inp_shape: src.shape().to_vec(),
            k_shape: kernel.shape().to_vec(),
            padding,
            stride,
            msg: "the number of in-channels on the input doesn't match the kernel size",
        });
    }

    let params = ParamsConv1D {
        b_size,
        l_in,
        c_out: c_out / groups,
        c_in: c_in / groups,
        k_size,
        padding,
        stride,
        dilation,
    };
    let dst_dim = dst.dim().clone();
    match (src.device(), kernel.device(), dst.device_mut()) {
        (Device::Cpu(s), Device::Cpu(k), Device::Cpu(d)) => {
            Conv1D(params).map(s, src.dim(), k, kernel.dim(), d, &dst_dim)?;
        }
        _ => {
            todo!()
        }
    }

    Ok(())
}

// trait Map {
//     fn f<T: TensorType>(&self, t: &Tensor) -> LNResult<DTensor<T>>;

//     fn out_shape<T: TensorType>(&self, t: &DTensor<T>) -> LNResult<Shape>;

//     fn map(&self, t: Tensor) -> LNResult<GTensor> {
//         let new_t = match t.as_value_ref().as_tensor_ref() {
//             GTensor::U8(t1) => GTensor::U8(self.f(t1)?),
//             GTensor::I8(t1) => GTensor::I8(self.f(t1)?),
//             GTensor::I16(t1) => GTensor::I16(self.f(t1)?),
//             GTensor::U16(t1) => GTensor::U16(self.f(t1)?),
//             GTensor::I32(t1) => GTensor::I32(self.f(t1)?),
//             GTensor::U32(t1) => GTensor::U32(self.f(t1)?),
//             GTensor::I64(t1) => GTensor::I64(self.f(t1)?),
//             GTensor::U64(t1) => GTensor::U64(self.f(t1)?),
//             GTensor::F16(t1) => GTensor::F16(self.f(t1)?),
//             GTensor::F32(t1) => GTensor::F32(self.f(t1)?),
//             GTensor::F64(t1) => GTensor::F64(self.f(t1)?),
//         };
//         Ok(new_t)
//     }
// }

// pub trait Distance {}

// #[macro_export]
// macro_rules! dim_max {
//     ($d1:expr, $d2:expr) => {
//         <$d1 as DimMax<$d2>>::Output
//     };
// }

// fn convert_iopsf<A: Clone, B: Clone>(f: impl Fn(A, B) -> A) -> impl FnMut((&mut A, &B)) {
//     move |(x, y)| *x = f(x.clone(), y.clone())
// }

// fn clone_opsf<A: Clone, B: Clone, C>(f: impl Fn(A, B) -> C) -> impl FnMut((&mut A, &B)) -> C {
//     move |(x, y)| f(x.clone(), y.clone())
// }

// macro_rules! impl_binary_op {
//     ($trt:ident, $mth:ident) => {
//         impl<A> std::ops::$trt<&Tensor<A>> for Tensor<A>
//         where
//             A: std::ops::$trt<A, Output = A> + TensorType,
//         {
//             type Output = Tensor<A>;
//             fn $mth(self, rhs: &Tensor<A>) -> Self::Output {
//                 let lhs = self;
//                 if lhs.shape() == rhs.shape() {
//                     println!("shape same");
//                     lhs.iter().zip2(rhs.iter()).ops(convert_iopsf(A::$mth));
//                     lhs
//                 } else {
//                     println!("shape not same");
//                     let broadcast_shape =
//                         broadcasting_binary_op::<A>(lhs.shape(), rhs.shape()).unwrap();

//                     let l_broadcast = broadcast_shape == *lhs.shape();
//                     let r_broadcast = broadcast_shape == *rhs.shape();

//                     let v = match (l_broadcast, r_broadcast) {
//                         (true, true) => {
//                             lhs.iter().zip2(rhs.iter()).ops(convert_iopsf(A::$mth));
//                             lhs
//                         }
//                         (true, false) => {
//                             lhs.iter()
//                                 .zip2(rhs.broadcast_with(&broadcast_shape).unwrap().iter())
//                                 .ops(convert_iopsf(A::$mth));
//                             lhs
//                         }
//                         (false, true) => lhs
//                             .broadcast_with(&broadcast_shape)
//                             .unwrap()
//                             .iter()
//                             .zip2(rhs.iter())
//                             .map(clone_opsf(A::$mth))
//                             .collect_tensor(lhs.dim.shape().clone()),
//                         (false, false) => lhs
//                             .broadcast_with(&broadcast_shape)
//                             .unwrap()
//                             .iter()
//                             .zip2(rhs.broadcast_with(&broadcast_shape).unwrap().iter())
//                             .map(clone_opsf(A::$mth))
//                             .collect_tensor(lhs.dim.shape().clone()),
//                     };
//                     v
//                 }
//             }
//         }

//         impl<A> std::ops::$trt<Tensor<A>> for Tensor<A>
//         where
//             A: std::ops::$trt<A, Output = A> + TensorType,
//         {
//             type Output = Tensor<A>;
//             fn $mth(self, rhs: Tensor<A>) -> Self::Output {
//                 let lhs = self;
//                 if lhs.shape() == rhs.shape() {
//                     lhs.iter().zip2(rhs.iter()).ops(convert_iopsf(A::$mth));
//                     lhs
//                 } else {
//                     let broadcast_shape =
//                         broadcasting_binary_op::<A>(lhs.shape(), rhs.shape()).unwrap();

//                     let l_broadcast = broadcast_shape == *lhs.shape();
//                     let r_broadcast = broadcast_shape == *rhs.shape();

//                     let v = match (l_broadcast, r_broadcast) {
//                         (true, true) => {
//                             lhs.iter().zip2(rhs.iter()).ops(convert_iopsf(A::$mth));
//                             lhs
//                         }
//                         (true, false) => {
//                             lhs.iter()
//                                 .zip2(rhs.broadcast_with(&broadcast_shape).unwrap().iter())
//                                 .ops(convert_iopsf(A::$mth));
//                             lhs
//                         }
//                         (false, true) => lhs
//                             .broadcast_with(&broadcast_shape)
//                             .unwrap()
//                             .iter()
//                             .zip2(rhs.iter())
//                             .map(clone_opsf(A::$mth))
//                             .collect_tensor(lhs.dim.shape().clone()),
//                         (false, false) => lhs
//                             .broadcast_with(&broadcast_shape)
//                             .unwrap()
//                             .iter()
//                             .zip2(rhs.broadcast_with(&broadcast_shape).unwrap().iter())
//                             .map(clone_opsf(A::$mth))
//                             .collect_tensor(lhs.dim.shape().clone()),
//                     };
//                     v
//                 }
//             }
//         }

//         impl<A> std::ops::$trt<&Tensor<A>> for &Tensor<A>
//         where
//             A: std::ops::$trt<A, Output = A> + TensorType,
//         {
//             type Output = Tensor<A>;
//             fn $mth(self, rhs: &Tensor<A>) -> Self::Output {
//                 if self.shape() == rhs.shape() {
//                     println!("shape same");
//                     self.iter()
//                         .zip2(rhs.iter())
//                         .map(clone_opsf(A::$mth))
//                         .collect_tensor(rhs.dim.shape().clone())
//                 } else {
//                     println!("shape not same");
//                     let (lhs, rhs2) = general_broadcasting::<A>(&self, &rhs).unwrap();
//                     lhs.iter()
//                         .zip2(rhs2.iter())
//                         .map(clone_opsf(A::$mth))
//                         .collect_tensor(lhs.dim.shape().clone())
//                 }
//             }
//         }

//         impl<A> std::ops::$trt<Tensor<A>> for &Tensor<A>
//         where
//             A: std::ops::$trt<A, Output = A> + TensorType,
//         {
//             type Output = Tensor<A>;
//             fn $mth(self, rhs: Tensor<A>) -> Self::Output {
//                 if self.shape() == rhs.shape() {
//                     println!("shape same");
//                     self.iter()
//                         .zip2(rhs.iter())
//                         .map(clone_opsf(A::$mth))
//                         .collect_tensor(rhs.dim.shape().clone())
//                 } else {
//                     println!("shape not same");
//                     let (lhs, rhs2) = general_broadcasting::<A>(&self, &rhs).unwrap();
//                     lhs.iter()
//                         .zip2(rhs2.iter())
//                         .map(clone_opsf(A::$mth))
//                         .collect_tensor(lhs.dim.shape().clone())
//                 }
//             }
//         }
//     };
// }

// macro_rules! impl_scalar_op {
//     ($trt:ident, $mth:ident, $op:tt) => {
//         impl<A> std::ops::$trt<A> for Tensor<A>
//         where
//             A: std::ops::$trt<A, Output = A> + TensorType,
//         {
//             type Output = Tensor<A>;
//             fn $mth(mut self, rhs: A) -> Self::Output {
//                 self.as_slice_mut().iter_mut().for_each(|x| *x = *x $op rhs);
//                 self
//             }
//         }

//         impl<A> std::ops::$trt<A> for &Tensor<A>
//         where
//             A: std::ops::$trt<A, Output = A> + TensorType,
//         {
//             type Output = Tensor<A>;
//             fn $mth(self, rhs: A) -> Self::Output {
//                let v = self.as_slice().iter().map(|x| *x $op rhs).collect();
//                Tensor::from_vec(v, self.dim().shape().clone())
//             }
//         }
//     };
// }

// impl<A> PartialEq<Tensor<A>> for Tensor<A>
// where
//     A: TensorType,
// {
//     fn eq(&self, other: &Tensor<A>) -> bool {
//         if self.shape() != other.shape() {
//             return false;
//         }
//         for (a, b) in self.iter().zip(other.iter()) {
//             if *a != *b {
//                 println!("a:{:?},b:{:?}", a, b);
//                 return false;
//             }
//         }
//         return true;
//     }
// }

// // Exp,
// // Log,
// // Sin,
// // Cos,
// // Tanh,
// // Neg,
// // Recip,
// // Sqr,
// // Sqrt,
pub trait UnaryOp {
    fn _exp(&self) -> Self;
    fn _ln(&self) -> Self;
    fn _sin(&self) -> Self;
    fn _cos(&self) -> Self;
    fn _tanh(&self) -> Self;
    fn _neg(&self) -> Self;
    fn _recip(&self) -> Self;
    fn _sqr(&self) -> Self;
    fn _sqrt(&self) -> Self;
}

macro_rules! impl_float_unary_op {
    ($a:ident) => {
        impl UnaryOp for $a {
            fn _exp(&self) -> Self {
                self.exp()
            }
            fn _ln(&self) -> Self {
                self.ln()
            }
            fn _sin(&self) -> Self {
                self.sin()
            }
            fn _cos(&self) -> Self {
                self.cos()
            }
            fn _tanh(&self) -> Self {
                self.tanh()
            }
            fn _neg(&self) -> Self {
                self.neg()
            }

            fn _recip(&self) -> Self {
                self.recip()
            }
            fn _sqr(&self) -> Self {
                self * self
            }
            fn _sqrt(&self) -> Self {
                self.sqrt()
            }
        }
    };
}

impl_float_unary_op!(f16);
impl_float_unary_op!(f32);
impl_float_unary_op!(f64);

// impl_binary_op!(Add, add); // +
// impl_binary_op!(Sub, sub); // -
// impl_binary_op!(Mul, mul); // *
// impl_binary_op!(Div, div); // /
// impl_binary_op!(Rem, rem); // %
//                            // impl_binary_op!(BitAnd, bitand); // &
//                            // impl_binary_op!(BitOr, bitor); // |
//                            // impl_binary_op!(BitXor, bitxor); // ^
//                            // impl_binary_op!(Shl, shl); // <<
//                            // impl_binary_op!(Shr, shr); // >>

// impl_scalar_op!(Add, add, +);
// impl_scalar_op!(Sub, sub,-); // -
// impl_scalar_op!(Mul, mul,*); // *
// impl_scalar_op!(Div, div,/); // /
// impl_scalar_op!(Rem, rem,%); // %

// mod tests {
//     use super::super::{arr, mat};
//     use super::*;
//     #[test]
//     fn test_add() {
//         let m1 = mat(&[
//             [0.0, 0.0, 0.0],
//             [1.0, 1.0, 1.0],
//             [2.0, 2.0, 2.0],
//             [3.0, 3.0, 3.0],
//         ]);
//         println!("m1 dim:{:?}", m1.dim);
//         println!("m1 {:?}", m1);
//         println!("m1 stride:{:?}", m1.dim.stride);
//         let m2 = arr(&[1.0, 2.0, 3.0]);

//         let m3 = m1 + &m2;
//         println!("m3:{:?}", m3);
//     }

//     #[test]
//     fn test_add_scalar() {
//         let m1 = mat(&[
//             [0.0, 0.0, 0.0],
//             [1.0, 1.0, 1.0],
//             [2.0, 2.0, 2.0],
//             [3.0, 3.0, 3.0],
//         ]);
//         // let m1 = m1 + 1.0;

//         let out = &m1 + 1.0;
//         // let m3 = m1 + 1;
//         println!("m1:{:?}", m1);
//         println!("m1:{:?}", out);
//     }

//     #[test]
//     fn test_sub() {
//         let m1 = mat(&[
//             [0.0, 0.0, 0.0],
//             [1.0, 1.0, 1.0],
//             [2.0, 2.0, 2.0],
//             [3.0, 3.0, 3.0],
//         ]);
//         println!("m1 dim:{:?}", m1.dim);
//         println!("m1 {:?}", m1.as_slice());
//         println!("m1 stride:{:?}", m1.dim.stride);
//         let m2 = arr(&[1.0, 2.0, 3.0]);

//         let m3 = &m1 - &m2;
//         println!("m3:{:?}", m3);
//     }

//     #[test]
//     fn test_mul() {
//         let m1 = mat(&[
//             [0.0, 0.0, 0.0],
//             [1.0, 1.0, 1.0],
//             [2.0, 2.0, 2.0],
//             [3.0, 3.0, 3.0],
//         ]);
//         println!("m1 dim:{:?}", m1.dim);
//         println!("m1 {:?}", m1);
//         println!("m1 stride:{:?}", m1.dim.stride);
//         let m2 = arr(&[1.0, 2.0, 3.0]);

//         let m3 = m1 * &m2;
//         println!("m3:{:?}", m3);
//     }

//     #[test]
//     fn test_div() {
//         let m1 = mat(&[
//             [0.0, 0.0, 0.0],
//             [1.0, 1.0, 1.0],
//             [2.0, 2.0, 2.0],
//             [3.0, 3.0, 3.0],
//         ]);
//         println!("m1 dim:{:?}", m1.dim);
//         println!("m1 {:?}", m1);
//         println!("m1 stride:{:?}", m1.dim.stride);
//         let m2 = arr(&[1.0, 2.0, 3.0]);

//         let m3 = m1 / &m2;
//         println!("m3:{:?}", m3);
//     }

//     #[test]
//     fn test_rem() {
//         let m1 = mat(&[
//             [0.0, 0.0, 0.0],
//             [1.0, 1.0, 1.0],
//             [2.0, 2.0, 2.0],
//             [3.0, 3.0, 3.0],
//         ]);
//         println!("m1 dim:{:?}", m1.dim);
//         println!("m1 {:?}", m1);
//         println!("m1 stride:{:?}", m1.dim.stride);
//         let m2 = arr(&[1.0, 2.0, 3.0]);

//         let m3 = m1 % m2;
//         println!("m3:{:?}", m3);
//     }

//     #[test]
//     fn test_sqrt() {
//         let a: f64 = 4.0;
//         let b = a.sqrt();
//     }

//     #[test]
//     fn test_eq() {
//         let m1 = mat(&[
//             [0.0, 0.0, 0.0],
//             [1.0, 1.0, 1.0],
//             [2.0, 2.0, 2.0],
//             [3.0, 3.0, 3.0],
//         ]);
//         let m2 = mat(&[
//             [0.0, 0.0, 0.0],
//             [1.0, 1.0, 1.0],
//             [2.0, 2.0, 2.0],
//             [3.0, 3.0, 3.0],
//         ]);
//         let m3 = arr(&[1.0, 2.0, 3.0]);

//         println!("{}", m1 == m2);
//         println!("{}", m2 == m3);
//     }
// }
