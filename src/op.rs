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

#[inline]
fn up32(n: i32) -> i32 {
    (n + 31) & !31
}

#[inline]
fn vec_cpy<T: Copy>(n: usize, y: &mut [T], x: &[T]) {
    for i in 0..n {
        y[i] = x[i];
    }
}

#[inline]
fn is_same_shape(a: &[usize], b: &[usize]) -> bool {
    a == b
}

fn simd_vec_add_f32(inp1: &[f32], inp2: &[f32], dst: &mut [f32]) {
    dst.chunks_exact_mut(4)
        .zip(inp1.chunks_exact(4).cycle())
        .zip(inp2.chunks_exact(4).cycle())
        .for_each(|((d, ia), ib)| {
            let va = std::simd::f32x4::from_slice(ia);
            let vb = std::simd::f32x4::from_slice(ib);
            let vc = va + vb;
            vc.copy_to_slice(d);
        });
    let dst_length = dst.len();
    // 处理剩余部分
    let remainder_a = &inp1[inp1.len() / 4 * 4..];
    let remainder_b = &inp2[inp2.len() / 4 * 4..];
    let remainder_result = &mut dst[dst_length / 4 * 4..];
    for i in 0..remainder_a.len() {
        remainder_result[i] = remainder_a[i] + remainder_b[i];
    }
}

trait Op {
    fn a();
}

impl Op for Tensor {
    fn a() {}
}

struct Add;

impl Map2 for Add {
    const OP: &'static str = "add";
    fn f_f32(
        &self,
        inp1: &[f32],
        inp1_d: &Dim,
        inp2: &[f32],
        inp2_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        assert!(
            is_same_shape(inp1_d.shape(), inp2_d.shape())
                && is_same_shape(inp1_d.shape(), dst_d.shape())
        );
        simd_vec_add_f32(inp1, inp2, dst);
        Ok(())
    }
}

struct Conv1D;

impl Map2 for Conv1D {
    const OP: &'static str = "conv1d";
    fn f<T: TensorType>(
        &self,
        inp: &[T],
        inp_d: &Dim,
        k: &[T],
        k_d: &Dim,
        dst: &mut [T],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_f16(
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
        let (_, nb01, nb02) = k_d.stride_3d();
        let (_, nb11) = inp_d.stride_2d();

        let nb1 = dst_d.stride()[1];

        let ith = 0;
        let nth: usize = 1;

        let nk = ne00;
        let nh: i32 = (nk / 2) as i32;

        let ew0 = up32(ne01 as i32) as usize;

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
            }
        }

        let mut inp_f16: Vec<f16> =
            vec![f16::from_f32(0.0); (ne10 + nh as usize) * ew0 * ne11 + ew0];
        for i11 in 0..ne11 {
            let src_chunk = &inp[i11 * nb11..];

            for i10 in 0..ne10 {
                let index = (i10 + nh as usize) * ew0 + i11;
                inp_f16[index] = f16::from_f32(src_chunk[i10]);
            }
        }

        // total rows in dst
        let nr = ne02;

        // rows per thread
        let dr = (nr + nth - 1) / nth;

        // row range for this thread
        let ir0 = dr * ith;
        let ir1 = std::cmp::min(ir0 + dr, nr);

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
                    unsafe { f16::vec_dot_f16(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0) }
                    dst_data[i0] += v;
                }
            }
        }
        Ok(())
    }
}

struct Repeat;

impl Map for Repeat {
    const OP: &'static str = "repeat";
    fn f<T: TensorType>(&self, inp: &[T], inp_d: &Dim, dst: &mut [T], dst_d: &Dim) -> GResult<()> {
        let (inp_d0, inp_d1, inp_d2, inp_d3) = inp_d.dim4();
        let (dst_d0, dst_d1, dst_d2, dst_d3) = dst_d.dim4();

        assert!(inp_d2 == 1);
        assert!(inp_d3 == 1);
        assert!(dst_d2 == 1);
        assert!(dst_d3 == 1);

        let nc = dst_d0;
        let nr = dst_d1;
        let nc0 = inp_d0;
        let nr0 = inp_d1;

        let ncr = nc / nc0;
        let nrr = nr / nr0;

        let (dst_s0, dst_s1) = dst_d.stride_2d();
        let (inp_s0, inp_s1) = inp_d.stride_2d();

        assert!(dst_s0 == 1);
        assert!(inp_s0 == 1);

        for i in 0..nrr {
            for j in 0..ncr {
                for k in 0..nr0 {
                    vec_cpy(
                        nc0,
                        &mut dst[(i * nr0 + k) * (dst_s1) + j * nc0 * (dst_s0)..],
                        &inp[(k) * (inp_s1)..],
                    );
                }
            }
        }
        Ok(())
    }
}

trait Map2 {
    const OP: &'static str;
    fn f<T: TensorType>(
        &self,
        inp: &[T],
        inp_d: &Dim,
        k: &[T],
        k_d: &Dim,
        dst: &mut [T],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_f32(
        &self,
        inp: &[f32],
        inp_l: &Dim,
        k: &[f32],
        k_l: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_f16(
        &self,
        inp: &[f32],
        inp_l: &Dim,
        k: &[f16],
        k_l: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

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
                self.f(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut(), d3)
            }
            (CpuDevice::F32(v1), CpuDevice::F32(v2), CpuDevice::F32(d)) => {
                self.f_f32(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut(), d3)
            }
            (CpuDevice::F32(v1), CpuDevice::F16(v2), CpuDevice::F32(d)) => {
                self.f_f16(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut(), d3)
            }
            _ => {
                todo!()
            }
        }
    }
}

pub fn galois_conv_1d_1s(src: &Tensor, kernel: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (src.device(), kernel.device(), dst_device) {
        (Device::Cpu(s), Device::Cpu(k), Device::Cpu(d)) => {
            Conv1D.map(s, src.dim(), k, kernel.dim(), d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_repeat(src: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (src.device(), dst_device) {
        (Device::Cpu(s), Device::Cpu(d)) => {
            Repeat.map(s, src.dim(), d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_add(a: &Tensor, b: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (a.device(), b.device(), dst_device) {
        (Device::Cpu(s), Device::Cpu(k), Device::Cpu(d)) => {
            Add.map(s, a.dim(), k, b.dim(), d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

trait Map {
    const OP: &'static str;
    fn f<T: TensorType>(&self, inp: &[T], inp_d: &Dim, dst: &mut [T], dst_d: &Dim) -> GResult<()>;

    fn map(&self, dev1: &CpuDevice, d1: &Dim, dst: &mut CpuDevice, d3: &Dim) -> GResult<()> {
        match (dev1, dst) {
            (CpuDevice::F16(v1), CpuDevice::F16(d)) => {
                self.f(v1.as_slice(), d1, d.as_slice_mut(), d3)
            }
            (CpuDevice::F32(v1), CpuDevice::F32(d)) => {
                self.f(v1.as_slice(), d1, d.as_slice_mut(), d3)
            }
            _ => {
                todo!()
            }
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_vec_add_f32() {
        let mut a = [1.0f32, 2.0, 3.0, 4.0, 1.0, 4.0];
        let mut b = [2.0f32, 3.0, 4.0, 5.0, 2.0, 5.0];
        let mut c = vec![0.0f32; a.len()];
        simd_vec_add_f32(&a, &b, &mut c);
        println!("{:?}", c);
    }
}

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
