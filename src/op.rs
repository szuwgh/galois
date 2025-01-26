use crate::cuda::CudaCpy;
use crate::cuda::CudaMap;
use crate::cuda::CudaMap2;
use crate::cuda::CudaMatMul;
use crate::cuda::CudaMul;
use crate::cuda::CudaRmsNorm;
use crate::cuda::CudaRope;
use crate::error::GResult;
use crate::ggml_quants::QuantType;
use crate::CpuStorageView;
use crate::GError;
use crate::Storage;
use crate::StorageProto;
use crate::StorageView;
use crate::TensorProto;
use core::time;
use std::cell::UnsafeCell;
use std::cmp::min;
use std::collections::VecDeque;
use std::f32::INFINITY;
use std::ops::Neg;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
//use super::broadcast::{broadcasting_binary_op, general_broadcasting};
use crate::ggml_quants::BlockQ4_0;
use crate::CpuStorageSlice;
use crate::Dim;
use crate::Tensor;
use crate::TensorType;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::cell::OnceCell;
use core::simd::f32x32;
use half::f16;
use lazy_static::lazy_static;
use rayon::prelude::*;
use std::simd::num::SimdFloat;

const COEF_A: f32 = 0.044715;
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
const CACHE_LINE_SIZE: usize = 64;
const CACHE_LINE_SIZE_F32: usize = CACHE_LINE_SIZE / std::mem::size_of::<f32>();

const STEP: usize = 32;
const EPR: usize = 8;
const ARR: usize = STEP / EPR;

lazy_static! {
    pub(crate) static ref GLOBAL_CPU_DEVICE_CACHE: CpuStorageSliceCache =
        CpuStorageSliceCache::new();
}

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

#[inline]
fn can_repeat_rows(a: &[usize], b: &[usize]) -> bool {
    a == b
}

#[inline]
fn compute_gelu(v: f32) -> f32 {
    0.5 * v
        * (1.0 + f32::tanh((2.0f32 / std::f32::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
}

#[inline]
fn compute_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// #[inline]
// fn vec_scale_f32(n: usize, y: &mut [f32], v: f32) {
//     let n32 = n & !(STEP - 1);
//     let scale = std::simd::f32x32::splat(v);
//     y[..n32].chunks_exact_mut(STEP).for_each(|d| {
//         let va = std::simd::f32x32::from_slice(d);
//         let va = va * scale;
//         va.copy_to_slice(d);
//     });
//     for i in n32..n {
//         y[i] *= v;
//     }
// }

// #[inline]
// fn vec_silu_f32(n: usize, y: &mut [f32], x: &[f32]) {
//     for i in 0..n {
//         y[i] = GLOBAL_CPU_DEVICE_CACHE
//             .get_silu_cache(f16::from_f32(x[i]).to_bits() as usize)
//             .to_f32()
//     }
// }

pub enum UnaryOp {
    Abs,
    Sgn,
    Neg,
    Step,
    Tanh,
    Elu,
    Relu,
    Gelu,
    GeluQuick,
    Silu,
}

#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn vec_dot_f16(lhs: *const f16, rhs: *const f16, res: *mut f32, len: usize) {
    *res = 0.0f32;
    for i in 0..len {
        *res += ((*lhs.add(i)).to_f32() * (*rhs.add(i)).to_f32());
    }
}

#[cfg(target_feature = "avx2")]
#[inline(always)]
pub(crate) unsafe fn vec_dot_f16(x: *const f16, y: *const f16, c: *mut f32, k: usize) {
    let mut sumf = 0.0f32;
    let n32 = k & !(STEP - 1);

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let (mut x0, mut x1, mut x2, mut x3) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    let (mut y0, mut y1, mut y2, mut y3) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );

    for i in (0..n32).step_by(STEP) {
        x0 = _mm256_cvtph_ps(_mm_loadu_si128(x.add(i + 0) as *const __m128i));
        x1 = _mm256_cvtph_ps(_mm_loadu_si128(x.add(i + 8) as *const __m128i));
        x2 = _mm256_cvtph_ps(_mm_loadu_si128(x.add(i + 16) as *const __m128i));
        x3 = _mm256_cvtph_ps(_mm_loadu_si128(x.add(i + 24) as *const __m128i));

        y0 = _mm256_cvtph_ps(_mm_loadu_si128(y.add(i + 0) as *const __m128i));
        y1 = _mm256_cvtph_ps(_mm_loadu_si128(y.add(i + 8) as *const __m128i));
        y2 = _mm256_cvtph_ps(_mm_loadu_si128(y.add(i + 16) as *const __m128i));
        y3 = _mm256_cvtph_ps(_mm_loadu_si128(y.add(i + 24) as *const __m128i));

        sum0 = _mm256_add_ps(_mm256_mul_ps(x0, y0), sum0);
        sum1 = _mm256_add_ps(_mm256_mul_ps(x1, y1), sum1);
        sum2 = _mm256_add_ps(_mm256_mul_ps(x2, y2), sum2);
        sum3 = _mm256_add_ps(_mm256_mul_ps(x3, y3), sum3);
    }

    let sum01 = _mm256_add_ps(sum0, sum1);
    let sum23 = _mm256_add_ps(sum2, sum3);
    let sum0123 = _mm256_add_ps(sum01, sum23);

    let r4 = _mm_add_ps(
        _mm256_castps256_ps128(sum0123),
        _mm256_extractf128_ps(sum0123, 1),
    );
    let r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
    let r1 = _mm_add_ss(r2, _mm_movehdup_ps(r2));

    sumf = _mm_cvtss_f32(r1);

    // leftovers
    for i in n32..k {
        sumf += (*x.add(i)).to_f32() * (*y.add(i)).to_f32();
    }
    *c = sumf;
}

unsafe impl Sync for CpuStorageSliceCache {}

pub(crate) struct CpuStorageSliceCache {
    gelu_cache: OnceCell<Vec<f16>>,
    exp_cache: OnceCell<Vec<f16>>,
    silu_cache: OnceCell<Vec<f16>>,
}

impl CpuStorageSliceCache {
    fn new() -> CpuStorageSliceCache {
        CpuStorageSliceCache {
            gelu_cache: OnceCell::from(Self::init_gelu_cache()),
            exp_cache: OnceCell::from(Self::init_exp_cache()),
            silu_cache: OnceCell::from(Self::init_silu_cache()),
        }
    }

    pub(crate) fn get_gelu_cache(&self, i: usize) -> f16 {
        self.gelu_cache.get().unwrap()[i]
    }

    pub(crate) fn get_silu_cache(&self, i: usize) -> f16 {
        self.silu_cache.get().unwrap()[i]
    }

    pub(crate) fn get_exp_cache(&self, i: usize) -> f16 {
        self.exp_cache.get().unwrap()[i]
    }

    fn init_silu_cache() -> Vec<f16> {
        (0..1 << 16)
            .map(|x| {
                let v = f16::from_bits(x as u16).to_f32();
                f16::from_f32(compute_silu(v))
            })
            .collect()
    }

    fn init_gelu_cache() -> Vec<f16> {
        (0..1 << 16)
            .map(|x| {
                let v = f16::from_bits(x as u16).to_f32();
                f16::from_f32(compute_gelu(v))
            })
            .collect()
    }

    fn init_exp_cache() -> Vec<f16> {
        (0..1 << 16)
            .map(|x| {
                let v = f16::from_bits(x as u16).to_f32();
                f16::from_f32(v.exp())
            })
            .collect()
    }
}

fn simd_vec_add_f32(inp1: &[f32], inp2: &[f32], dst: &mut [f32]) {
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

fn simd_vec_mul_f32(inp1: &[f32], inp2: &[f32], dst: &mut [f32]) {
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

trait Op {
    fn a();
}

impl Op for Tensor {
    fn a() {}
}

struct Add;

impl Map2 for Add {
    const OP: &'static str = "add";

    fn f<T: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[T],
        inp1_d: &Dim,
        dst: &mut [T],
        dst_d: &Dim,
    ) -> GResult<()> {
        assert!(is_same_shape(inp0_d.shape(), dst_d.shape()));
        let time1 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let ith: usize = 0;
        let nth: usize = 1;

        let nr = inp0_d.nrows();
        let nc = inp0_d.dim1();

        let (ne00, ne01, ne02, ne03) = inp0_d.dim4();
        let (ne10, ne11, ne12, ne13) = inp1_d.dim4();

        // let (ne0, ne1, ne2, ne3) = dst_d.dim4();

        let (nb00, nb01, nb02, nb03) = inp0_d.stride_4d();
        let (nb10, nb11, nb12, nb13) = inp1_d.stride_4d();
        let (nb0, nb1, nb2, nb3) = dst_d.stride_4d();

        assert!(nb0 == 1);
        assert!(nb00 == 1);

        // rows per thread
        let dr = (nr + nth - 1) / nth;

        // row range for this thread
        let ir0 = 0; //dr * ith;
        let ir1 = nr; //min(ir0 + dr, nr);

        if nb10 == 1 {
            assert!(dst.len() % ne00 == 0);
            dst.par_chunks_mut(ne00)
                .enumerate()
                .for_each(|(ir, dst_chunk)| {
                    let i03 = ir / (ne02 * ne01);
                    let i02 = (ir - i03 * ne02 * ne01) / ne01;
                    let i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

                    let i13 = i03 % ne13;
                    let i12 = i02 % ne12;
                    let i11 = i01 % ne11;

                    let src0_pos = i03 * nb03 + i02 * nb02 + i01 * nb01;
                    let src1_pos = i13 * nb13 + i12 * nb12 + i11 * nb11;

                    let src0_ptr = &inp0[src0_pos..src0_pos + ne00];
                    let src1_ptr = &inp1[src1_pos..src1_pos + ne00];

                    T::vec_add(src0_ptr, src1_ptr, dst_chunk);
                });
            // (0..nr).into_iter().for_each(|ir| {
            //     let i03 = ir / (ne02 * ne01);
            //     let i02 = (ir - i03 * ne02 * ne01) / ne01;
            //     let i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            //     let i13 = i03 % ne13;
            //     let i12 = i02 % ne12;
            //     let i11 = i01 % ne11;

            //     let dst_pos = i03 * nb3 + i02 * nb2 + i01 * nb1;
            //     let src0_pos = i03 * nb03 + i02 * nb02 + i01 * nb01;
            //     let src1_pos = i13 * nb13 + i12 * nb12 + i11 * nb11;

            //     let dst_ptr = &mut dst[dst_pos..dst_pos + ne00];
            //     let src0_ptr = &inp0[src0_pos..src0_pos + ne00];
            //     let src1_ptr = &inp1[src1_pos..src1_pos + ne00];

            //     T::vec_add(src0_ptr, src1_ptr, dst_ptr);
            // });
        } else {
            // for j in 0..n {
            //     let dst_ptr = &mut dst[j * nb1..];
            //     let src0_ptr = &inp1[j * nb01..];
            //     for i in 0..nc {
            //         dst_ptr[i] = src0_ptr[i] + inp2[j * nb11 + i * nb10];
            //     }
            // }
            println!("不连续");
            todo!()

            // dst.par_chunks_mut(nb1)
            //     .enumerate()
            //     .for_each(|(j, dst_ptr)| {
            //         let src0_ptr = &inp0[j * nb01..];
            //         for i in 0..nc {
            //             dst_ptr[i] = src0_ptr[i] + inp1[j * nb11 + i * nb10];
            //         }
            //     });
        }
        let time2 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        println!("{} time:{} ns", Self::OP, time2 - time1);
        Ok(())
    }

    fn f_q_f32_f32<T: QuantType + TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[f32],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_q_x_f32<T: QuantType + TensorType, X: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[X],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }
    // fn f_x_y_x<X, Y>(
    //     &self,
    //     inp: &[X],
    //     inp_d: &Dim,
    //     k: &[Y],
    //     k_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }
}

struct Mul;

impl Map2 for Mul {
    const OP: &'static str = "mul";

    fn f<T: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[T],
        inp1_d: &Dim,
        dst: &mut [T],
        dst_d: &Dim,
    ) -> GResult<()> {
        println!(
            "{:?},{:?},{:?}",
            inp0_d.shape(),
            inp1_d.shape(),
            dst_d.shape()
        );
        assert!(
            // is_same_shape(inp1_d.shape(), inp2_d.shape())
            is_same_shape(inp0_d.shape(), dst_d.shape())
        );
        let time1 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let n = inp0_d.nrows();
        let nc = inp0_d.dim1();

        // let nb00 = inp1_d.stride_1d();

        // let nb10 = inp2_d.stride_1d();

        // let nb0 = dst_d.stride_1d();

        let (ne00, ne01, ne02, ne03) = inp0_d.dim4();
        let (ne10, ne11, ne12, ne13) = inp1_d.dim4();

        let (nb00, nb01, nb02, nb03) = inp0_d.stride_4d();

        let (nb10, nb11, nb12, nb13) = inp1_d.stride_4d();

        let (nb0, nb1, nb2, nb3) = dst_d.stride_4d();
        let nr = ne01 * ne02 * ne03;
        assert!(nb00 == 1);
        assert!(nb10 == 1);
        assert!(nb0 == 1);
        let ith: usize = 0;
        let nth: usize = 1;
        // simd_vec_mul_f32(inp1, inp2, dst);
        if nb10 == 1 {
            dst.par_chunks_mut(ne00)
                .enumerate()
                .for_each(|(ir, dst_chunk)| {
                    let i03 = ir / (ne02 * ne01);
                    let i02 = (ir - i03 * ne02 * ne01) / ne01;
                    let i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

                    let i13 = i03 % ne13;
                    let i12 = i02 % ne12;
                    let i11 = i01 % ne11;

                    let src0_pos = i03 * nb03 + i02 * nb02 + i01 * nb01;
                    let src1_pos = i13 * nb13 + i12 * nb12 + i11 * nb11;

                    let src0_ptr = &inp0[src0_pos..src0_pos + ne00];
                    let src1_ptr = &inp1[src1_pos..src1_pos + ne00];
                    // let i03 = ir / (ne02 * ne01);
                    // let i02 = (ir - i03 * ne02 * ne01) / ne01;
                    // let i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

                    // let i13 = i03 % ne13;
                    // let i12 = i02 % ne12;
                    // let i11 = i01 % ne11;

                    // // let dst_ptr = &mut dst[i03 * nb3 + i02 * nb2 + i01 * nb1..];
                    // let src0_ptr = &inp0[i03 * nb03 + i02 * nb02 + i01 * nb01..];
                    // let src1_ptr = &inp1[i13 * nb13 + i12 * nb12 + i11 * nb11..];

                    T::vec_mul(src0_ptr, src1_ptr, dst_chunk);
                });
        } else {
            todo!()
            // for j in 0..n {
            //     let dst_ptr = &mut dst[j * nb1..];
            //     let src0_ptr = &inp1[j * nb01..];
            //     for i in 0..nc {
            //         dst_ptr[i] = src0_ptr[i] + inp2[j * nb11 + i * nb10];
            //     }
            // }

            // dst.par_chunks_mut(nb1)
            //     .enumerate()
            //     .for_each(|(j, dst_ptr)| {
            //         let src0_ptr = &inp1[j * nb01..];
            //         for i in 0..nc {
            //             dst_ptr[i] = src0_ptr[i] * inp2[j * nb11 + i * nb10];
            //         }
            //     });
        }
        let time2 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        println!("{} time:{} ns", Self::OP, time2 - time1);
        Ok(())
    }

    fn f_q_f32_f32<T: QuantType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[f32],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_q_x_f32<T: QuantType + TensorType, X: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[X],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    // fn f_x_y_x<X: TensorType, Y: TensorType>(
    //     &self,
    //     inp0: &[X],
    //     inp0_d: &Dim,
    //     inp1: &[Y],
    //     inp1_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }
}

struct Gelu;

impl Map for Gelu {
    const OP: &'static str = "gelu";
    fn f<T: TensorType>(&self, inp: &[T], inp_d: &Dim, dst: &mut [T], dst_d: &Dim) -> GResult<()> {
        assert!(is_same_shape(inp_d.shape(), dst_d.shape()));
        dst.par_iter_mut().zip(inp.par_iter()).for_each(|(d, a)| {
            *d = T::from_x(
                GLOBAL_CPU_DEVICE_CACHE
                    .get_gelu_cache(f16::from_f32(a.to_f32()).to_bits() as usize),
            )
        });
        Ok(())
    }

    // fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()> {
    //     assert!(is_same_shape(inp_d.shape(), dst_d.shape()));
    //     dst.par_iter_mut().zip(inp.par_iter()).for_each(|(d, a)| {
    //         *d = GLOBAL_CPU_DEVICE_CACHE
    //             .get_gelu_cache(f16::from_f32(*a).to_bits() as usize)
    //             .to_f32()
    //     });
    //     Ok(())
    // }

    fn f_x_y<X: TensorType, Y: TensorType>(
        &self,
        inp: &[X],
        inp_d: &Dim,
        dst: &mut [Y],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    // fn f_f32_f16(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f16], dst_d: &Dim) -> GResult<()> {
    //     Ok(())
    // }
}

struct Conv1D1S;

impl Map2 for Conv1D1S {
    const OP: &'static str = "conv1d1s";

    fn f<T: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[T],
        inp1_d: &Dim,
        dst: &mut [T],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_q_f32_f32<T: QuantType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[f32],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_q_x_f32<T: QuantType + TensorType, X: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[X],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    // fn f_quant_num<T: QuantType, X>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[X],
    //     inp2_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_x_y_x<X, Y>(
    //     &self,
    //     inp: &[X],
    //     inp_d: &Dim,
    //     k: &[Y],
    //     k_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_quant_num2<T: QuantType, X, Y>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[X],
    //     inp2_d: &Dim,
    //     dst: &mut [Y],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_f16_f32(
    //     &self,
    //     kernel: &[f16],
    //     k_d: &Dim,
    //     inp: &[f32],
    //     inp_d: &Dim,
    //     dst: &mut [f32],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     let time1 = SystemTime::now()
    //         .duration_since(UNIX_EPOCH)
    //         .unwrap()
    //         .as_millis();
    //     let (ne00, ne01, ne02) = k_d.dim3();

    //     let (ne10, ne11) = inp_d.dim2();
    //     let (_, nb01, nb02) = k_d.stride_3d();
    //     let (_, nb11) = inp_d.stride_2d();

    //     let nb1 = dst_d.stride()[1];

    //     let ith = 0;
    //     let nth: usize = 1;

    //     let nk = ne00;
    //     let nh: i32 = (nk / 2) as i32;

    //     let ew0 = up32(ne01 as i32) as usize;

    //     let mut ker_f16: Vec<f16> = vec![f16::from_f32(0.0); ne02 * ew0 * ne00];

    //     for i02 in 0..ne02 {
    //         for i01 in 0..ne01 {
    //             let src_start = i02 * nb02 + i01 * nb01;
    //             //  let src_end = src_start + ne00;
    //             let src_slice = &kernel[src_start..];

    //             let dst_start = i02 * ew0 * ne00;
    //             // let dst_end = dst_start + ne00 * ew0;
    //             let dst_slice = &mut ker_f16[dst_start..];
    //             for i00 in 0..ne00 {
    //                 dst_slice[i00 * ew0 + i01] = src_slice[i00];
    //             }
    //         }
    //     }

    //     let mut inp_f16: Vec<f16> =
    //         vec![f16::from_f32(0.0); (ne10 + nh as usize) * ew0 * ne11 + ew0];
    //     for i11 in 0..ne11 {
    //         let src_chunk = &inp[i11 * nb11..];

    //         for i10 in 0..ne10 {
    //             let index = (i10 + nh as usize) * ew0 + i11;
    //             inp_f16[index] = f16::from_f32(src_chunk[i10]);
    //         }
    //     }

    //     // total rows in dst
    //     let nr = ne02;

    //     // rows per thread
    //     let dr = nr; //(nr + nth - 1) / nth;

    //     // row range for this thread
    //     let ir0 = 0; //dr * ith;
    //     let ir1 = std::cmp::min(ir0 + dr, nr);

    //     // unsafe {
    //     //     for i1 in ir0..ir1 {
    //     //         let dst_data = &mut dst[i1 * nb1..];

    //     //         for i0 in 0..ne10 {
    //     //             dst_data[i0] = 0.0;
    //     //             for k in -nh..=nh {
    //     //                 let mut v = 0.0f32;
    //     //                 let wdata1_idx =
    //     //                     ((i1 * ew0 * ne00) as i32 + (nh + k) * ew0 as i32) as usize; //kernel
    //     //                 let wdata2_idx = ((i0 as i32 + nh + k) * ew0 as i32) as usize; //src
    //     //                 let wdata1 = &ker_f16[wdata1_idx..wdata1_idx + ew0];
    //     //                 let wdata2 = &inp_f16[wdata2_idx..wdata2_idx + ew0];
    //     //                 unsafe { vec_dot_f16(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0) }
    //     //                 dst_data[i0] += v;
    //     //             }
    //     //         }
    //     //     }
    //     // }
    //     unsafe {
    //         dst.par_chunks_mut(nb1)
    //             .enumerate()
    //             .for_each(|(i1, dst_data)| {
    //                 for i0 in 0..ne10 {
    //                     dst_data[i0] = 0.0;
    //                     for k in -nh..=nh {
    //                         let mut v = 0.0f32;
    //                         let wdata1_idx =
    //                             ((i1 * ew0 * ne00) as i32 + (nh + k) * ew0 as i32) as usize; //kernel
    //                         let wdata2_idx = ((i0 as i32 + nh + k) * ew0 as i32) as usize; //src
    //                         let wdata1 = &ker_f16[wdata1_idx..wdata1_idx + ew0];
    //                         let wdata2 = &inp_f16[wdata2_idx..wdata2_idx + ew0];
    //                         vec_dot_f16(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0);
    //                         dst_data[i0] += v;
    //                     }
    //                 }
    //             });
    //     }
    //     let time2 = SystemTime::now()
    //         .duration_since(UNIX_EPOCH)
    //         .unwrap()
    //         .as_millis();
    //     println!("{} time:{} ms", Self::OP, time2 - time1);
    //     // (0..nr).into_par_iter().for_each(|i1| {
    //     //     let dst_data = &mut dst[i1 * nb1..];

    //     //     for i0 in 0..ne10 {
    //     //         dst_data[i0] = 0.0;
    //     //         for k in -nh..=nh {
    //     //             let mut v = 0.0f32;
    //     //             let wdata1_idx = ((i1 * ew0 * ne00) as i32 + (nh + k) * ew0 as i32) as usize; //kernel
    //     //             let wdata2_idx = ((i0 as i32 + nh + k) * ew0 as i32) as usize; //src
    //     //             let wdata1 = &ker_f16[wdata1_idx..wdata1_idx + ew0];
    //     //             let wdata2 = &inp_f16[wdata2_idx..wdata2_idx + ew0];
    //     //             unsafe { f16::vec_dot_f16(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0) }
    //     //             dst_data[i0] += v;
    //     //         }
    //     //     }
    //     // });

    //     Ok(())
    // }

    // fn f_quant_f32<T: QuantType>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[f32],
    //     inp2_d: &Dim,
    //     dst: &mut [f32],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }
}

struct Conv1D2S;

impl Map2 for Conv1D2S {
    const OP: &'static str = "conv1d2s";
    fn f<T: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[T],
        inp1_d: &Dim,
        dst: &mut [T],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_q_f32_f32<T: QuantType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[f32],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_q_x_f32<T: QuantType + TensorType, X: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[X],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    // fn f_x_y_x<X: TensorType, Y: TensorType>(
    //     &self,
    //     inp0: &[X],
    //     inp0_d: &Dim,
    //     inp1: &[Y],
    //     inp1_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }
    // fn f_quant_num<T: QuantType, X>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[X],
    //     inp2_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_x_y_x<X, Y>(
    //     &self,
    //     inp: &[X],
    //     inp_d: &Dim,
    //     k: &[Y],
    //     k_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_quant_num2<T: QuantType, X, Y>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[X],
    //     inp2_d: &Dim,
    //     dst: &mut [Y],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_f16_f32(
    //     &self,
    //     kernel: &[f16],
    //     k_d: &Dim,
    //     inp: &[f32],
    //     inp_d: &Dim,
    //     dst: &mut [f32],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     let (ne00, ne01, ne02) = k_d.dim3();

    //     let (ne10, ne11) = inp_d.dim2();
    //     let (nb00, nb01, nb02) = k_d.stride_3d();
    //     let (nb10, nb11) = inp_d.stride_2d();

    //     let nb1 = dst_d.stride()[1];

    //     let ith = 0;
    //     let nth: usize = 1;

    //     let nk = ne00;
    //     let nh: i32 = (nk / 2) as i32;

    //     let ew0 = up32(ne01 as i32) as usize;

    //     assert!(ne00 % 2 == 1);
    //     assert!(nb00 == 1);
    //     assert!(nb10 == 1);

    //     let mut ker_f16: Vec<f16> = vec![f16::from_f32(0.0); ne02 * ew0 * ne00];

    //     (0..ne02)
    //         .flat_map(|i02| (0..ne01).map(move |i01| (i02, i01)))
    //         .for_each(|(i02, i01)| {
    //             let src_start = i02 * nb02 + i01 * nb01;
    //             let src_slice = &kernel[src_start..];

    //             let dst_start = i02 * ew0 * ne00;
    //             let dst_slice = &mut ker_f16[dst_start..];

    //             (0..ne00).zip(src_slice.iter()).for_each(|(i00, &src_val)| {
    //                 dst_slice[i00 * ew0 + i01] = src_val;
    //             });
    //         });

    //     // for i02 in 0..ne02 {
    //     //     for i01 in 0..ne01 {
    //     //         let src_start = i02 * nb02 + i01 * nb01;
    //     //         //  let src_end = src_start + ne00;
    //     //         let src_slice = &kernel[src_start..];

    //     //         let dst_start = i02 * ew0 * ne00;
    //     //         // let dst_end = dst_start + ne00 * ew0;
    //     //         let dst_slice = &mut ker_f16[dst_start..];
    //     //         for i00 in 0..ne00 {
    //     //             dst_slice[i00 * ew0 + i01] = src_slice[i00];
    //     //         }
    //     //     }
    //     // }

    //     // Create a vector of (i02, i01) pairs
    //     // let indices: Vec<(usize, usize)> = (0..ne02)
    //     //     .flat_map(|i02| (0..ne01).map(move |i01| (i02, i01)))
    //     //     .collect();

    //     // indices.par_iter().for_each(|&(i02, i01)| {
    //     //     let src_start = i02 * nb02 + i01 * nb01;
    //     //     let src_slice = &kernel[src_start..];

    //     //     let dst_start = i02 * ew0 * ne00;
    //     //     let dst_slice = &mut ker_f16[dst_start..];

    //     //     (0..ne00).zip(src_slice.iter()).for_each(|(i00, &src_val)| {
    //     //         dst_slice[i00 * ew0 + i01] = src_val;
    //     //     });
    //     // });

    //     let mut inp_f16: Vec<f16> =
    //         vec![f16::from_f32(0.0); (ne10 + nh as usize) * ew0 * ne11 + ew0];

    //     for i11 in 0..ne11 {
    //         let src_chunk = &inp[i11 * nb11..];
    //         for i10 in 0..ne10 {
    //             let index = (i10 + nh as usize) * ew0 + i11;
    //             inp_f16[index] = f16::from_f32(src_chunk[i10]);
    //         }
    //     }
    //     let time1 = SystemTime::now()
    //         .duration_since(UNIX_EPOCH)
    //         .unwrap()
    //         .as_millis();
    //     dst.par_chunks_mut(nb1)
    //         .enumerate()
    //         .for_each(|(i1, dst_data)| {
    //             for i0 in (0..ne10).step_by(2) {
    //                 dst_data[i0 / 2] = 0.0;
    //                 for k in -nh..=nh {
    //                     let mut v = 0.0f32;
    //                     let wdata1_idx =
    //                         ((i1 * ew0 * ne00) as i32 + (nh + k) * ew0 as i32) as usize; //kernel
    //                     let wdata2_idx = ((i0 as i32 + nh + k) * ew0 as i32) as usize; //src
    //                     let wdata1 = &ker_f16[wdata1_idx..wdata1_idx + ew0];
    //                     let wdata2 = &inp_f16[wdata2_idx..wdata2_idx + ew0];
    //                     unsafe { vec_dot_f16(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0) }
    //                     dst_data[i0 / 2] += v;
    //                 }
    //             }
    //         });

    //     // total rows in dst
    //     // let nr = ne02;

    //     // // // rows per thread
    //     // let dr = (nr + nth - 1) / nth;

    //     // // row range for this thread
    //     // let ir0 = dr * ith;
    //     // let ir1 = std::cmp::min(ir0 + dr, nr);

    //     // for i1 in ir0..ir1 {
    //     //     let dst_data = &mut dst[i1 * nb1..];

    //     //     for i0 in (0..ne10).step_by(2) {
    //     //         dst_data[i0 / 2] = 0.0;
    //     //         for k in -nh..=nh {
    //     //             let mut v = 0.0f32;
    //     //             let wdata1_idx = ((i1 * ew0 * ne00) as i32 + (nh + k) * ew0 as i32) as usize; //kernel
    //     //             let wdata2_idx = ((i0 as i32 + nh + k) * ew0 as i32) as usize; //src
    //     //             let wdata1 = &ker_f16[wdata1_idx..wdata1_idx + ew0];
    //     //             let wdata2 = &inp_f16[wdata2_idx..wdata2_idx + ew0];
    //     //             unsafe { f16::vec_dot_f16(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0) }
    //     //             dst_data[i0 / 2] += v;
    //     //         }
    //     //     }
    //     // }
    //     let time2 = SystemTime::now()
    //         .duration_since(UNIX_EPOCH)
    //         .unwrap()
    //         .as_millis();
    //     println!("{} time:{} ms", Self::OP, time2 - time1);
    //     Ok(())
    // }

    // fn f_quant_f32<T: QuantType>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[f32],
    //     inp2_d: &Dim,
    //     dst: &mut [f32],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }
}

struct Repeat;

impl Map for Repeat {
    const OP: &'static str = "repeat";

    fn f_x_y<X: TensorType, Y: TensorType>(
        &self,
        inp: &[X],
        inp_d: &Dim,
        dst: &mut [Y],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

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

    // fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()> {
    //     self.f(inp, inp_d, dst, dst_d)
    // }

    // fn f_f32_f16(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f16], dst_d: &Dim) -> GResult<()> {
    //     Ok(())
    // }
}

// struct Norm;
// use std::time::{SystemTime, UNIX_EPOCH};
// impl Map for Norm {
//     const OP: &'static str = "norm";
//     fn f<T: TensorType>(&self, inp: &[T], inp_d: &Dim, dst: &mut [T], dst_d: &Dim) -> GResult<()> {
//         let time1 = SystemTime::now()
//             .duration_since(UNIX_EPOCH)
//             .unwrap()
//             .as_millis();
//         assert!(is_same_shape(inp_d.shape(), dst_d.shape()));
//         let (ne00, ne01, ne02, ne03) = inp_d.dim4();
//         let (_, nb01, nb02, nb03) = inp_d.stride_4d();
//         let (_, nb1, nb2, nb3) = dst_d.stride_4d();

//         let eps: f64 = 1e-5;
//         for i03 in 0..ne03 {
//             for i02 in 0..ne02 {
//                 for i01 in 0..ne01 {
//                     let x = &inp[i01 * nb01 + i02 * nb02 + i03 * nb03..];

//                     let mut mean: f64 = x[..ne00]
//                         .chunks_exact(32)
//                         .map(|chunk| {
//                             let v = f32x32::from_slice(chunk);
//                             v.reduce_sum() as f64
//                         })
//                         .sum();
//                     // for i00 in 0..ne00 {
//                     //     mean += x[i00] as f64;
//                     // }
//                     mean /= ne00 as f64;
//                     let y = &mut dst[i01 * nb1 + i02 * nb2 + i03 * nb3..];
//                     let v_mean = std::simd::f32x32::splat(mean as f32);
//                     // let mut sum2 = 0.0f64;
//                     let sum2: f64 = y[..ne00]
//                         .chunks_exact_mut(32)
//                         .zip(x[..ne00].chunks_exact(32))
//                         .map(|(d, a)| {
//                             let va = std::simd::f32x32::from_slice(a);
//                             let va = va - v_mean;
//                             va.copy_to_slice(d);
//                             (va * va).reduce_sum() as f64
//                         })
//                         .sum();

//                     // for i00 in 0..ne00 {
//                     //     let v = x[i00] as f64 - mean;
//                     //     y[i00] = v as f32;
//                     //     sum2 += v * v;
//                     // }
//                     let scale =
//                         std::simd::f32x32::splat((1.0 / (sum2 / ne00 as f64 + eps).sqrt()) as f32);
//                     y[..ne00].chunks_exact_mut(32).for_each(|d| {
//                         let va = std::simd::f32x32::from_slice(d);
//                         let va = va * scale;
//                         va.copy_to_slice(d);
//                     });
//                     // vec_scale_f32(ne00, y, scale);
//                 }
//             }
//         }
//         let time2 = SystemTime::now()
//             .duration_since(UNIX_EPOCH)
//             .unwrap()
//             .as_millis();
//         println!("{} time:{} ms", Self::OP, time2 - time1);
//         Ok(())
//     }

//     fn f_x_y<X: TensorType, Y: TensorType>(
//         &self,
//         inp: &[X],
//         inp_d: &Dim,
//         dst: &mut [Y],
//         dst_d: &Dim,
//     ) -> GResult<()> {
//         todo!()
//     }

//     // fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()> {}

//     // fn f_f32_f16(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f16], dst_d: &Dim) -> GResult<()> {
//     //     Ok(())
//     // }
// }

struct RmsNorm {
    eps: f32,
}

impl Map for RmsNorm {
    const OP: &'static str = "rms_norm";
    // fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()> {
    //     assert!(is_same_shape(inp_d.shape(), dst_d.shape()));

    // }

    fn f<T: TensorType>(&self, inp: &[T], inp_d: &Dim, dst: &mut [T], dst_d: &Dim) -> GResult<()> {
        let (ne00, ne01, ne02, ne03) = inp_d.dim4();
        let (_, nb01, nb02, nb03) = inp_d.stride_4d();
        let (_, nb1, nb2, nb3) = dst_d.stride_4d();
        let eps: f32 = self.eps;
        for i03 in 0..ne03 {
            for i02 in 0..ne02 {
                for i01 in 0..ne01 {
                    let x = &inp[i01 * nb01 + i02 * nb02 + i03 * nb03..];
                    let y = &mut dst[i01 * nb1 + i02 * nb2 + i03 * nb3..];
                    T::rms_norm(ne00, x, y, eps);

                    // let n32 = ne00 & !(STEP - 1);
                    // let mut mean = x[..n32]
                    //     .chunks_exact(STEP)
                    //     .map(|chunk| T::reduce_sum(chunk).to_f32())
                    //     .sum::<f32>();
                    // mean /= ne00 as f32;
                    // y[..ne00].copy_from_slice(&x[..ne00]);
                    // let scale = 1.0 / (mean + eps).sqrt();
                    // f32::vec_scale(ne00, y, scale);
                }
            }
        }
        Ok(())
    }

    fn f_x_y<X: TensorType, Y: TensorType>(
        &self,
        inp: &[X],
        inp_d: &Dim,
        dst: &mut [Y],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    // fn f_f32_f16(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f16], dst_d: &Dim) -> GResult<()> {
    //     todo!()
    // }
}

struct MatMulTask {
    ir0: (usize, usize),
    ir1: (usize, usize),
}

impl MatMulTask {
    fn ir110(&self) -> usize {
        self.ir1.0
    }

    fn ir111(&self) -> usize {
        self.ir1.1
    }

    fn ir010(&self) -> usize {
        self.ir0.0
    }

    fn ir011(&self) -> usize {
        self.ir0.1
    }
}

unsafe impl<'a> Send for UnsafeF32Slice<'a> {}
unsafe impl<'a> Sync for UnsafeF32Slice<'a> {}

struct UnsafeF32Slice<'a>(UnsafeCell<&'a mut [f32]>);

impl<'a> UnsafeF32Slice<'a> {
    fn new(f: &'a mut [f32]) -> UnsafeF32Slice {
        Self(UnsafeCell::new(f))
    }

    pub(crate) unsafe fn borrow_mut(&self) -> &'a mut [f32] {
        *self.0.get()
    }
}

struct MatMul;

impl Map2 for MatMul {
    const OP: &'static str = "matmul";

    // fn f_quant_num<T: QuantType, X>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[X],
    //     inp2_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_x_y_x<X, Y>(
    //     &self,
    //     inp: &[X],
    //     inp_d: &Dim,
    //     k: &[Y],
    //     k_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    fn f_q_f32_f32<T: QuantType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[f32],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        let time1 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        let block_size = T::VecDotType::BLCK_SIZE;
        let mut quant_list = vec![T::VecDotType::zero(); inp1.len() / block_size];
        let (ne00, ne01, ne02, ne03) = inp0_d.dim4();
        let (ne10, ne11, ne12, ne13) = inp1_d.dim4();

        let (ne0, ne1, ne2, ne3) = dst_d.dim4();

        let (nb00, nb01, nb02, nb03) = inp0_d.stride_4d();
        let (nb10, nb11, nb12, nb13) = inp1_d.stride_4d();
        let (nb0, nb1, nb2, nb3) = dst_d.stride_4d();
        let mut idx = 0;
        let row_size = ne10 / block_size;
        for i13 in 0..ne13 {
            for i12 in 0..ne12 {
                for i11 in 0..ne11 {
                    T::VecDotType::from_f32(
                        &inp1[i13 * nb13 + i12 * nb12 + i11 * nb11
                            ..i13 * nb13 + i12 * nb12 + i11 * nb11 + ne10],
                        &mut quant_list[idx..idx + row_size],
                    )?;
                    idx += row_size;
                }
            }
        }

        assert!(ne0 == ne01);
        assert!(ne1 == ne11);
        assert!(ne2 == ne12);
        assert!(ne3 == ne13);

        // we don't support permuted src0 or src1
        assert!(nb00 == 1);
        assert!(nb10 == 1);

        // dst cannot be transposed or permuted
        assert!(nb0 == 1);
        assert!(nb0 <= nb1);
        assert!(nb1 <= nb2);
        assert!(nb2 <= nb3);

        // let ith = 0;
        let nth = 4;

        // let matmul_tasks: Vec<MatMulTask> =
        //     .collect();

        // let nr0 = ne01; // src0 rows
        // let nr1 = ne11 * ne12 * ne13; // src1 rows
        // let row_size = ne10 / block_size;

        // //printf("nr0 = %lld, nr1 = %lld\n", nr0, nr1);

        // // distribute the thread work across the inner or outer loop based on which one is larger

        // let nth0 = if nr0 > nr1 { nth } else { 1 }; // parallelize by src0 rows
        // let nth1 = if nr0 > nr1 { 1 } else { nth }; // parallelize by src1 rows

        // let ith0 = ith % nth0;
        // let ith1 = ith / nth0;

        // let dr0 = (nr0 + nth0 - 1) / nth0;
        // let dr1 = (nr1 + nth1 - 1) / nth1;

        // let ir010 = dr0 * ith0;
        // let ir011 = min(ir010 + dr0, nr0);

        // let ir110 = dr1 * ith1;
        // let ir111 = min(ir110 + dr1, nr1);

        //printf("ir010 = %6lld, ir011 = %6lld, ir110 = %6lld, ir111 = %6lld\n", ir010, ir011, ir110, ir111);

        // threads with no work simply yield (not sure if it helps)
        // if (ir010 >= ir011 || ir110 >= ir111) {
        //     return Ok(());
        // }

        assert!(ne12 % ne02 == 0);
        assert!(ne13 % ne03 == 0);

        // block-tiling attempt
        let blck_0 = 16;
        let blck_1 = 16;

        // attempt to reduce false-sharing (does not seem to make a difference)
        //  float tmp[16];
        let r2 = ne12 / ne02;
        let r3 = ne13 / ne03;

        let unsafe_dst = UnsafeF32Slice::new(dst);

        (0..nth)
            .into_par_iter()
            .map(|ith| {
                let nr0 = ne01; // src0 rows
                let nr1 = ne11 * ne12 * ne13; // src1 rows
                let row_size = ne10 / block_size;
                let nth0 = if nr0 > nr1 { nth } else { 1 }; // parallelize by src0 rows
                let nth1 = if nr0 > nr1 { 1 } else { nth }; // parallelize by src1 rows

                let ith0 = ith % nth0;
                let ith1 = ith / nth0;

                let dr0 = (nr0 + nth0 - 1) / nth0;
                let dr1 = (nr1 + nth1 - 1) / nth1;

                let ir010 = dr0 * ith0;
                let ir011 = min(ir010 + dr0, nr0);

                let ir110 = dr1 * ith1;
                let ir111 = min(ir110 + dr1, nr1);
                MatMulTask {
                    ir0: (ir010, ir011),
                    ir1: (ir110, ir111),
                }
            })
            .for_each(|task| {
                for iir1 in (task.ir110()..task.ir111()).step_by(blck_1 as usize) {
                    for iir0 in (task.ir010()..task.ir011()).step_by(blck_0 as usize) {
                        let ir1_range = iir1..(iir1 + blck_1).min(task.ir111());
                        for ir1 in ir1_range {
                            let i13 = ir1 / (ne12 * ne11);
                            let i12 = (ir1 - i13 * ne12 * ne11) / ne11;
                            let i11 = ir1 - i13 * ne12 * ne11 - i12 * ne11;

                            // Broadcast src0 into src1
                            let i03 = i13 / r3;
                            let i02 = i12 / r2;

                            let i1 = i11;
                            let i2 = i12;
                            let i3 = i13;

                            let src0_row = &inp0[(i02 * nb02 + i03 * nb03)..];
                            // Calculate the offset for src1
                            let src1_col =
                                &quant_list[(i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size..];
                            let d = unsafe {
                                &mut unsafe_dst.borrow_mut()[(i1 * nb1 + i2 * nb2 + i3 * nb3)
                                    ..(i1 * nb1 + i2 * nb2 + i3 * nb3) + iir0 + blck_0]
                            };
                            assert!(ne00 % 32 == 0);
                            for ir0 in (iir0..(iir0 + blck_0).min(task.ir011())).map(|x| x as usize)
                            {
                                d[ir0] = T::vec_dot(
                                    &src0_row[ir0 * nb01..(ir0 * nb01 + ne00 / block_size)],
                                    &src1_col[..ne00 / block_size],
                                );
                            }
                        }
                    }
                }
            });

        // let ith = 0;
        // let nth = 1;

        // let nr = ne01 * ne02 * ne03;

        // // rows per thread
        // let dr = (nr + nth - 1) / nth;

        // // row range for this thread
        // let ir0 = dr * ith;
        // let ir1 = std::cmp::min(ir0 + dr, nr);

        // //   ggml_fp16_t * wdata = params->wdata;
        // if nb01 >= nb00 {
        //     for ir in ir0..ir1 {
        //         // src0 indices
        //         let i03 = ir / (ne02 * ne01);
        //         let i02 = (ir - i03 * ne02 * ne01) / ne01;
        //         let i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        //         let i13 = i03;
        //         let i12 = i02;

        //         let i0 = i01;
        //         let i2 = i02;
        //         let i3 = i03;

        //         let src0_row = &lhs[i01 * nb01 + i02 * nb02 + i03 * nb03..];
        //         let src1_col =
        //             &quant_list[(0 + i12 * ne11 + i13 * ne12 * ne11) * ne00 / block_size..];

        //         let dst_col = &mut dst[i0 * nb0 + 0 * nb1 + i2 * nb2 + i3 * nb3..];
        //         assert!(ne00 % 32 == 0);
        //         for ic in 0..ne11 {
        //             dst_col[ic * ne0] =
        //                 T::vec_dot(src0_row, &src1_col[ic * ne00 / block_size..], ne00)
        //         }
        //     }
        // } else {
        //     todo!()
        // }

        let time2 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        println!("quant {} time:{} ms", Self::OP, time2 - time1);
        return Ok(());
    }

    fn f_q_x_f32<T: QuantType + TensorType, X: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[X],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    // fn f_quant_num2<T: QuantType, X, Y>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[X],
    //     inp2_d: &Dim,
    //     dst: &mut [Y],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    fn f<T: TensorType>(
        &self,
        lhs: &[T],
        lhs_l: &Dim,
        rhs: &[T],
        rhs_l: &Dim,
        dst: &mut [T],
        dst_d: &Dim,
    ) -> GResult<()> {
        // let l_dim = lhs_l.shape();
        // let r_dim: &[usize] = rhs_l.shape();
        // let dim = l_dim.len();
        // if dim < 2 || r_dim.len() != dim {
        //     return Err(GError::ShapeMismatchBinaryOp {
        //         lhs: lhs_l.shape.clone(),
        //         rhs: rhs_l.shape.clone(),
        //         op: "matmul",
        //     });
        // }
        // let m = l_dim[dim - 2];
        // let k = l_dim[dim - 1];
        // let k2 = r_dim[dim - 2];
        // let n = r_dim[dim - 1];
        // // let mut c_dim = l_dim[..dim - 2].to_vec();
        // // c_dim.extend(&[m, n]);
        // // let c_n_dims = c_dim.len();
        // // let c_shape = Shape::from_vec(c_dim);
        // let batching: usize = l_dim[..dim - 2].iter().product();
        // let batching_b: usize = r_dim[..dim - 2].iter().product();
        // if k != k2 || batching != batching_b {
        //     return Err(GError::ShapeMismatchBinaryOp {
        //         lhs: lhs_l.shape.clone(),
        //         rhs: rhs_l.shape.clone(),
        //         op: "matmul",
        //     });
        // }
        // self.compute_mul_mat_use_gemm(lhs, lhs_l, &rhs, rhs_l, dst, (batching, m, n, k))?;
        todo!()
    }

    // fn f_quant_f32<T: QuantType>(
    //     &self,
    //     lhs: &[T],
    //     lhs_l: &Dim,
    //     rhs: &[f32],
    //     rhs_l: &Dim,
    //     dst: &mut [f32],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     let time1 = SystemTime::now()
    //         .duration_since(UNIX_EPOCH)
    //         .unwrap()
    //         .as_millis();

    //     let block_size = T::VecDotType::BLCK_SIZE;
    //     let mut quant_list = vec![T::VecDotType::zero(); rhs.len() / block_size];
    //     let (ne00, ne01, ne02, ne03) = lhs_l.dim4();
    //     let (ne10, ne11, ne12, ne13) = rhs_l.dim4();

    //     let (ne0, ne1, ne2, ne3) = dst_d.dim4();

    //     let (nb00, nb01, nb02, nb03) = lhs_l.stride_4d();
    //     let (nb10, nb11, nb12, nb13) = rhs_l.stride_4d();
    //     let (nb0, nb1, nb2, nb3) = dst_d.stride_4d();
    //     let mut idx = 0;
    //     let row_size = ne10 / block_size;
    //     for i13 in 0..ne13 {
    //         for i12 in 0..ne12 {
    //             for i11 in 0..ne11 {
    //                 T::VecDotType::from_f32(
    //                     &rhs[i13 * nb13 + i12 * nb12 + i11 * nb11
    //                         ..i13 * nb13 + i12 * nb12 + i11 * nb11 + ne10],
    //                     &mut quant_list[idx..idx + row_size],
    //                 )?;
    //                 idx += row_size;
    //             }
    //         }
    //     }

    //     assert!(ne0 == ne01);
    //     assert!(ne1 == ne11);
    //     assert!(ne2 == ne12);
    //     assert!(ne3 == ne13);

    //     // we don't support permuted src0 or src1
    //     assert!(nb00 == 1);
    //     assert!(nb10 == 1);

    //     // dst cannot be transposed or permuted
    //     assert!(nb0 == 1);
    //     assert!(nb0 <= nb1);
    //     assert!(nb1 <= nb2);
    //     assert!(nb2 <= nb3);

    //     let ith = 0;
    //     let nth = 1;

    //     let row_size = ne10 / block_size;

    //     let nr0 = ne01; // src0 rows
    //     let nr1 = ne11 * ne12 * ne13; // src1 rows

    //     //printf("nr0 = %lld, nr1 = %lld\n", nr0, nr1);

    //     // distribute the thread work across the inner or outer loop based on which one is larger

    //     let nth0 = 1; //nr0 > nr1 ? nth : 1; // parallelize by src0 rows
    //     let nth1 = 1; //nr0 > nr1 ? 1 : nth; // parallelize by src1 rows

    //     let ith0 = 0; //ith % nth0;
    //     let ith1 = 0; //ith / nth0;

    //     let dr0 = (nr0 + nth0 - 1) / nth0;
    //     let dr1 = (nr1 + nth1 - 1) / nth1;

    //     let ir010 = dr0 * ith0;
    //     let ir011 = std::cmp::min(ir010 + dr0, nr0);

    //     let ir110 = dr1 * ith1;
    //     let ir111 = std::cmp::min(ir110 + dr1, nr1);

    //     //printf("ir010 = %6lld, ir011 = %6lld, ir110 = %6lld, ir111 = %6lld\n", ir010, ir011, ir110, ir111);

    //     // threads with no work simply yield (not sure if it helps)
    //     // if (ir010 >= ir011 || ir110 >= ir111) {
    //     //     return Ok(());
    //     // }

    //     assert!(ne12 % ne02 == 0);
    //     assert!(ne13 % ne03 == 0);

    //     // block-tiling attempt
    //     let blck_0 = 16;
    //     let blck_1 = 16;

    //     // attempt to reduce false-sharing (does not seem to make a difference)
    //     //  float tmp[16];
    //     let r2 = ne12 / ne02;
    //     let r3 = ne13 / ne03;

    //     for iir1 in (ir110..ir111).step_by(blck_1 as usize) {
    //         for iir0 in (ir010..ir011).step_by(blck_0 as usize) {
    //             let ir1_range = iir1..(iir1 + blck_1).min(ir111);
    //             for ir1 in ir1_range {
    //                 let i13 = ir1 / (ne12 * ne11);
    //                 let i12 = (ir1 - i13 * ne12 * ne11) / ne11;
    //                 let i11 = ir1 - i13 * ne12 * ne11 - i12 * ne11;

    //                 // Broadcast src0 into src1
    //                 let i03 = i13 / r3;
    //                 let i02 = i12 / r2;

    //                 let i1 = i11;
    //                 let i2 = i12;
    //                 let i3 = i13;

    //                 let src0_row = &lhs[(0 + i02 * nb02 + i03 * nb03)..];
    //                 // Calculate the offset for src1
    //                 let src1_col = &quant_list[(i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size..];

    //                 let dst_col = &mut dst[(i1 * nb1 + i2 * nb2 + i3 * nb3)..];
    //                 assert!(ne00 % 32 == 0);
    //                 for ir0 in (iir0..(iir0 + blck_0).min(ir011)).map(|x| x as usize) {
    //                     dst_col[ir0] = T::vec_dot(
    //                         &src0_row[ir0 * nb01..(ir0 * nb01 + ne00 / block_size)],
    //                         &src1_col[..ne00 / block_size],
    //                     );
    //                 }
    //             }
    //         }
    //     }

    // let ith = 0;
    // let nth = 1;

    // let nr = ne01 * ne02 * ne03;

    // // rows per thread
    // let dr = (nr + nth - 1) / nth;

    // // row range for this thread
    // let ir0 = dr * ith;
    // let ir1 = std::cmp::min(ir0 + dr, nr);

    // //   ggml_fp16_t * wdata = params->wdata;
    // if nb01 >= nb00 {
    //     for ir in ir0..ir1 {
    //         // src0 indices
    //         let i03 = ir / (ne02 * ne01);
    //         let i02 = (ir - i03 * ne02 * ne01) / ne01;
    //         let i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

    //         let i13 = i03;
    //         let i12 = i02;

    //         let i0 = i01;
    //         let i2 = i02;
    //         let i3 = i03;

    //         let src0_row = &lhs[i01 * nb01 + i02 * nb02 + i03 * nb03..];
    //         let src1_col =
    //             &quant_list[(0 + i12 * ne11 + i13 * ne12 * ne11) * ne00 / block_size..];

    //         let dst_col = &mut dst[i0 * nb0 + 0 * nb1 + i2 * nb2 + i3 * nb3..];
    //         assert!(ne00 % 32 == 0);
    //         for ic in 0..ne11 {
    //             dst_col[ic * ne0] =
    //                 T::vec_dot(src0_row, &src1_col[ic * ne00 / block_size..], ne00)
    //         }
    //     }
    // } else {
    //     todo!()
    // }

    //     let time2 = SystemTime::now()
    //         .duration_since(UNIX_EPOCH)
    //         .unwrap()
    //         .as_millis();
    //     println!("quant {} time:{} ms", Self::OP, time2 - time1);
    //     return Ok(());
    // }

    // fn f_f16_f32(
    //     &self,
    //     lhs: &[f16],
    //     lhs_l: &Dim,
    //     rhs: &[f32],
    //     rhs_l: &Dim,
    //     dst: &mut [f32],
    //     dst_d1: &Dim,
    // ) -> GResult<()> {
    //     let time1 = SystemTime::now()
    //         .duration_since(UNIX_EPOCH)
    //         .unwrap()
    //         .as_millis();
    //     let rhs_f16: Vec<f16> = rhs.par_iter().map(|e| f16::from_f32(*e)).collect();

    //     let (ne00, ne01, ne02, ne03) = lhs_l.dim4();
    //     let (ne10, ne11, ne12, ne13) = rhs_l.dim4();

    //     let (ne0, ne1, ne2, ne3) = dst_d1.dim4();

    //     let (nb00, nb01, nb02, nb03) = lhs_l.stride_4d();
    //     let (nb10, nb11, nb12, nb13) = rhs_l.stride_4d();
    //     let (nb0, nb1, nb2, nb3) = dst_d1.stride_4d();
    //     //const int64_t ne   = ne0*ne1*ne2*ne3;

    //     let ith = 0;
    //     let nth = 1;

    //     let nr = ne01 * ne02 * ne03;

    //     // rows per thread
    //     let dr = (nr + nth - 1) / nth;

    //     // row range for this thread
    //     let ir0 = dr * ith;
    //     let ir1 = std::cmp::min(ir0 + dr, nr);

    //     //   ggml_fp16_t * wdata = params->wdata;

    //     for ir in ir0..ir1 {
    //         // src0 indices
    //         let i03 = ir / (ne02 * ne01);
    //         let i02 = (ir - i03 * ne02 * ne01) / ne01;
    //         let i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

    //         let i13 = i03;
    //         let i12 = i02;

    //         let i0 = i01;
    //         let i2 = i02;
    //         let i3 = i03;

    //         let src0_row = &lhs[i01 * nb01 + i02 * nb02 + i03 * nb03..];
    //         let src1_col = &rhs_f16[(0 + i12 * ne11 + i13 * ne12 * ne11) * ne00..];

    //         let dst_col = &mut dst[i0 * nb0 + 0 * nb1 + i2 * nb2 + i3 * nb3..];

    //         for ic in 0..ne11 {
    //             //  assert!(ne00 % 32 == 0);
    //             unsafe {
    //                 vec_dot_f16(
    //                     src0_row.as_ptr(),
    //                     src1_col[ic * ne00..].as_ptr(),
    //                     dst_col[ic * ne0..].as_mut_ptr(),
    //                     ne00,
    //                 )
    //             }
    //         }
    //     }
    //     let time2 = SystemTime::now()
    //         .duration_since(UNIX_EPOCH)
    //         .unwrap()
    //         .as_millis();
    //     println!("{} time:{} ms", Self::OP, time2 - time1);
    //     return Ok(());
    // }
}

struct Cpy;

impl Map for Cpy {
    const OP: &'static str = "copy";
    fn f<T: TensorType>(&self, inp: &[T], inp_d: &Dim, dst: &mut [T], dst_d: &Dim) -> GResult<()> {
        assert!(dst_d.ggml_is_contiguous());
        assert_eq!(inp_d.elem_count(), dst_d.elem_count());
        if inp_d.ggml_is_contiguous() {
            dst.copy_from_slice(inp);
            return Ok(());
        }

        let (ne00, ne01, ne02, ne03) = inp_d.dim4();
        let (nb00, nb01, nb02, nb03) = inp_d.stride_4d();
        // const int ne01 = src0->ne[1];
        // const int ne02 = src0->ne[2];
        // const int ne03 = src0->ne[3];

        // const size_t nb00 = src0->nb[0];
        // const size_t nb01 = src0->nb[1];
        // const size_t nb02 = src0->nb[2];
        // const size_t nb03 = src0->nb[3];
        if inp_d.stride()[0] == 1 {
            let mut id = 0;
            let rs = ne00 * nb00;

            for i03 in 0..ne03 {
                for i02 in 0..ne02 {
                    for i01 in 0..ne01 {
                        let src0_ptr = &inp[i01 * nb01 + i02 * nb02 + i03 * nb03..];
                        let dst_ptr = &mut dst[id * rs..];

                        dst_ptr[..rs].copy_from_slice(&src0_ptr[..rs]);
                        id += 1;
                    }
                }
            }
        } else {
            let mut id = 0;
            // float * dst_ptr = (float *) dst->data;
            for i03 in 0..ne03 {
                for i02 in 0..ne02 {
                    for i01 in 0..ne01 {
                        for i00 in 0..ne00 {
                            let src0_ptr = &inp[i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03];
                            dst[id] = *src0_ptr;
                            id += 1;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()> {
    //     self.f(inp, inp_d, dst, dst_d)
    // }

    fn f_x_y<X: TensorType, Y: TensorType>(
        &self,
        inp: &[X],
        inp_d: &Dim,
        dst: &mut [Y],
        dst_d: &Dim,
    ) -> GResult<()> {
        let (ne00, ne01, ne02, ne03) = inp_d.dim4();
        let (nb00, nb01, nb02, nb03) = inp_d.stride_4d();

        let nr = ne01;
        // number of rows per thread
        let dr = nr;
        // row range for this thread
        let ir0 = 0;
        let ir1 = dr;

        if dst_d.ggml_is_contiguous() {
            //连续
            if nb00 == 1 {
                let mut id: usize = 0;
                for i03 in 0..ne03 {
                    for i02 in 0..ne02 {
                        for i01 in 0..ne01 {
                            for i00 in 0..ne00 {
                                let src0_ptr =
                                    inp[i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03];
                                dst[id] = Y::from_x(src0_ptr);
                                id = id + 1;
                            }
                        }
                    }
                }
            } else {
                todo!()
            }
        } else {
            let (ne0, ne1, ne2, ne3) = dst_d.dim4();
            let (nb0, nb1, nb2, nb3) = dst_d.stride_4d();
            let mut i10 = 0;
            let mut i11 = 0;
            let mut i12 = 0;
            let mut i13 = 0;
            for i03 in 0..ne03 {
                for i02 in 0..ne02 {
                    i10 += ne00 * ir0;
                    while i10 >= ne0 {
                        i10 -= ne0;
                        i11 += 1;
                        if i11 == ne1 {
                            i11 = 0;
                            i12 += 1;
                            if i12 == ne2 {
                                i12 = 0;
                                i13 += 1;
                                if i13 == ne3 {
                                    i13 = 0;
                                }
                            }
                        }
                    }

                    for i01 in ir0..ir1 {
                        for i00 in 0..ne00 {
                            let src0_ptr = &inp[i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03];
                            let dst_ptr = &mut dst[i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3];

                            *dst_ptr = Y::from_x(*src0_ptr);
                            i10 += 1;
                            if i10 == ne0 {
                                i10 = 0;
                                i11 += 1;
                                if i11 == ne1 {
                                    i11 = 0;
                                    i12 += 1;
                                    if i12 == ne2 {
                                        i12 = 0;
                                        i13 += 1;
                                        if i13 == ne3 {
                                            i13 = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    i10 += ne00 * (ne01 - ir1);
                    while i10 >= ne0 {
                        i10 -= ne0;
                        i11 += 1;
                        if i11 == ne1 {
                            i11 = 0;
                            i12 += 1;
                            if i12 == ne2 {
                                i12 = 0;
                                i13 += 1;
                                if i13 == ne3 {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // fn f_f32_f16(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f16], dst_d: &Dim) -> GResult<()> {
    //     assert!(dst_d.ggml_is_contiguous());
    //     assert_eq!(inp_d.elem_count(), dst_d.elem_count());

    //     let mut id: usize = 0;
    //     // ggml_fp16_t *dst_ptr = (ggml_fp16_t *)dst->data;

    //     let (ne00, ne01, ne02, ne03) = inp_d.dim4();
    //     let (nb00, nb01, nb02, nb03) = inp_d.stride_4d();

    //     // dst.par_iter_mut()
    //     //     .zip(inp.par_iter())
    //     //     .for_each(|(d, a)| *d = f16::from_f32(*a));

    //     for i03 in 0..ne03 {
    //         for i02 in 0..ne02 {
    //             for i01 in 0..ne01 {
    //                 for i00 in 0..ne00 {
    //                     let src0_ptr = inp[i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03];
    //                     dst[id] = f16::from_f32(src0_ptr);
    //                     id = id + 1;
    //                 }
    //             }
    //         }
    //     }
    //     Ok(())
    // }
}

// struct FlashAttn;

// impl Map4 for FlashAttn {
//     const OP: &'static str = "flashattn";
//     fn f_f16_f16_f16_f32(
//         &self,
//         q: &[f16],
//         q_d: &Dim,
//         k: &[f16],
//         k_d: &Dim,
//         v: &[f16],
//         v_d: &Dim,
//         dst: &mut [f32],
//         dst_d: &Dim,
//     ) -> GResult<()> {
//         let (neq0, neq1, neq2, neq3) = q_d.dim4();
//         let (nek0, nek1) = k_d.dim2();
//         // const int neq0 = q->ne[0];
//         // const int neq1 = q->ne[1];
//         // const int neq2 = q->ne[2];
//         // const int neq3 = q->ne[3];

//         // const int nek2 = k->ne[2];
//         // const int nek3 = k->ne[3];

//         // const int nev0 = v->ne[0];
//         //const int nev1 = v->ne[1];

//         let (_, nev1) = v_d.dim2();
//         // const int nev2 = v->ne[2];
//         // const int nev3 = v->ne[3];

//         let (ne0, ne1) = dst_d.dim2();

//         // const int ne0 = dst->ne[0];
//         // const int ne1 = dst->ne[1];
//         // const int ne2  = dst->ne[2];
//         // const int ne3  = dst->ne[3];

//         let (nbk0, nbk1, nbk2, nbk3) = k_d.stride_4d();
//         let (nbq0, nbq1, nbq2, nbq3) = q_d.stride_4d();
//         let (nbv0, nbv1, nbv2, nbv3) = v_d.stride_4d();
//         let (nb0, nb1, nb2, nb3) = dst_d.stride_4d();

//         // const int nbk0 = k->nb[0];
//         // const int nbk1 = k->nb[1];
//         // const int nbk2 = k->nb[2];
//         // const int nbk3 = k->nb[3];

//         // const int nbq0 = q->nb[0];
//         // const int nbq1 = q->nb[1];
//         // const int nbq2 = q->nb[2];
//         // const int nbq3 = q->nb[3];

//         // const int nbv0 = v->nb[0];
//         // const int nbv1 = v->nb[1];
//         // const int nbv2 = v->nb[2];
//         // const int nbv3 = v->nb[3];

//         // const int nb0 = dst->nb[0];
//         // const int nb1 = dst->nb[1];
//         // const int nb2 = dst->nb[2];
//         // const int nb3 = dst->nb[3];

//         let ith = 0;
//         let nth = 1;

//         let D = neq0;
//         let N = neq1;
//         let P = nek1 - N;
//         let M = P + N;

//         assert!(ne0 == D);
//         assert!(ne1 == N);
//         assert!(P >= 0);

//         assert!(nbq0 == 1);
//         assert!(nbk0 == 1);
//         assert!(nbv0 == 1);

//         assert!(neq0 == D);
//         assert!(nek0 == D);
//         assert!(nev1 == D);

//         assert!(neq1 == N);
//         assert!(nek1 == N + P);
//         assert!(nev1 == D);

//         // dst cannot be transposed or permuted
//         assert!(nb0 == 1);
//         assert!(nb0 <= nb1);
//         assert!(nb1 <= nb2);
//         assert!(nb2 <= nb3);

//         let nr = neq1 * neq2 * neq3;

//         // rows per thread
//         let dr = (nr + nth - 1) / nth;

//         // row range for this thread
//         let ir0 = dr * ith;
//         let ir1 = std::cmp::min(ir0 + dr, nr);

//         let scale = (1.0 / (D as f64).sqrt()) as f32;

//         //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

//         let wdata = vec![
//             0u8;
//             M * std::mem::size_of::<f32>() //ith * (2 * M + CACHE_LINE_SIZE_F32)
//                 + M * std::mem::size_of::<f16>()
//         ];

//         for ir in ir0..ir1 {
//             // q indices
//             let iq3 = ir / (neq2 * neq1);
//             let iq2 = (ir - iq3 * neq2 * neq1) / neq1;
//             let iq1 = ir - iq3 * neq2 * neq1 - iq2 * neq1;

//             let Sf32 = unsafe {
//                 std::slice::from_raw_parts_mut(
//                     wdata.as_ptr() as *mut f32,
//                     wdata.len() / std::mem::size_of::<f32>(),
//                 )
//             };

//             let (S, S2) = Sf32.split_at_mut(M); //ith * (2 * M + CACHE_LINE_SIZE_F32) +

//             // let S = &mut Sf32[ith * (2 * M + CACHE_LINE_SIZE_F32)..];

//             for ic in 0..nek1 {
//                 // k indices
//                 let ik3 = iq3;
//                 let ik2 = iq2;
//                 let ik1 = ic;

//                 // S indices
//                 let i1 = ik1;

//                 unsafe {
//                     vec_dot_f16(
//                         k[ik1 * nbk1 + ik2 * nbk2 + ik3 * nbk3..].as_ptr(),
//                         q[iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3..].as_ptr(),
//                         S[i1..].as_mut_ptr(),
//                         neq0,
//                     )
//                 };
//             }

//             // scale
//             vec_scale_f32(nek1, S, scale);

//             if false {
//                 for i in P..M {
//                     if (i > P + iq1) {
//                         S[i] = -std::f32::INFINITY;
//                     }
//                 }
//             }

//             // softmax
//             {
//                 let mut max = -std::f32::INFINITY;
//                 for i in 0..M {
//                     max = max.max(S[i]) //f32::max(max, );
//                 }

//                 let mut sum: f32 = 0.0;

//                 //  let ss: u16 = 0;
//                 for i in 0..M {
//                     if S[i] == -std::f32::INFINITY {
//                         S[i] = 0.0;
//                     } else {
//                         //const float val = (S[i] == -INFINITY) ? 0.0 : exp(S[i] - max);
//                         let s = f16::from_f32(S[i] - max);
//                         // let ss: u16 = unsafe { std::mem::transmute(s) };
//                         let val = GLOBAL_CPU_DEVICE_CACHE
//                             .get_exp_cache(s.to_bits() as usize)
//                             .to_f32();
//                         sum += val;
//                         S[i] = val;
//                     }
//                 }

//                 assert!(sum > 0.0f32);

//                 sum = 1.0 / sum;
//                 vec_scale_f32(M, S, sum);
//             }

//             let S16 = unsafe { std::slice::from_raw_parts_mut(S2.as_ptr() as *mut f16, M) };

//             for i in 0..M {
//                 S16[i] = f16::from_f32(S[i]);
//             }

//             for ic in 0..nev1 {
//                 // dst indices
//                 let i1 = iq1;
//                 let i2 = iq2;
//                 let i3 = iq3;

//                 unsafe {
//                     vec_dot_f16(
//                         v[ic * nbv1 + i2 * nbv2 + i3 * nbv3..].as_ptr(),
//                         S16.as_ptr(),
//                         dst[ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3..].as_mut_ptr(),
//                         nek1,
//                     );
//                 }
//             }
//         }
//         println!("std::f32::INFINITY:{}", std::f32::INFINITY);

//         Ok(())
//     }
// }

struct Scale;

impl Map2 for Scale {
    const OP: &'static str = "scale";

    fn f_q_f32_f32<T: QuantType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[f32],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_q_x_f32<T: QuantType + TensorType, X: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[X],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    // fn f_x_y_x<X: TensorType, Y: TensorType>(
    //     &self,
    //     inp0: &[X],
    //     inp0_d: &Dim,
    //     inp1: &[Y],
    //     inp1_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }
    // fn f_quant_num<T: QuantType, X>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[X],
    //     inp2_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_x_y_x<X, Y>(
    //     &self,
    //     inp: &[X],
    //     inp_d: &Dim,
    //     k: &[Y],
    //     k_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_quant_num2<T: QuantType, X, Y>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[X],
    //     inp2_d: &Dim,
    //     dst: &mut [Y],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    fn f<T: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[T],
        inp1_d: &Dim,
        dst: &mut [T],
        dst_d: &Dim,
    ) -> GResult<()> {
        assert!(inp0_d.ggml_is_contiguous());
        assert!(dst_d.ggml_is_contiguous());
        assert!(is_same_shape(inp0_d.shape(), dst_d.shape()));
        assert!(inp1_d.is_scalar());

        // scale factor
        let v = inp1[0];
        println!("v:{:?}", v);

        let ith = 0;
        let nth = 1;

        let nc = inp0_d.dim1();
        let nr = inp0_d.nrows();

        // rows per thread
        let dr = (nr + nth - 1) / nth;

        // row range for this thread
        let ir0 = dr * ith;
        let ir1 = std::cmp::min(ir0 + dr, nr);

        let (_, nb01) = inp0_d.stride_2d();
        let (_, nb1) = dst_d.stride_2d();

        for i1 in ir0..ir1 {
            if dst.as_ptr() != inp0.as_ptr() {
                dst[i1 * nb1..i1 * nb1 + nc].copy_from_slice(&inp0[i1 * nb01..i1 * nb01 + nc]);
            }
            T::vec_scale(nc, &mut dst[i1 * nb1..], v);
        }
        Ok(())
        // assert!(inp0_d.ggml_is_contiguous());
        // assert!(dst_d.ggml_is_contiguous());
        // assert!(is_same_shape(inp0_d.shape(), dst_d.shape()));
        // assert!(inp1_d.is_scalar());

        // // scale factor
        // let v = inp1[0];

        // let ith = 0;
        // let nth = 1;

        // let nc = inp0_d.dim1();
        // let nr = inp0_d.nrows();

        // let (_, nb1) = dst_d.stride_2d();

        // // rows per thread
        // let dr = (nr + nth - 1) / nth;

        // // row range for this thread
        // let ir0 = dr * ith;
        // let ir1 = std::cmp::min(ir0 + dr, nr);

        // for i1 in ir0..ir1 {
        //     vec_scale_f32(nc, &mut dst[i1 * nb1..], v);
        // }
    }

    // fn f_quant_f32<T: QuantType>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[f32],
    //     inp2_d: &Dim,
    //     dst: &mut [f32],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_f32(
    //     &self,
    //     inp0: &[f32],
    //     inp0_d: &Dim,
    //     inp1: &[f32],
    //     inp1_d: &Dim,
    //     dst: &mut [f32],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     assert!(inp0_d.ggml_is_contiguous());
    //     assert!(dst_d.ggml_is_contiguous());
    //     assert!(is_same_shape(inp0_d.shape(), dst_d.shape()));
    //     assert!(inp1_d.is_scalar());

    //     // scale factor
    //     let v = inp1[0];
    //     println!("v:{}", v);

    //     let ith = 0;
    //     let nth = 1;

    //     let nc = inp0_d.dim1();
    //     let nr = inp0_d.nrows();

    //     // rows per thread
    //     let dr = (nr + nth - 1) / nth;

    //     // row range for this thread
    //     let ir0 = dr * ith;
    //     let ir1 = std::cmp::min(ir0 + dr, nr);

    //     let (_, nb01) = inp0_d.stride_2d();
    //     let (_, nb1) = dst_d.stride_2d();

    //     for i1 in ir0..ir1 {
    //         if dst.as_ptr() != inp0.as_ptr() {
    //             dst[i1 * nb1..i1 * nb1 + nc].copy_from_slice(&inp0[i1 * nb01..i1 * nb01 + nc]);
    //         }
    //         f32::vec_scale(nc, &mut dst[i1 * nb1..], v);
    //     }
    //     Ok(())
    // }
}

struct GetRows;

impl Map2 for GetRows {
    const OP: &'static str = "get_rows";

    // fn f_quant_num<T: QuantType, X>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[X],
    //     inp2_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }
    fn f<T: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[T],
        inp1_d: &Dim,
        dst: &mut [T],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_q_f32_f32<T: QuantType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[f32],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    // fn f_q_x_f32<T: QuantType + TensorType, X: TensorType, Y: TensorType>(
    //     &self,
    //     inp0: &[T],
    //     inp0_d: &Dim,
    //     inp1: &[X],
    //     inp1_d: &Dim,
    //     dst: &mut [f32],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    fn f_q_x_f32<T: QuantType + TensorType, X: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[X],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        let nc = inp0_d.dim1();
        let nr = inp1_d.elem_count();
        let (ne0, ne1) = dst_d.dim2();
        let (_, nbv1) = inp0_d.stride_2d();
        let (_, nbd1) = dst_d.stride_2d();
        assert!(ne0 == nc);
        assert!(ne1 == nr);
        assert!(inp0_d.stride_1d() == 1);
        for i in 0..nr {
            let r = inp1[i].to_usize();
            <T as QuantType>::to_f32(
                &inp0[r * nbv1..r * nbv1 + nc],
                &mut dst[i * nbd1..i * nbd1 + nc],
            )
        }
        Ok(())
    }

    // fn f_quant_num2<T: QuantType, X, Y>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[X],
    //     inp2_d: &Dim,
    //     dst: &mut [Y],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_Q40_I32_f32(
    //     &self,
    //     inp0: &[BlockQ4_0],
    //     inp0_d: &Dim,
    //     inp1: &[i32],
    //     inp1_d: &Dim,
    //     dst: &mut [f32],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     let nc = inp0_d.dim1();
    //     let nr = inp1_d.elem_count();
    //     let (ne0, ne1) = dst_d.dim2();
    //     let (_, nbv1) = inp0_d.stride_2d();
    //     let (_, nbd1) = dst_d.stride_2d();
    //     assert!(ne0 == nc);
    //     assert!(ne1 == nr);
    //     assert!(inp0_d.stride_1d() == 1);
    //     for i in 0..nr {
    //         let r = inp1[i] as usize;
    //         <BlockQ4_0 as QuantType>::to_f32(
    //             &inp0[r * nbv1..r * nbv1 + nc],
    //             &mut dst[i * nbd1..i * nbd1 + nc],
    //         )
    //     }
    //     Ok(())
    // }

    // fn f_quant_f32<T: QuantType>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[f32],
    //     inp2_d: &Dim,
    //     dst: &mut [f32],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }
}

struct RopeCustom {
    n_dims: usize,
    mode: i32,
    n_ctx: i32,
    freq_base: f32,
    freq_scale: f32,
    xpos_base: f32,
    xpos_down: bool,
}

impl Map2 for RopeCustom {
    const OP: &'static str = "rope_custom";
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

    fn f_x_y_x<X: TensorType, Y: TensorType>(
        &self,
        lhs: &[X],
        lhs_l: &Dim,
        rhs: &[Y],
        rhs_l: &Dim,
        dst: &mut [X],
        dst_d: &Dim,
    ) -> GResult<()> {
        let n_dims = self.n_dims;

        let mode = self.mode;
        let n_ctx = self.n_ctx;
        let freq_base = self.freq_base;
        let freq_scale = self.freq_scale;
        let xpos_base = self.xpos_base;
        let xpos_down = self.xpos_down;

        let (ne00, ne01, ne02, ne03) = lhs_l.dim4();
        let (ne10, ne11, ne12, ne13) = rhs_l.dim4();
        let (ne0, ne1, ne2, ne3) = dst_d.dim4();
        let (nb00, nb01, nb02, nb03) = lhs_l.stride_4d();
        let (nb10, nb11, nb12, nb13) = rhs_l.stride_4d();
        let (nb0, nb1, nb2, nb3) = dst_d.stride_4d();

        //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
        //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

        assert!(nb00 == 1);

        let ith = 0;
        let nth = 1;

        let nr = dst_d.nrows();

        assert!(n_dims <= ne0);
        assert!(n_dims % 2 == 0);

        // rows per thread
        let dr = nr; //(nr + nth - 1) / nth;

        // row range for this thread
        let ir0 = 0; //dr * ith;
        let ir1 = nr; //min(ir0 + dr, nr);

        // row index used to determine which thread to use
        let mut ir = 0;

        let theta_scale = (freq_base as f32).powf(-2.0 / n_dims as f32);
        println!("theta_scale{}", theta_scale);

        let is_neox = mode & 2 != 0;
        let is_glm = mode & 4 != 0;

        for i3 in 0..ne3 {
            for i2 in 0..ne2 {
                let p = rhs[i2].to_f32();
                for i1 in 0..ne1 {
                    if ir < ir0 {
                        ir += 1;
                        continue;
                    }

                    if ir > ir1 {
                        break;
                    }
                    ir += 1;
                    //ir += 1; // 如果通过了第二个条件，执行这行代码
                    let mut theta = freq_scale * p.to_f32();
                    if is_glm {
                        todo!()
                    } else if !is_neox {
                        for i0 in (0..ne0).step_by(2) {
                            let cos_theta = theta.cos();
                            let sin_theta = theta.sin();

                            let mut zeta = if xpos_base != 0.0 {
                                ((i0 as f32 + 0.4f32 * ne0 as f32) / (1.4f32 * ne0 as f32))
                                    .powf(p / xpos_base)
                            } else {
                                1.0
                            };
                            if xpos_down {
                                zeta = 1.0f32 / zeta;
                            }
                            theta *= theta_scale;
                            let src = &lhs[i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00..];
                            let dst_data = &mut dst[i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0..];

                            let x0 = src[0];
                            let x1 = src[1];
                            dst_data[0] = X::from_f32(
                                x0.to_f32() * cos_theta * zeta - x1.to_f32() * sin_theta * zeta,
                            );
                            dst_data[1] = X::from_f32(
                                x0.to_f32() * sin_theta * zeta + x1.to_f32() * cos_theta * zeta,
                            );
                        }
                    } else {
                        todo!()
                    }
                    // rest of the code goes here
                }
            }
        }

        Ok(())
    }

    fn f_q_f32_f32<T: QuantType + TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[f32],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_q_x_f32<T: QuantType + TensorType, X: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[X],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    // fn f_q_x_x<T: QuantType, X: TensorType>(
    //     &self,
    //     inp0: &[T],
    //     inp0_d: &Dim,
    //     inp1: &[X],
    //     inp1_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_q_x_y<T: QuantType + TensorType, X: TensorType, Y: TensorType>(
    //     &self,
    //     inp0: &[T],
    //     inp0_d: &Dim,
    //     inp1: &[X],
    //     inp1_d: &Dim,
    //     dst: &mut [Y],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_quant_num<T: QuantType, X>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[X],
    //     inp2_d: &Dim,
    //     dst: &mut [X],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_quant_num2<T: QuantType, X, Y>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[X],
    //     inp2_d: &Dim,
    //     dst: &mut [Y],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }

    // fn f_quant_f32<T: QuantType>(
    //     &self,
    //     inp1: &[T],
    //     inp1_d: &Dim,
    //     inp2: &[f32],
    //     inp2_d: &Dim,
    //     dst: &mut [f32],
    //     dst_d: &Dim,
    // ) -> GResult<()> {
    //     todo!()
    // }
}

struct Silu;
impl Map for Silu {
    const OP: &'static str = "silu";
    fn f<T: TensorType>(&self, inp: &[T], inp_d: &Dim, dst: &mut [T], dst_d: &Dim) -> GResult<()> {
        let ith = 0;
        let nth = 1;

        let nc = inp_d.dim1();
        let nr = inp_d.nrows();

        let (nb00, nb01) = inp_d.stride_2d();
        let (nb0, nb1) = dst_d.stride_2d();

        // rows per thread
        let dr = (nr + nth - 1) / nth;

        // row range for this thread
        let ir0 = dr * ith;
        let ir1 = min(ir0 + dr, nr);

        for i1 in ir0..ir1 {
            T::vec_silu(nc, &mut dst[i1 * nb1..], &inp[i1 * nb01..]);
        }

        Ok(())
    }
    // fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()> {
    //     let ith = 0;
    //     let nth = 1;

    //     let nc = inp_d.dim1();
    //     let nr = inp_d.nrows();

    //     let (nb00, nb01) = inp_d.stride_2d();
    //     let (nb0, nb1) = dst_d.stride_2d();

    //     // rows per thread
    //     let dr = (nr + nth - 1) / nth;

    //     // row range for this thread
    //     let ir0 = dr * ith;
    //     let ir1 = min(ir0 + dr, nr);

    //     for i1 in ir0..ir1 {
    //         vec_silu_f32(nc, &mut dst[i1 * nb1..], &inp[i1 * nb01..]);
    //     }

    //     Ok(())
    // }

    fn f_x_y<X: TensorType, Y: TensorType>(
        &self,
        inp: &[X],
        inp_d: &Dim,
        dst: &mut [Y],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }
}

struct SoftMax;

impl Map for SoftMax {
    const OP: &'static str = "soft max";
    fn f_x_y<X: TensorType, Y: TensorType>(
        &self,
        inp: &[X],
        inp_d: &Dim,
        dst: &mut [Y],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }
    fn f<T: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        dst: &mut [T],
        dst_d: &Dim,
    ) -> GResult<()> {
        let ith = 0;
        let nth = 1;

        let (nb00, nb01, nb02, nb03) = inp0_d.stride_4d();
        let (nb0, nb1, nb2, nb3) = dst_d.stride_4d();

        let nc = inp0_d.dim1();
        let nr = inp0_d.nrows();

        // rows per thread
        let dr = (nr + nth - 1) / nth;

        // row range for this thread
        let ir0 = dr * ith;
        let ir1 = min(ir0 + dr, nr);

        for i1 in ir0..ir1 {
            let sp = &inp0[i1 * nb01..];
            let dp = &mut dst[i1 * nb1..];

            // softmax
            {
                T::softmax(nc, dp, sp);
                // let mut max = -std::f32::INFINITY;
                // for i in 0..nc {
                //     max = max.max(sp[i]) //f32::max(max, );
                // }

                // let mut sum: f32 = 0.0;

                // //  let ss: u16 = 0;
                // for i in 0..nc {
                //     if sp[i] == -std::f32::INFINITY {
                //         dp[i] = 0.0;
                //     } else {
                //         //const float val = (S[i] == -INFINITY) ? 0.0 : exp(S[i] - max);
                //         let s = (sp[i] - max).to_f16();
                //         // let ss: u16 = unsafe { std::mem::transmute(s) };
                //         let val = GLOBAL_CPU_DEVICE_CACHE
                //             .get_exp_cache(s.to_bits() as usize)
                //             .to_f32();
                //         sum += val;
                //         dp[i] = val;
                //     }
                // }

                // assert!(sum > 0.0f32);

                // sum = 1.0 / sum;
                // vec_scale_f32(nc, dp, sum);
            }
        }
        Ok(())
    }
    // fn f_f32(&self, inp0: &[f32], inp0_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()> {
    //     let ith = 0;
    //     let nth = 1;

    //     let (nb00, nb01, nb02, nb03) = inp0_d.stride_4d();
    //     let (nb0, nb1, nb2, nb3) = dst_d.stride_4d();

    //     let nc = inp0_d.dim1();
    //     let nr = inp0_d.nrows();

    //     // rows per thread
    //     let dr = (nr + nth - 1) / nth;

    //     // row range for this thread
    //     let ir0 = dr * ith;
    //     let ir1 = min(ir0 + dr, nr);

    //     for i1 in ir0..ir1 {
    //         let sp = &inp0[i1 * nb01..];
    //         let dp = &mut dst[i1 * nb1..];

    //         // softmax
    //         {
    //             let mut max = -std::f32::INFINITY;
    //             for i in 0..nc {
    //                 max = max.max(sp[i]) //f32::max(max, );
    //             }

    //             let mut sum: f32 = 0.0;

    //             //  let ss: u16 = 0;
    //             for i in 0..nc {
    //                 if sp[i] == -std::f32::INFINITY {
    //                     dp[i] = 0.0;
    //                 } else {
    //                     //const float val = (S[i] == -INFINITY) ? 0.0 : exp(S[i] - max);
    //                     let s = (sp[i] - max).to_f16();
    //                     // let ss: u16 = unsafe { std::mem::transmute(s) };
    //                     let val = GLOBAL_CPU_DEVICE_CACHE
    //                         .get_exp_cache(s.to_bits() as usize)
    //                         .to_f32();
    //                     sum += val;
    //                     dp[i] = val;
    //                 }
    //             }

    //             assert!(sum > 0.0f32);

    //             sum = 1.0 / sum;
    //             vec_scale_f32(nc, dp, sum);
    //         }
    //     }
    //     Ok(())
    // }
}

pub fn galois_conv_1d_1s<T: TensorProto>(kernel: &T, src: &T, dst: &mut T) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (kernel.storage().view(), src.storage().view(), dst_device) {
        (StorageView::Cpu(k), StorageView::Cpu(s), StorageView::Cpu(mut d)) => {
            Conv1D1S.map(k, kernel.dim(), s, src.dim(), &mut d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_conv_1d_2s<T: TensorProto>(kernel: &T, src: &T, dst: &mut T) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (kernel.storage().view(), src.storage().view(), dst_device) {
        (StorageView::Cpu(k), StorageView::Cpu(s), StorageView::Cpu(mut d)) => {
            Conv1D2S.map(k, kernel.dim(), s, src.dim(), &mut d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_repeat<T: TensorProto>(src: &T, dst: &mut T) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (src.storage().view(), dst_device) {
        (StorageView::Cpu(s), StorageView::Cpu(mut d)) => {
            Repeat.map(s, src.dim(), &mut d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_add<T: TensorProto>(a: &T, b: &T, dst: &mut T) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (a.storage().view(), b.storage().view(), dst_device) {
        (StorageView::Cpu(s), StorageView::Cpu(k), StorageView::Cpu(mut d)) => {
            Add.map(s, a.dim(), k, b.dim(), &mut d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_matmul<X: TensorProto, Y: TensorProto, Z: TensorProto>(
    a: &X,
    b: &Y,
    dst: &mut Z,
) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (a.storage().view(), b.storage().view(), dst_device) {
        (StorageView::Cpu(a1), StorageView::Cpu(b1), StorageView::Cpu(mut d)) => {
            MatMul.map(a1, a.dim(), b1, &mut b.dim(), &mut d, dst_dim)?;
        }
        (StorageView::Gpu(s0), StorageView::Gpu(s1), StorageView::Gpu(mut d)) => {
            CudaMatMul.map(s0, a.dim(), s1, b.dim(), &mut d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_mul<T: TensorProto>(a: &T, b: &T, dst: &mut T) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (a.storage().view(), b.storage().view(), dst_device) {
        (StorageView::Cpu(a1), StorageView::Cpu(b1), StorageView::Cpu(mut d)) => {
            Mul.map(a1, a.dim(), b1, b.dim(), &mut d, dst_dim)?;
        }
        (StorageView::Gpu(s0), StorageView::Gpu(s1), StorageView::Gpu(mut d)) => {
            CudaMul.map(s0, a.dim(), s1, b.dim(), &mut d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_gelu<T: TensorProto>(src: &T, dst: &mut T) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (src.storage().view(), dst_device) {
        (StorageView::Cpu(s), StorageView::Cpu(mut d)) => {
            Gelu.map(s, src.dim(), &mut d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

// pub fn galois_norm(src: &Tensor, dst: &mut Tensor) -> GResult<()> {
//     let (dst_device, dst_dim) = dst.device_dim();
//     match (src.storage(), dst_device) {
//         (Device::Cpu(s), Device::Cpu(d)) => {
//             Norm.map(s, src.dim(), d, dst_dim)?;
//         }
//         _ => {
//             todo!()
//         }
//     }
//     Ok(())
// }

pub fn galois_rms_norm<T: TensorProto>(src: &T, dst: &mut T, eps: f32) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (src.storage().view(), dst_device) {
        (StorageView::Cpu(s), StorageView::Cpu(mut d)) => {
            RmsNorm { eps }.map(s, src.dim(), &mut d, dst_dim)?;
        }
        (StorageView::Gpu(s), StorageView::Gpu(mut d)) => {
            CudaRmsNorm { eps }.map(s, src.dim(), &mut d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_cpy<X: TensorProto, Y: TensorProto>(src: &X, dst: &mut Y) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (src.storage().view(), dst_device) {
        (StorageView::Cpu(s), StorageView::Cpu(mut d)) => {
            Cpy.map(s, src.dim(), &mut d, dst_dim)?;
        }
        (StorageView::Gpu(s), StorageView::Gpu(mut d)) => {
            CudaCpy.map(s, src.dim(), &mut d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_cont<T: TensorProto>(src: &T, dst: &mut T) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (src.storage().view(), dst_device) {
        (StorageView::Cpu(s), StorageView::Cpu(mut d)) => {
            Cpy.map(s, src.dim(), &mut d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

// pub fn galois_flash_attn(Q: &Tensor, K: &Tensor, V: &Tensor, dst: &mut Tensor) -> GResult<()> {
//     let (dst_device, dst_dim) = dst.device_dim();
//     match (Q.storage(), K.storage(), V.storage(), dst_device) {
//         (Device::Cpu(q), Device::Cpu(k), Device::Cpu(v), Device::Cpu(d)) => {
//             FlashAttn.map(q, Q.dim(), k, K.dim(), v, V.dim(), d, dst_dim)?;
//         }
//         _ => {
//             todo!()
//         }
//     }
//     Ok(())
// }

pub fn galois_soft_max<T: TensorProto>(src: &T, dst: &mut T) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (src.storage().view(), dst_device) {
        (StorageView::Cpu(s), StorageView::Cpu(mut d)) => {
            SoftMax.map(s, src.dim(), &mut d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_unary<T: TensorProto>(src: &T, dst: &mut T, op: UnaryOp) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (src.storage().view(), dst_device) {
        (StorageView::Cpu(s), StorageView::Cpu(mut d)) => match op {
            UnaryOp::Silu => {
                Silu.map(s, src.dim(), &mut d, dst_dim)?;
            }
            _ => {
                todo!()
            }
        },
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_scale<T: TensorProto>(src0: &T, src1: &T, dst: &mut T) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (src0.storage().view(), src1.storage().view(), dst_device) {
        (StorageView::Cpu(s0), StorageView::Cpu(s1), StorageView::Cpu(mut d)) => {
            Scale.map(s0, src0.dim(), s1, src1.dim(), &mut d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_get_rows<T: TensorProto>(src0: &T, src1: &T, dst: &mut T) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (src0.storage().view(), src1.storage().view(), dst_device) {
        (StorageView::Cpu(s0), StorageView::Cpu(s1), StorageView::Cpu(mut d)) => {
            GetRows.map(s0, src0.dim(), s1, src1.dim(), &mut d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub struct RopeCustomOption {
    pub n_dims: usize,
    pub mode: i32,
    pub n_ctx: i32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub xpos_base: f32,
    pub xpos_down: bool,
}

pub fn galois_rope_custom<X: TensorProto, Y: TensorProto, R: TensorProto>(
    op: RopeCustomOption,
    src0: &X,
    src1: &Y,
    dst: &mut R,
) -> GResult<()> {
    let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
    match (src0.storage().view(), src1.storage().view(), dst_device) {
        (StorageView::Cpu(s0), StorageView::Cpu(s1), StorageView::Cpu(mut d)) => {
            RopeCustom {
                n_dims: op.n_dims,
                mode: op.mode,
                n_ctx: op.n_ctx,
                freq_base: op.freq_base,
                freq_scale: op.freq_scale,
                xpos_base: op.xpos_base,
                xpos_down: op.xpos_down,
            }
            .map(s0, src0.dim(), s1, src1.dim(), &mut d, dst_dim)?;
        }
        (StorageView::Gpu(s0), StorageView::Gpu(s1), StorageView::Gpu(mut d)) => {
            CudaRope {
                n_dims: op.n_dims,
                mode: op.mode,
                n_ctx: op.n_ctx,
                freq_base: op.freq_base,
                freq_scale: op.freq_scale,
                xpos_base: op.xpos_base,
                xpos_down: op.xpos_down,
            }
            .map(s0, src0.dim(), s1, src1.dim(), &mut d, dst_dim)?;
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

    fn f_x_y<X: TensorType, Y: TensorType>(
        &self,
        inp: &[X],
        inp_d: &Dim,
        dst: &mut [Y],
        dst_d: &Dim,
    ) -> GResult<()>;

    //fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()>;

    //fn f_f32_f16(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f16], dst_d: &Dim) -> GResult<()>;

    fn map(
        &self,
        dev1: CpuStorageView,
        d1: &Dim,
        dst: &mut CpuStorageView,
        d3: &Dim,
    ) -> GResult<()> {
        match (dev1, dst) {
            (CpuStorageView::F16(v1), CpuStorageView::F16(d)) => {
                self.f(v1.as_slice(), d1, d.as_slice_mut(), d3)
            }
            (CpuStorageView::F32(v1), CpuStorageView::F32(d)) => {
                self.f(v1.as_slice(), d1, d.as_slice_mut(), d3)
            }
            (CpuStorageView::F32(v1), CpuStorageView::F16(d)) => {
                self.f_x_y(v1.as_slice(), d1, d.as_slice_mut(), d3)
            }
            _ => {
                todo!()
            }
        }
    }
}

trait Map2 {
    const OP: &'static str;
    fn f<T: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[T],
        inp1_d: &Dim,
        dst: &mut [T],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_x_y_x<X: TensorType, Y: TensorType>(
        &self,
        inp0: &[X],
        inp0_d: &Dim,
        inp1: &[Y],
        inp1_d: &Dim,
        dst: &mut [X],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_q_f32_f32<T: QuantType + TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[f32],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()>;

    fn f_q_x_f32<T: QuantType + TensorType, X: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[X],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()>;

    fn map(
        &self,
        dev1: CpuStorageView,
        d1: &Dim,
        dev2: CpuStorageView,
        d2: &Dim,
        dst: &mut CpuStorageView,
        d3: &Dim,
    ) -> GResult<()> {
        match (dev1, dev2, dst) {
            (CpuStorageView::F16(v1), CpuStorageView::F16(v2), CpuStorageView::F16(d)) => {
                self.f(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut(), d3)
            }
            (CpuStorageView::F32(v1), CpuStorageView::F32(v2), CpuStorageView::F32(d)) => {
                self.f(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut(), d3)
            }
            (CpuStorageView::F16(v1), CpuStorageView::F32(v2), CpuStorageView::F32(d)) => {
                self.f_q_f32_f32(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut(), d3)
            }
            (CpuStorageView::Q4_0(v1), CpuStorageView::I32(v2), CpuStorageView::F32(d)) => {
                self.f_q_x_f32(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut(), d3)
            }
            (CpuStorageView::Q4_0(v1), CpuStorageView::F32(v2), CpuStorageView::F32(d)) => {
                self.f_q_f32_f32(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut(), d3)
            }
            (CpuStorageView::Q6K(v1), CpuStorageView::F32(v2), CpuStorageView::F32(d)) => {
                self.f_q_f32_f32(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut(), d3)
            }
            (CpuStorageView::F32(v1), CpuStorageView::I32(v2), CpuStorageView::F32(d)) => {
                self.f_x_y_x(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut(), d3)
            }
            _ => {
                println!("not todo{}", Self::OP);
                todo!()
            }
        }
    }
}

// trait Map4 {
//     const OP: &'static str;

//     fn f_f16_f16_f16_f32(
//         &self,
//         inp1: &[f16],
//         inp1_d: &Dim,
//         inp2: &[f16],
//         inp2_d: &Dim,
//         inp3: &[f16],
//         inp3_d: &Dim,
//         dst: &mut [f32],
//         dst_d: &Dim,
//     ) -> GResult<()> {
//         todo!()
//     }

//     fn map(
//         &self,
//         dev1: &CpuStorageSlice,
//         d1: &Dim,
//         dev2: &CpuStorageSlice,
//         d2: &Dim,
//         dev3: &CpuStorageSlice,
//         d3: &Dim,
//         dst: &mut CpuStorageSlice,
//         d4: &Dim,
//     ) -> GResult<()> {
//         match (dev1, dev2, dev3, dst) {
//             (CpuStorageSlice::F16(v1), CpuStorageSlice::F16(v2), CpuStorageSlice::F16(v3), CpuStorageSlice::F32(d)) => self
//                 .f_f16_f16_f16_f32(
//                     v1.as_slice(),
//                     d1,
//                     v2.as_slice(),
//                     d2,
//                     v3.as_slice(),
//                     d3,
//                     d.as_slice_mut(),
//                     d4,
//                 ),
//             _ => {
//                 todo!()
//             }
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use half::vec;

    use super::*;
    use crate::Device;
    use crate::Shape;
    //  [[ F32(7.0), F32(10.0), F32(14.0)],
    //  [ F32(15.0), F32(22.0), F32(32.0)],
    //  [ F32(15.0), F32(22.0), F32(32.0)]]

    // [7.0, 10.0, 14.0, 15.0, 22.0, 32.0, 15.0, 22.0, 32.0]
    //     [[ F32(7.0), F32(15.0), F32(12.0)],
    //  [ F32(26.0), F32(16.0), F32(36.0)]]

    // [[ F32(7.0), F32(15.0)],
    // [ F32(12.0), F32(26.0)]]
    /*



    */
    #[test]
    fn test_matmul_f32() {
        let a =
            Tensor::mat_slice(&[[1.0f32, 3.0, 5.0], [7.0f32, 9.0, 11.0]], &Device::Cpu).unwrap();
        let b = Tensor::mat_slice(
            &[[2.0f32, 4.0], [6.0f32, 8.0], [10.0f32, 12.0]],
            &Device::Cpu,
        )
        .unwrap();
        println!("a{:?}", a.shape());
        println!("b{:?}", b.shape());
        // let ne = [
        //     a.ggml_shape()[1],
        //     b.ggml_shape()[1],
        //     a.ggml_shape()[2],
        //     b.ggml_shape()[3],
        // ];
        //println!("{:?}", ne);
        let v = vec![0.0f32; 15];
        let mut d = Tensor::from_vec(v, 2, Shape::from_array([2, 2]), &Device::Cpu).unwrap();
        // MatMul.map(
        //     &m1.storage(),
        //     &m1.dim,
        //     m2.storage(),
        //     &m2.dim,
        //     d.device_mut(),
        //     &d.dim,
        // );
        galois_matmul(&a, &b, &mut d).unwrap();
        println!("{:?}", d);
        println!("{:?}", unsafe { d.as_slice::<f32>() });
    }

    /**
     *
     * 1 2   5 11 17
     * 3 4   11 25 39
     *
     * 1 2 4
     * 3 5 6
     *
     * 1 3 5
     * 2 4 6
     *
     *
     *
     */
    #[test]
    fn test_matmul_f16() {
        // let m1 = mat(&[
        //     [f16::from_f32(1.0), f16::from_f32(2.0)],
        //     [f16::from_f32(3.0), f16::from_f32(4.0)],
        //     // [f16::from_f32(5.0), f16::from_f32(6.0)],
        // ]);

        /*
         * 如果要计算
         * a=[[1.0f32, 3.0, 5.0],
         *  [7.0f32, 9.0, 11.0]] *
         *
         * b=[[2.0f32, 4.0],
         * [6.0f32, 8.0],
         * [10.0f32, 12.0]]
         *
         * b对进行转置 transpose 然后相乘
         *
         * 结果再transpose
         *
         */
        let mut m1 = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(3.0),
                f16::from_f32(5.0),
                f16::from_f32(7.0),
                f16::from_f32(9.0),
                f16::from_f32(11.0),
            ],
            2,
            Shape::from_array([3, 2]),
            &Device::Cpu,
        )
        .unwrap();

        // let mut m2 = Tensor::from_vec(
        //     vec![2.0f32, 6.0, 10.0, 4.0, 8.0, 12.0],
        //     2,
        //     Shape::from_array([3, 2]),
        // );

        let mut m2 = Tensor::from_vec(
            vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0],
            2,
            Shape::from_array([2, 3]),
            &Device::Cpu,
        )
        .unwrap();

        println!("m1:{:?}", m1);
        println!("m2:{:?}", m2);

        //  let m2 = mat(&[[1.0f32, 2.0, 4.0], [3.0f32, 5.0, 6.0]]);

        let v = vec![0.0f32; 15];
        let mut d = Tensor::from_vec(v, 2, Shape::from_array([2, 2]), &Device::Cpu).unwrap();
        // MatMul.map(
        //     &m1.storage(),
        //     &m1.dim,
        //     m2.storage(),
        //     &m2.dim,
        //     d.device_mut(),
        //     &d.dim,
        // );
        galois_matmul(&m1, &m2, &mut d).unwrap();
        println!("{:?}", unsafe { d.as_slice::<f32>() });
    }

    #[test]
    fn test_simd_vec_add_f32() {
        let mut a = [1.0f32, 2.0, 3.0, 1.0f32, 2.0, 3.0];
        let mut b = [2.0f32, 3.0, 4.0, 1.0f32, 2.0, 3.0];
        let mut c = vec![0.0f32; a.len()];
        simd_vec_add_f32(&a, &b, &mut c);
        println!("{:?}", c);
    }
}
