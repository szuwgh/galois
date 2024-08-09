use core::time;
use std::ops::Neg;

use crate::error::GResult;
use crate::ggml_quants::QuantType;
use crate::Device;
use crate::GError;
//use super::broadcast::{broadcasting_binary_op, general_broadcasting};
use crate::BlockV1_Q4_0;
use crate::CpuDevice;
use crate::Dim;
use crate::Shape;
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
    static ref GLOBAL_CPU_DEVICE_CACHE: CpuDeviceCache = CpuDeviceCache::new();
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
fn compute_gelu(v: f32) -> f32 {
    0.5 * v
        * (1.0 + f32::tanh((2.0f32 / std::f32::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
}

#[inline]
fn vec_scale_f32(n: usize, y: &mut [f32], v: f32) {
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

// #[inline]
// pub unsafe fn vld1q_f16(ptr: *const f16) -> float16x8_t {
//     core::arch::aarch64::vld1q_u16(ptr as *const u16) as float16x8_t
// }

// #[inline]
// pub unsafe fn vst1q_f16(ptr: *mut f16, a: float16x8_t) {
//     core::arch::aarch64::vst1q_u16(ptr as *mut u16, a as uint16x8_t);
// }

// unsafe {
//     let mut sumv0 = vdupq_n_f16(f16::ZERO.to_bits());
//     let mut sumv1 = vdupq_n_f16(f16::ZERO.to_bits());
//     let k_rounded = k - k % 16;
//     for ki in (0..k_rounded).step_by(16) {
//         let av0 = vld1q_f16(a.as_ptr().add(a_offset + ki));
//         let bv0 = vld1q_f16(b.as_ptr().add(b_offset + ki));
//         let av1 = vld1q_f16(a.as_ptr().add(a_offset + ki + 8));
//         let bv1 = vld1q_f16(b.as_ptr().add(b_offset + ki + 8));
//         sumv0 = vfmaq_f16(sumv0, av0, bv0);
//         sumv1 = vfmaq_f16(sumv1, av1, bv1);
//     }

//     let mut sum = myaarch64::vaddvq_f16(sumv0) + myaarch64::vaddvq_f16(sumv1);
//     for ki in k_rounded..k {
//         sum += (a.get_unchecked(a_offset + ki) * b.get_unchecked(b_offset + ki)).to_f32();
//     }
//     sum
// }

#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
unsafe fn vec_dot_f16(lhs: *const f16, rhs: *const f16, res: *mut f32, len: usize) {
    *res = 0.0f32;
    for i in 0..len {
        *res += ((*lhs.add(i)).to_f32() * (*rhs.add(i)).to_f32());
    }
}

#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn vec_dot_f16(x: *const f16, y: *const f16, c: *mut f32, k: usize) {
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
    // let mut sumf = 0.0f32;
    // let np = k & !(STEP - 1);

    // let mut sum = [_mm256_setzero_ps(); ARR];
    // let mut ax = [_mm256_setzero_ps(); ARR];
    // let mut ay = [_mm256_setzero_ps(); ARR];

    // for i in (0..np).step_by(STEP) {
    //     for j in 0..ARR {
    //         ax[j] = _mm256_cvtph_ps(_mm_loadu_si128(a_row.add(i + j * EPR) as *const __m128i));
    //         ay[j] = _mm256_cvtph_ps(_mm_loadu_si128(b_row.add(i + j * EPR) as *const __m128i));

    //         sum[j] = _mm256_add_ps(_mm256_mul_ps(ax[j], ay[j]), sum[j]);
    //     }
    // }

    // let mut offset = ARR >> 1;
    // for i in 0..offset {
    //     sum[i] = _mm256_add_ps(sum[i], sum[offset + i]);
    // }
    // offset >>= 1;
    // for i in 0..offset {
    //     sum[i] = _mm256_add_ps(sum[i], sum[offset + i]);
    // }
    // offset >>= 1;
    // for i in 0..offset {
    //     sum[i] = _mm256_add_ps(sum[i], sum[offset + i]);
    // }
    // let t0 = _mm_add_ps(
    //     _mm256_castps256_ps128(sum[0]),
    //     _mm256_extractf128_ps(sum[0], 1),
    // );
    // let t1 = _mm_hadd_ps(t0, t0);
    // sumf = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    // // leftovers
    // for i in np..k {
    //     sumf += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
    // }
    // *c = sumf;
}

unsafe impl Sync for CpuDeviceCache {}

struct CpuDeviceCache {
    gelu_cache: OnceCell<Vec<f16>>,
    exp_cache: OnceCell<Vec<f16>>,
}

impl CpuDeviceCache {
    fn new() -> CpuDeviceCache {
        CpuDeviceCache {
            gelu_cache: OnceCell::from(Self::init_gelu_cache()),
            exp_cache: OnceCell::from(Self::init_exp_cache()),
        }
    }

    fn get_gelu_cache(&self, i: usize) -> f16 {
        self.gelu_cache.get().unwrap()[i]
    }

    fn get_exp_cache(&self, i: usize) -> f16 {
        self.exp_cache.get().unwrap()[i]
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

        let ith: usize = 0;
        let nth: usize = 1;

        let n = inp1_d.nrows();
        let nc = inp1_d.dim1();

        let (nb00, nb01) = inp1_d.stride_2d();

        let (nb10, nb11) = inp2_d.stride_2d();

        let (nb0, nb1) = dst_d.stride_2d();

        assert!(nb0 == 1);
        assert!(nb00 == 1);

        if nb10 == 1 {
            simd_vec_add_f32(inp1, inp2, dst);
        } else {
            // for j in 0..n {
            //     let dst_ptr = &mut dst[j * nb1..];
            //     let src0_ptr = &inp1[j * nb01..];
            //     for i in 0..nc {
            //         dst_ptr[i] = src0_ptr[i] + inp2[j * nb11 + i * nb10];
            //     }
            // }

            dst.par_chunks_mut(nb1)
                .enumerate()
                .for_each(|(j, dst_ptr)| {
                    let src0_ptr = &inp1[j * nb01..];
                    for i in 0..nc {
                        dst_ptr[i] = src0_ptr[i] + inp2[j * nb11 + i * nb10];
                    }
                });
        }

        Ok(())
    }
}

struct Mul;

impl Map2 for Mul {
    const OP: &'static str = "mul";
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
        let time1 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        // let n = inp1_d.nrows();
        // let nc = inp1_d.dim1();

        let nb00 = inp1_d.stride_1d();

        let nb10 = inp2_d.stride_1d();

        let nb0 = dst_d.stride_1d();

        assert!(nb00 == 1);
        assert!(nb10 == 1);
        assert!(nb0 == 1);
        simd_vec_mul_f32(inp1, inp2, dst);
        // if nb10 == 1 {
        //     simd_vec_add_f32(inp1, inp2, dst);
        // } else {
        //     // for j in 0..n {
        //     //     let dst_ptr = &mut dst[j * nb1..];
        //     //     let src0_ptr = &inp1[j * nb01..];
        //     //     for i in 0..nc {
        //     //         dst_ptr[i] = src0_ptr[i] + inp2[j * nb11 + i * nb10];
        //     //     }
        //     // }

        //     dst.par_chunks_mut(nb1)
        //         .enumerate()
        //         .for_each(|(j, dst_ptr)| {
        //             let src0_ptr = &inp1[j * nb01..];
        //             for i in 0..nc {
        //                 dst_ptr[i] = src0_ptr[i] + inp2[j * nb11 + i * nb10];
        //             }
        //         });
        // }
        let time2 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        println!("{} time:{} ms", Self::OP, time2 - time1);
        Ok(())
    }
}

struct Gelu;

impl Map for Gelu {
    const OP: &'static str = "gelu";
    fn f<T: TensorType>(&self, inp: &[T], inp_d: &Dim, dst: &mut [T], dst_d: &Dim) -> GResult<()> {
        todo!()
    }

    fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()> {
        assert!(is_same_shape(inp_d.shape(), dst_d.shape()));
        dst.par_iter_mut().zip(inp.par_iter()).for_each(|(d, a)| {
            *d = GLOBAL_CPU_DEVICE_CACHE
                .get_gelu_cache(f16::from_f32(*a).to_bits() as usize)
                .to_f32()
        });
        Ok(())
    }

    fn f_f32_f16(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f16], dst_d: &Dim) -> GResult<()> {
        Ok(())
    }
}

struct Conv1D1S;

impl Map2 for Conv1D1S {
    const OP: &'static str = "conv1d1s";

    fn f_f16_f32(
        &self,
        kernel: &[f16],
        k_d: &Dim,
        inp: &[f32],
        inp_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        let time1 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
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
        let dr = nr; //(nr + nth - 1) / nth;

        // row range for this thread
        let ir0 = 0; //dr * ith;
        let ir1 = std::cmp::min(ir0 + dr, nr);

        // unsafe {
        //     for i1 in ir0..ir1 {
        //         let dst_data = &mut dst[i1 * nb1..];

        //         for i0 in 0..ne10 {
        //             dst_data[i0] = 0.0;
        //             for k in -nh..=nh {
        //                 let mut v = 0.0f32;
        //                 let wdata1_idx =
        //                     ((i1 * ew0 * ne00) as i32 + (nh + k) * ew0 as i32) as usize; //kernel
        //                 let wdata2_idx = ((i0 as i32 + nh + k) * ew0 as i32) as usize; //src
        //                 let wdata1 = &ker_f16[wdata1_idx..wdata1_idx + ew0];
        //                 let wdata2 = &inp_f16[wdata2_idx..wdata2_idx + ew0];
        //                 unsafe { vec_dot_f16(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0) }
        //                 dst_data[i0] += v;
        //             }
        //         }
        //     }
        // }
        unsafe {
            dst.par_chunks_mut(nb1)
                .enumerate()
                .for_each(|(i1, dst_data)| {
                    for i0 in 0..ne10 {
                        dst_data[i0] = 0.0;
                        for k in -nh..=nh {
                            let mut v = 0.0f32;
                            let wdata1_idx =
                                ((i1 * ew0 * ne00) as i32 + (nh + k) * ew0 as i32) as usize; //kernel
                            let wdata2_idx = ((i0 as i32 + nh + k) * ew0 as i32) as usize; //src
                            let wdata1 = &ker_f16[wdata1_idx..wdata1_idx + ew0];
                            let wdata2 = &inp_f16[wdata2_idx..wdata2_idx + ew0];
                            vec_dot_f16(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0);
                            dst_data[i0] += v;
                        }
                    }
                });
        }
        let time2 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        println!("{} time:{} ms", Self::OP, time2 - time1);
        // (0..nr).into_par_iter().for_each(|i1| {
        //     let dst_data = &mut dst[i1 * nb1..];

        //     for i0 in 0..ne10 {
        //         dst_data[i0] = 0.0;
        //         for k in -nh..=nh {
        //             let mut v = 0.0f32;
        //             let wdata1_idx = ((i1 * ew0 * ne00) as i32 + (nh + k) * ew0 as i32) as usize; //kernel
        //             let wdata2_idx = ((i0 as i32 + nh + k) * ew0 as i32) as usize; //src
        //             let wdata1 = &ker_f16[wdata1_idx..wdata1_idx + ew0];
        //             let wdata2 = &inp_f16[wdata2_idx..wdata2_idx + ew0];
        //             unsafe { f16::vec_dot_f16(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0) }
        //             dst_data[i0] += v;
        //         }
        //     }
        // });

        Ok(())
    }
}

struct Conv1D2S;

impl Map2 for Conv1D2S {
    const OP: &'static str = "conv1d2s";

    fn f_f16_f32(
        &self,
        kernel: &[f16],
        k_d: &Dim,
        inp: &[f32],
        inp_d: &Dim,
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

        assert!(ne00 % 2 == 1);
        assert!(nb00 == 1);
        assert!(nb10 == 1);

        let mut ker_f16: Vec<f16> = vec![f16::from_f32(0.0); ne02 * ew0 * ne00];

        (0..ne02)
            .flat_map(|i02| (0..ne01).map(move |i01| (i02, i01)))
            .for_each(|(i02, i01)| {
                let src_start = i02 * nb02 + i01 * nb01;
                let src_slice = &kernel[src_start..];

                let dst_start = i02 * ew0 * ne00;
                let dst_slice = &mut ker_f16[dst_start..];

                (0..ne00).zip(src_slice.iter()).for_each(|(i00, &src_val)| {
                    dst_slice[i00 * ew0 + i01] = src_val;
                });
            });

        // for i02 in 0..ne02 {
        //     for i01 in 0..ne01 {
        //         let src_start = i02 * nb02 + i01 * nb01;
        //         //  let src_end = src_start + ne00;
        //         let src_slice = &kernel[src_start..];

        //         let dst_start = i02 * ew0 * ne00;
        //         // let dst_end = dst_start + ne00 * ew0;
        //         let dst_slice = &mut ker_f16[dst_start..];
        //         for i00 in 0..ne00 {
        //             dst_slice[i00 * ew0 + i01] = src_slice[i00];
        //         }
        //     }
        // }

        // Create a vector of (i02, i01) pairs
        // let indices: Vec<(usize, usize)> = (0..ne02)
        //     .flat_map(|i02| (0..ne01).map(move |i01| (i02, i01)))
        //     .collect();

        // indices.par_iter().for_each(|&(i02, i01)| {
        //     let src_start = i02 * nb02 + i01 * nb01;
        //     let src_slice = &kernel[src_start..];

        //     let dst_start = i02 * ew0 * ne00;
        //     let dst_slice = &mut ker_f16[dst_start..];

        //     (0..ne00).zip(src_slice.iter()).for_each(|(i00, &src_val)| {
        //         dst_slice[i00 * ew0 + i01] = src_val;
        //     });
        // });

        let mut inp_f16: Vec<f16> =
            vec![f16::from_f32(0.0); (ne10 + nh as usize) * ew0 * ne11 + ew0];

        for i11 in 0..ne11 {
            let src_chunk = &inp[i11 * nb11..];
            for i10 in 0..ne10 {
                let index = (i10 + nh as usize) * ew0 + i11;
                inp_f16[index] = f16::from_f32(src_chunk[i10]);
            }
        }
        let time1 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        dst.par_chunks_mut(nb1)
            .enumerate()
            .for_each(|(i1, dst_data)| {
                for i0 in (0..ne10).step_by(2) {
                    dst_data[i0 / 2] = 0.0;
                    for k in -nh..=nh {
                        let mut v = 0.0f32;
                        let wdata1_idx =
                            ((i1 * ew0 * ne00) as i32 + (nh + k) * ew0 as i32) as usize; //kernel
                        let wdata2_idx = ((i0 as i32 + nh + k) * ew0 as i32) as usize; //src
                        let wdata1 = &ker_f16[wdata1_idx..wdata1_idx + ew0];
                        let wdata2 = &inp_f16[wdata2_idx..wdata2_idx + ew0];
                        unsafe { vec_dot_f16(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0) }
                        dst_data[i0 / 2] += v;
                    }
                }
            });

        // total rows in dst
        // let nr = ne02;

        // // // rows per thread
        // let dr = (nr + nth - 1) / nth;

        // // row range for this thread
        // let ir0 = dr * ith;
        // let ir1 = std::cmp::min(ir0 + dr, nr);

        // for i1 in ir0..ir1 {
        //     let dst_data = &mut dst[i1 * nb1..];

        //     for i0 in (0..ne10).step_by(2) {
        //         dst_data[i0 / 2] = 0.0;
        //         for k in -nh..=nh {
        //             let mut v = 0.0f32;
        //             let wdata1_idx = ((i1 * ew0 * ne00) as i32 + (nh + k) * ew0 as i32) as usize; //kernel
        //             let wdata2_idx = ((i0 as i32 + nh + k) * ew0 as i32) as usize; //src
        //             let wdata1 = &ker_f16[wdata1_idx..wdata1_idx + ew0];
        //             let wdata2 = &inp_f16[wdata2_idx..wdata2_idx + ew0];
        //             unsafe { f16::vec_dot_f16(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0) }
        //             dst_data[i0 / 2] += v;
        //         }
        //     }
        // }
        let time2 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        println!("{} time:{} ms", Self::OP, time2 - time1);
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

    fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()> {
        self.f(inp, inp_d, dst, dst_d)
    }

    fn f_f32_f16(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f16], dst_d: &Dim) -> GResult<()> {
        Ok(())
    }
}

struct Norm;
use std::time::{SystemTime, UNIX_EPOCH};
impl Map for Norm {
    const OP: &'static str = "norm";
    fn f<T: TensorType>(&self, inp: &[T], inp_d: &Dim, dst: &mut [T], dst_d: &Dim) -> GResult<()> {
        Ok(())
    }

    fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()> {
        let time1 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        assert!(is_same_shape(inp_d.shape(), dst_d.shape()));
        let (ne00, ne01, ne02, ne03) = inp_d.dim4();
        let (_, nb01, nb02, nb03) = inp_d.stride_4d();
        let (_, nb1, nb2, nb3) = dst_d.stride_4d();

        let eps: f64 = 1e-5;
        for i03 in 0..ne03 {
            for i02 in 0..ne02 {
                for i01 in 0..ne01 {
                    let x = &inp[i01 * nb01 + i02 * nb02 + i03 * nb03..];

                    let mut mean: f64 = x[..ne00]
                        .chunks_exact(32)
                        .map(|chunk| {
                            let v = f32x32::from_slice(chunk);
                            v.reduce_sum() as f64
                        })
                        .sum();
                    // for i00 in 0..ne00 {
                    //     mean += x[i00] as f64;
                    // }
                    mean /= ne00 as f64;
                    let y = &mut dst[i01 * nb1 + i02 * nb2 + i03 * nb3..];
                    let v_mean = std::simd::f32x32::splat(mean as f32);
                    // let mut sum2 = 0.0f64;
                    let sum2: f64 = y[..ne00]
                        .chunks_exact_mut(32)
                        .zip(x[..ne00].chunks_exact(32))
                        .map(|(d, a)| {
                            let va = std::simd::f32x32::from_slice(a);
                            let va = va - v_mean;
                            va.copy_to_slice(d);
                            (va * va).reduce_sum() as f64
                        })
                        .sum();

                    // for i00 in 0..ne00 {
                    //     let v = x[i00] as f64 - mean;
                    //     y[i00] = v as f32;
                    //     sum2 += v * v;
                    // }
                    let scale =
                        std::simd::f32x32::splat((1.0 / (sum2 / ne00 as f64 + eps).sqrt()) as f32);
                    y[..ne00].chunks_exact_mut(32).for_each(|d| {
                        let va = std::simd::f32x32::from_slice(d);
                        let va = va * scale;
                        va.copy_to_slice(d);
                    });
                    // vec_scale_f32(ne00, y, scale);
                }
            }
        }
        let time2 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        println!("{} time:{} ms", Self::OP, time2 - time1);
        Ok(())
    }

    fn f_f32_f16(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f16], dst_d: &Dim) -> GResult<()> {
        Ok(())
    }
}

struct RmsNorm;

impl Map for RmsNorm {
    const OP: &'static str = "rms_norm";
    fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()> {
        assert!(is_same_shape(inp_d.shape(), dst_d.shape()));
        let (ne00, ne01, ne02, ne03) = inp_d.dim4();
        let (_, nb01, nb02, nb03) = inp_d.stride_4d();
        let (_, nb1, nb2, nb3) = dst_d.stride_4d();
        let eps: f32 = 1e-5;
        for i03 in 0..ne03 {
            for i02 in 0..ne02 {
                for i01 in 0..ne01 {
                    let x = &inp[i01 * nb01 + i02 * nb02 + i03 * nb03..];
                    let n32 = ne00 & !(STEP - 1);
                    let mut mean: f32 = x[..n32]
                        .chunks_exact(STEP)
                        .map(|chunk| {
                            let mut v = f32x32::from_slice(chunk);
                            v *= v;
                            v.reduce_sum()
                        })
                        .sum();
                    mean /= ne00 as f32;
                    let y = &mut dst[i01 * nb1 + i02 * nb2 + i03 * nb3..];
                    y[..ne00].copy_from_slice(&x[..ne00]);
                    let scale = 1.0 / (mean + eps).sqrt();
                    vec_scale_f32(ne00, y, scale);
                }
            }
        }
        Ok(())
    }

    fn f<T: TensorType>(&self, inp: &[T], inp_d: &Dim, dst: &mut [T], dst_d: &Dim) -> GResult<()> {
        todo!()
    }

    fn f_f32_f16(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f16], dst_d: &Dim) -> GResult<()> {
        todo!()
    }
}

struct MatMul;

impl MatMul {
    fn can_use_gemm(&self, lhs_l: &Dim, rhs_l: &Dim) -> bool {
        // TODO: find the optimal values for these
        if lhs_l.ggml_is_contiguous() && rhs_l.ggml_is_contiguous() {
            return true;
        }
        return false;
    }

    fn compute_mul_mat_use_gemm<T: TensorType>(
        &self,
        lhs: &[T],
        lhs_l: &Dim,
        rhs: &[T],
        rhs_l: &Dim,
        dst: &mut [T],
        (batching, m, n, k): (usize, usize, usize, usize),
    ) -> GResult<()> {
        // let (b, m, n, k) = (batching, m, n, k);
        // let lhs_stride = lhs_l.nd_stride();
        // let rhs_stride = rhs_l.nd_stride();
        // let rank = lhs_l.n_dims();
        // let lhs_cs = lhs_stride[rank - 1];
        // let lhs_rs = lhs_stride[rank - 2];

        // let rhs_cs = rhs_stride[rank - 1];
        // let rhs_rs = rhs_stride[rank - 2];

        // let a_skip: usize = match lhs_stride[..rank - 2] {
        //     [s1, stride] if s1 == stride * lhs_l.shape()[1] => stride,
        //     [stride] => stride,
        //     [] => m * k,
        //     _ => {
        //         return Err(GError::MatMulUnexpectedStriding {
        //             lhs_l: lhs_l.clone(),
        //             rhs_l: rhs_l.clone(),
        //             bmnk: (b, m, n, k),
        //             msg: "non-contiguous lhs",
        //         });
        //     }
        // };
        // let b_skip: usize = match rhs_stride[..rank - 2] {
        //     [s1, stride] if s1 == stride * rhs_l.shape()[1] => stride,
        //     [stride] => stride,
        //     [] => n * k,
        //     _ => {
        //         return Err(GError::MatMulUnexpectedStriding {
        //             lhs_l: lhs_l.clone(),
        //             rhs_l: rhs_l.clone(),
        //             bmnk: (b, m, n, k),
        //             msg: "non-contiguous lhs",
        //         });
        //     }
        // };
        // let c_skip: usize = m * n;
        // let dst_shape: Shape = Shape::from_array([m, n]);
        // let dst_strides = dst_shape.stride_contiguous();
        // let dst_rs = dst_strides[0];
        // let dst_cs = dst_strides[1];
        // //let mut dst = vec![T::zero(); b * m * n];
        // let num_threads = num_cpus::get();
        // let parallelism = if num_threads > 1 {
        //     Parallelism::Rayon(num_threads)
        // } else {
        //     Parallelism::None
        // };
        // for step in 0..b {
        //     let lhs_p = &lhs[step * a_skip..];
        //     let rhs_p = &rhs[step * b_skip..];
        //     let dst_p = &mut dst[step * c_skip..];
        //     unsafe {
        //         gemm(
        //             /* m: usize = */ m,
        //             /* n: usize = */ n,
        //             /* k: usize = */ k,
        //             /* dst: *mut T = */ dst_p.as_mut_ptr(),
        //             /* dst_cs: isize = */ dst_cs as isize,
        //             /* dst_rs: isize = */ dst_rs as isize,
        //             /* read_dst: bool = */ false,
        //             /* lhs: *const T = */ lhs_p.as_ptr(),
        //             /* lhs_cs: isize = */ lhs_cs as isize,
        //             /* lhs_rs: isize = */ lhs_rs as isize,
        //             /* rhs: *const T = */ rhs_p.as_ptr(),
        //             /* rhs_cs: isize = */ rhs_cs as isize,
        //             /* rhs_rs: isize = */ rhs_rs as isize,
        //             /* alpha: T = */ T::zero(),
        //             /* beta: T = */ T::one(),
        //             /* conj_dst: bool = */ false,
        //             /* conj_lhs: bool = */ false,
        //             /* conj_rhs: bool = */ false,
        //             parallelism,
        //         )
        //     }
        // }
        Ok(())
    }
}

impl Map2 for MatMul {
    const OP: &'static str = "matmul";

    fn f<T: TensorType>(
        &self,
        lhs: &[T],
        lhs_l: &Dim,
        rhs: &[T],
        rhs_l: &Dim,
        dst: &mut [T],
        dst_d: &Dim,
    ) -> GResult<()> {
        let l_dim = lhs_l.shape();
        let r_dim: &[usize] = rhs_l.shape();
        let dim = l_dim.len();
        if dim < 2 || r_dim.len() != dim {
            return Err(GError::ShapeMismatchBinaryOp {
                lhs: lhs_l.shape.clone(),
                rhs: rhs_l.shape.clone(),
                op: "matmul",
            });
        }
        let m = l_dim[dim - 2];
        let k = l_dim[dim - 1];
        let k2 = r_dim[dim - 2];
        let n = r_dim[dim - 1];
        // let mut c_dim = l_dim[..dim - 2].to_vec();
        // c_dim.extend(&[m, n]);
        // let c_n_dims = c_dim.len();
        // let c_shape = Shape::from_vec(c_dim);
        let batching: usize = l_dim[..dim - 2].iter().product();
        let batching_b: usize = r_dim[..dim - 2].iter().product();
        if k != k2 || batching != batching_b {
            return Err(GError::ShapeMismatchBinaryOp {
                lhs: lhs_l.shape.clone(),
                rhs: rhs_l.shape.clone(),
                op: "matmul",
            });
        }
        self.compute_mul_mat_use_gemm(lhs, lhs_l, &rhs, rhs_l, dst, (batching, m, n, k))?;
        Ok(())
    }

    fn f_f16_f32(
        &self,
        lhs: &[f16],
        lhs_l: &Dim,
        rhs: &[f32],
        rhs_l: &Dim,
        dst: &mut [f32],
        dst_d1: &Dim,
    ) -> GResult<()> {
        let time1 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let rhs_f16: Vec<f16> = rhs.par_iter().map(|e| f16::from_f32(*e)).collect();

        let (ne00, ne01, ne02, ne03) = lhs_l.dim4();
        let (ne10, ne11, ne12, ne13) = rhs_l.dim4();

        let (ne0, ne1, ne2, ne3) = dst_d1.dim4();

        let (nb00, nb01, nb02, nb03) = lhs_l.stride_4d();
        let (nb10, nb11, nb12, nb13) = rhs_l.stride_4d();
        let (nb0, nb1, nb2, nb3) = dst_d1.stride_4d();
        //const int64_t ne   = ne0*ne1*ne2*ne3;

        let ith = 0;
        let nth = 1;

        let nr = ne01 * ne02 * ne03;

        // rows per thread
        let dr = (nr + nth - 1) / nth;

        // row range for this thread
        let ir0 = dr * ith;
        let ir1 = std::cmp::min(ir0 + dr, nr);

        //   ggml_fp16_t * wdata = params->wdata;

        for ir in ir0..ir1 {
            // src0 indices
            let i03 = ir / (ne02 * ne01);
            let i02 = (ir - i03 * ne02 * ne01) / ne01;
            let i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            let i13 = i03;
            let i12 = i02;

            let i0 = i01;
            let i2 = i02;
            let i3 = i03;

            let src0_row = &lhs[i01 * nb01 + i02 * nb02 + i03 * nb03..];
            let src1_col = &rhs_f16[(0 + i12 * ne11 + i13 * ne12 * ne11) * ne00..];

            let dst_col = &mut dst[i0 * nb0 + 0 * nb1 + i2 * nb2 + i3 * nb3..];

            for ic in 0..ne11 {
                //  assert!(ne00 % 32 == 0);
                unsafe {
                    vec_dot_f16(
                        src0_row.as_ptr(),
                        src1_col[ic * ne00..].as_ptr(),
                        dst_col[ic * ne0..].as_mut_ptr(),
                        ne00,
                    )
                }
            }
        }
        let time2 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        println!("{} time:{} ms", Self::OP, time2 - time1);
        return Ok(());

        // if self.can_use_gemm(lhs_l, rhs_l) {
        //     // let rhs_l = rhs_l.clone().transpose(0, 1)?;
        //     let l_dim = lhs_l.shape();
        //     let r_dim: &[usize] = rhs_l.shape();
        //     let dim = l_dim.len();
        //     if dim < 2 || r_dim.len() != dim {
        //         return Err(GError::ShapeMismatchBinaryOp {
        //             lhs: lhs_l.shape.clone(),
        //             rhs: rhs_l.shape.clone(),
        //             op: "matmul",
        //         });
        //     }
        //     let m = l_dim[dim - 2];
        //     let k = l_dim[dim - 1];
        //     let k2 = r_dim[dim - 2];
        //     let n = r_dim[dim - 1];
        //     // let mut c_dim = l_dim[..dim - 2].to_vec();
        //     // c_dim.extend(&[m, n]);
        //     // let c_n_dims = c_dim.len();
        //     // let c_shape = Shape::from_vec(c_dim);
        //     let batching: usize = l_dim[..dim - 2].iter().product();
        //     let batching_b: usize = r_dim[..dim - 2].iter().product();
        //     if k != k2 || batching != batching_b {
        //         return Err(GError::ShapeMismatchBinaryOp {
        //             lhs: lhs_l.shape.clone(),
        //             rhs: rhs_l.shape.clone(),
        //             op: "matmul",
        //         });
        //     }
        //     let mut dst_f16 = vec![f16::zero(); batching * m * n];
        //     self.compute_mul_mat_use_gemm(
        //         lhs,
        //         lhs_l,
        //         &rhs_f16,
        //         &rhs_l,
        //         &mut dst_f16,
        //         (batching, m, n, k),
        //     )?;
        //     dst.par_iter_mut()
        //         .zip(dst_f16.par_iter())
        //         .for_each(|(d, s)| *d = s.to_f32());
        // } else {
        //     println!("not use gemm");
        // }

        // let time2 = SystemTime::now()
        //     .duration_since(UNIX_EPOCH)
        //     .unwrap()
        //     .as_millis();
        // println!("{} time:{} ms", Self::OP, time2 - time1);
        // Ok(())
    }
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

    fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()> {
        self.f(inp, inp_d, dst, dst_d)
    }

    fn f_f32_f16(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f16], dst_d: &Dim) -> GResult<()> {
        assert!(dst_d.ggml_is_contiguous());
        assert_eq!(inp_d.elem_count(), dst_d.elem_count());

        let mut id: usize = 0;
        // ggml_fp16_t *dst_ptr = (ggml_fp16_t *)dst->data;

        let (ne00, ne01, ne02, ne03) = inp_d.dim4();
        let (nb00, nb01, nb02, nb03) = inp_d.stride_4d();

        // dst.par_iter_mut()
        //     .zip(inp.par_iter())
        //     .for_each(|(d, a)| *d = f16::from_f32(*a));

        for i03 in 0..ne03 {
            for i02 in 0..ne02 {
                for i01 in 0..ne01 {
                    for i00 in 0..ne00 {
                        let src0_ptr = inp[i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03];
                        dst[id] = f16::from_f32(src0_ptr);
                        id = id + 1;
                    }
                }
            }
        }
        Ok(())
    }
}

struct FlashAttn;

impl Map4 for FlashAttn {
    const OP: &'static str = "flashattn";
    fn f_f16_f16_f16_f32(
        &self,
        q: &[f16],
        q_d: &Dim,
        k: &[f16],
        k_d: &Dim,
        v: &[f16],
        v_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        let (neq0, neq1, neq2, neq3) = q_d.dim4();
        let (nek0, nek1) = k_d.dim2();
        // const int neq0 = q->ne[0];
        // const int neq1 = q->ne[1];
        // const int neq2 = q->ne[2];
        // const int neq3 = q->ne[3];

        // const int nek2 = k->ne[2];
        // const int nek3 = k->ne[3];

        // const int nev0 = v->ne[0];
        //const int nev1 = v->ne[1];

        let (_, nev1) = v_d.dim2();
        // const int nev2 = v->ne[2];
        // const int nev3 = v->ne[3];

        let (ne0, ne1) = dst_d.dim2();

        // const int ne0 = dst->ne[0];
        // const int ne1 = dst->ne[1];
        // const int ne2  = dst->ne[2];
        // const int ne3  = dst->ne[3];

        let (nbk0, nbk1, nbk2, nbk3) = k_d.stride_4d();
        let (nbq0, nbq1, nbq2, nbq3) = q_d.stride_4d();
        let (nbv0, nbv1, nbv2, nbv3) = v_d.stride_4d();
        let (nb0, nb1, nb2, nb3) = dst_d.stride_4d();

        // const int nbk0 = k->nb[0];
        // const int nbk1 = k->nb[1];
        // const int nbk2 = k->nb[2];
        // const int nbk3 = k->nb[3];

        // const int nbq0 = q->nb[0];
        // const int nbq1 = q->nb[1];
        // const int nbq2 = q->nb[2];
        // const int nbq3 = q->nb[3];

        // const int nbv0 = v->nb[0];
        // const int nbv1 = v->nb[1];
        // const int nbv2 = v->nb[2];
        // const int nbv3 = v->nb[3];

        // const int nb0 = dst->nb[0];
        // const int nb1 = dst->nb[1];
        // const int nb2 = dst->nb[2];
        // const int nb3 = dst->nb[3];

        let ith = 0;
        let nth = 1;

        let D = neq0;
        let N = neq1;
        let P = nek1 - N;
        let M = P + N;

        assert!(ne0 == D);
        assert!(ne1 == N);
        assert!(P >= 0);

        assert!(nbq0 == 1);
        assert!(nbk0 == 1);
        assert!(nbv0 == 1);

        assert!(neq0 == D);
        assert!(nek0 == D);
        assert!(nev1 == D);

        assert!(neq1 == N);
        assert!(nek1 == N + P);
        assert!(nev1 == D);

        // dst cannot be transposed or permuted
        assert!(nb0 == 1);
        assert!(nb0 <= nb1);
        assert!(nb1 <= nb2);
        assert!(nb2 <= nb3);

        let nr = neq1 * neq2 * neq3;

        // rows per thread
        let dr = (nr + nth - 1) / nth;

        // row range for this thread
        let ir0 = dr * ith;
        let ir1 = std::cmp::min(ir0 + dr, nr);

        let scale = (1.0 / (D as f64).sqrt()) as f32;

        //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

        let wdata = vec![
            0u8;
            M * std::mem::size_of::<f32>() //ith * (2 * M + CACHE_LINE_SIZE_F32)
                + M * std::mem::size_of::<f16>()
        ];

        for ir in ir0..ir1 {
            // q indices
            let iq3 = ir / (neq2 * neq1);
            let iq2 = (ir - iq3 * neq2 * neq1) / neq1;
            let iq1 = ir - iq3 * neq2 * neq1 - iq2 * neq1;

            let Sf32 = unsafe {
                std::slice::from_raw_parts_mut(
                    wdata.as_ptr() as *mut f32,
                    wdata.len() / std::mem::size_of::<f32>(),
                )
            };

            let (S, S2) = Sf32.split_at_mut(M); //ith * (2 * M + CACHE_LINE_SIZE_F32) +

            // let S = &mut Sf32[ith * (2 * M + CACHE_LINE_SIZE_F32)..];

            for ic in 0..nek1 {
                // k indices
                let ik3 = iq3;
                let ik2 = iq2;
                let ik1 = ic;

                // S indices
                let i1 = ik1;

                unsafe {
                    vec_dot_f16(
                        k[ik1 * nbk1 + ik2 * nbk2 + ik3 * nbk3..].as_ptr(),
                        q[iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3..].as_ptr(),
                        S[i1..].as_mut_ptr(),
                        neq0,
                    )
                };
            }

            // scale
            vec_scale_f32(nek1, S, scale);

            if false {
                for i in P..M {
                    if (i > P + iq1) {
                        S[i] = -std::f32::INFINITY;
                    }
                }
            }

            // softmax
            {
                let mut max = -std::f32::INFINITY;
                for i in 0..M {
                    max = max.max(S[i]) //f32::max(max, );
                }

                let mut sum: f32 = 0.0;

                //  let ss: u16 = 0;
                for i in 0..M {
                    if S[i] == -std::f32::INFINITY {
                        S[i] = 0.0;
                    } else {
                        //const float val = (S[i] == -INFINITY) ? 0.0 : exp(S[i] - max);
                        let s = f16::from_f32(S[i] - max);
                        // let ss: u16 = unsafe { std::mem::transmute(s) };
                        let val = GLOBAL_CPU_DEVICE_CACHE
                            .get_exp_cache(s.to_bits() as usize)
                            .to_f32();
                        sum += val;
                        S[i] = val;
                    }
                }

                assert!(sum > 0.0f32);

                sum = 1.0 / sum;
                vec_scale_f32(M, S, sum);
            }

            let S16 = unsafe { std::slice::from_raw_parts_mut(S2.as_ptr() as *mut f16, M) };

            for i in 0..M {
                S16[i] = f16::from_f32(S[i]);
            }

            for ic in 0..nev1 {
                // dst indices
                let i1 = iq1;
                let i2 = iq2;
                let i3 = iq3;

                unsafe {
                    vec_dot_f16(
                        v[ic * nbv1 + i2 * nbv2 + i3 * nbv3..].as_ptr(),
                        S16.as_ptr(),
                        dst[ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3..].as_mut_ptr(),
                        nek1,
                    );
                }
            }
        }
        println!("std::f32::INFINITY:{}", std::f32::INFINITY);

        Ok(())
    }
}

struct Scale;

impl Map2 for Scale {
    const OP: &'static str = "scale";
    fn f<T: TensorType>(
        &self,
        inp0: &[T],
        inp0_d: &Dim,
        inp1: &[T],
        inp1_d: &Dim,
        dst: &mut [T],
        dst_d: &Dim,
    ) -> GResult<()> {
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
        Ok(())
    }

    fn f_f32(
        &self,
        inp0: &[f32],
        inp0_d: &Dim,
        inp1: &[f32],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        assert!(inp0_d.ggml_is_contiguous());
        assert!(dst_d.ggml_is_contiguous());
        assert!(is_same_shape(inp0_d.shape(), dst_d.shape()));
        assert!(inp1_d.is_scalar());

        // scale factor
        let v = inp1[0];

        let ith = 0;
        let nth = 1;

        let nc = inp0_d.dim1();
        let nr = inp0_d.nrows();

        let (_, nb1) = dst_d.stride_2d();

        // rows per thread
        let dr = (nr + nth - 1) / nth;

        // row range for this thread
        let ir0 = dr * ith;
        let ir1 = std::cmp::min(ir0 + dr, nr);

        for i1 in ir0..ir1 {
            vec_scale_f32(nc, &mut dst[i1 * nb1..], v);
        }
        Ok(())
    }
}

struct GetRows;

impl Map2 for GetRows {
    const OP: &'static str = "get_rows";
    fn f_Q40_I32_f32(
        &self,
        inp0: &[BlockV1_Q4_0],
        inp0_d: &Dim,
        inp1: &[i32],
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
            let r = inp1[i] as usize;
            BlockV1_Q4_0::to_f32(
                &inp0[r * nbv1..r * nbv1 + nc],
                &mut dst[i * nbd1..i * nbd1 + nc],
            )
        }
        Ok(())
    }
}

pub fn galois_conv_1d_1s(kernel: &Tensor, src: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (kernel.device(), src.device(), dst_device) {
        (Device::Cpu(k), Device::Cpu(s), Device::Cpu(d)) => {
            Conv1D1S.map(k, kernel.dim(), s, src.dim(), d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_conv_1d_2s(kernel: &Tensor, src: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (kernel.device(), src.device(), dst_device) {
        (Device::Cpu(k), Device::Cpu(s), Device::Cpu(d)) => {
            Conv1D2S.map(k, kernel.dim(), s, src.dim(), d, dst_dim)?;
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

pub fn galois_matmul(a: &Tensor, b: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (a.device(), b.device(), dst_device) {
        (Device::Cpu(a1), Device::Cpu(b1), Device::Cpu(d)) => {
            MatMul.map(a1, a.dim(), b1, b.dim(), d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_mul(a: &Tensor, b: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (a.device(), b.device(), dst_device) {
        (Device::Cpu(a1), Device::Cpu(b1), Device::Cpu(d)) => {
            Mul.map(a1, a.dim(), b1, b.dim(), d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_gelu(src: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (src.device(), dst_device) {
        (Device::Cpu(s), Device::Cpu(d)) => {
            Gelu.map(s, src.dim(), d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_norm(src: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (src.device(), dst_device) {
        (Device::Cpu(s), Device::Cpu(d)) => {
            Norm.map(s, src.dim(), d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_rms_norm(src: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (src.device(), dst_device) {
        (Device::Cpu(s), Device::Cpu(d)) => {
            RmsNorm.map(s, src.dim(), d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_cpy(src: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (src.device(), dst_device) {
        (Device::Cpu(s), Device::Cpu(d)) => {
            Cpy.map(s, src.dim(), d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_flash_attn(Q: &Tensor, K: &Tensor, V: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (Q.device(), K.device(), V.device(), dst_device) {
        (Device::Cpu(q), Device::Cpu(k), Device::Cpu(v), Device::Cpu(d)) => {
            FlashAttn.map(q, Q.dim(), k, K.dim(), v, V.dim(), d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_scale(src0: &Tensor, src1: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (src0.device(), src1.device(), dst_device) {
        (Device::Cpu(s0), Device::Cpu(s1), Device::Cpu(d)) => {
            Scale.map(s0, src0.dim(), s1, src1.dim(), d, dst_dim)?;
        }
        _ => {
            todo!()
        }
    }
    Ok(())
}

pub fn galois_get_rows(src0: &Tensor, src1: &Tensor, dst: &mut Tensor) -> GResult<()> {
    let (dst_device, dst_dim) = dst.device_dim();
    match (src0.device(), src1.device(), dst_device) {
        (Device::Cpu(s0), Device::Cpu(s1), Device::Cpu(d)) => {
            GetRows.map(s0, src0.dim(), s1, src1.dim(), d, dst_dim)?;
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

    fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()>;

    fn f_f32_f16(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f16], dst_d: &Dim) -> GResult<()>;

    fn map(&self, dev1: &CpuDevice, d1: &Dim, dst: &mut CpuDevice, d3: &Dim) -> GResult<()> {
        match (dev1, dst) {
            (CpuDevice::F16(v1), CpuDevice::F16(d)) => {
                self.f(v1.as_slice(), d1, d.as_slice_mut(), d3)
            }
            (CpuDevice::F32(v1), CpuDevice::F32(d)) => {
                self.f_f32(v1.as_slice(), d1, d.as_slice_mut(), d3)
            }
            (CpuDevice::F32(v1), CpuDevice::F16(d)) => {
                self.f_f32_f16(v1.as_slice(), d1, d.as_slice_mut(), d3)
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
        inp_d: &Dim,
        k: &[f32],
        k_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        self.f(inp, inp_d, k, k_d, dst, dst_d)
    }

    fn f_Q40_I32_f32(
        &self,
        inp0: &[BlockV1_Q4_0],
        inp0_d: &Dim,
        inp1: &[i32],
        inp1_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        todo!()
    }

    fn f_f16_f32(
        &self,
        k: &[f16],
        k_d: &Dim,
        inp: &[f32],
        inp_d: &Dim,
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
            (CpuDevice::F16(v1), CpuDevice::F32(v2), CpuDevice::F32(d)) => {
                self.f_f16_f32(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut(), d3)
            }
            (CpuDevice::Q4V1_0(v1), CpuDevice::I32(v2), CpuDevice::F32(d)) => {
                self.f_Q40_I32_f32(v1.as_slice(), d1, v2.as_slice(), d2, d.as_slice_mut(), d3)
            }
            _ => {
                todo!()
            }
        }
    }
}

trait Map4 {
    const OP: &'static str;

    fn f_f16_f16_f16_f32(
        &self,
        inp1: &[f16],
        inp1_d: &Dim,
        inp2: &[f16],
        inp2_d: &Dim,
        inp3: &[f16],
        inp3_d: &Dim,
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
        dev3: &CpuDevice,
        d3: &Dim,
        dst: &mut CpuDevice,
        d4: &Dim,
    ) -> GResult<()> {
        match (dev1, dev2, dev3, dst) {
            (CpuDevice::F16(v1), CpuDevice::F16(v2), CpuDevice::F16(v3), CpuDevice::F32(d)) => self
                .f_f16_f16_f16_f32(
                    v1.as_slice(),
                    d1,
                    v2.as_slice(),
                    d2,
                    v3.as_slice(),
                    d3,
                    d.as_slice_mut(),
                    d4,
                ),
            _ => {
                todo!()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use half::vec;

    use super::*;
    use crate::mat;

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
        let a = mat(&[[1.0f32, 3.0, 5.0], [7.0f32, 9.0, 11.0]]);
        let b = mat(&[[2.0f32, 4.0], [6.0f32, 8.0], [10.0f32, 12.0]]);
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
        let mut d = Tensor::from_vec(v, 2, Shape::from_array([2, 2]));
        // MatMul.map(
        //     &m1.device(),
        //     &m1.dim,
        //     m2.device(),
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
        );

        // let mut m2 = Tensor::from_vec(
        //     vec![2.0f32, 6.0, 10.0, 4.0, 8.0, 12.0],
        //     2,
        //     Shape::from_array([3, 2]),
        // );

        let mut m2 = Tensor::from_vec(
            vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0],
            2,
            Shape::from_array([2, 3]),
        );

        println!("m1:{:?}", m1);
        println!("m2:{:?}", m2);

        //  let m2 = mat(&[[1.0f32, 2.0, 4.0], [3.0f32, 5.0, 6.0]]);

        let v = vec![0.0f32; 15];
        let mut d = Tensor::from_vec(v, 2, Shape::from_array([2, 2]));
        // MatMul.map(
        //     &m1.device(),
        //     &m1.dim,
        //     m2.device(),
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

// macro_rules! impl_float_unary_op {
//     ($a:ident) => {
//         impl UnaryOp for $a {
//             fn _exp(&self) -> Self {
//                 self.exp()
//             }
//             fn _ln(&self) -> Self {
//                 self.ln()
//             }
//             fn _sin(&self) -> Self {
//                 self.sin()
//             }
//             fn _cos(&self) -> Self {
//                 self.cos()
//             }
//             fn _tanh(&self) -> Self {
//                 self.tanh()
//             }
//             fn _neg(&self) -> Self {
//                 self.neg()
//             }

//             fn _recip(&self) -> Self {
//                 self.recip()
//             }
//             fn _sqr(&self) -> Self {
//                 self * self
//             }
//             fn _sqrt(&self) -> Self {
//                 self.sqrt()
//             }
//         }
//     };
// }

// impl_float_unary_op!(f16);
// impl_float_unary_op!(f32);
// impl_float_unary_op!(f64);

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
