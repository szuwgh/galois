use core::time;
use std::ops::Neg;

use crate::error::GResult;
use crate::Device;
use crate::GError;
//use super::broadcast::{broadcasting_binary_op, general_broadcasting};
use crate::CpuDevice;
use crate::Dim;
use crate::Shape;
use crate::Tensor;
use crate::TensorType;
use core::cell::OnceCell;
use core::simd::f32x32;
use gemm::f16;
use gemm::gemm;
use gemm::Parallelism;
use lazy_static::lazy_static;
use num_traits::Float;
use num_traits::One;
use num_traits::Zero;
use rayon::prelude::*;
use std::simd::num::SimdFloat;

const COEF_A: f32 = 0.044715;
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

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
    for i in 0..n {
        y[i] *= v;
    }
}

unsafe impl Sync for CpuDeviceCache {}

struct CpuDeviceCache {
    gelu_cache: OnceCell<Vec<f16>>,
}

impl CpuDeviceCache {
    fn new() -> CpuDeviceCache {
        CpuDeviceCache {
            gelu_cache: OnceCell::from(Self::init_gelu_cache()),
        }
    }

    fn get_gelu_cache(&self) -> &Vec<f16> {
        self.gelu_cache.get().unwrap()
    }

    fn init_gelu_cache() -> Vec<f16> {
        (0..1 << 16)
            .map(|x| {
                let v = f16::from_bits(x as u16).to_f32();
                f16::from_f32(compute_gelu(v))
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
            .as_micros();
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
            .as_micros();
        println!("{} time:{} us", Self::OP, time2 - time1);
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
            *d = GLOBAL_CPU_DEVICE_CACHE.get_gelu_cache()[f16::from_f32(*a).to_bits() as usize]
                .to_f32()
        });
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

        // for i1 in ir0..ir1 {
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
        // }
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
                        unsafe { f16::vec_dot_f16(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0) }
                        dst_data[i0] += v;
                    }
                }
            });

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
                        unsafe { f16::vec_dot_f16(wdata1.as_ptr(), wdata2.as_ptr(), &mut v, ew0) }
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
}

struct MatMul;

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
                lhs: lhs_l.s.clone(),
                rhs: rhs_l.s.clone(),
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
                lhs: lhs_l.s.clone(),
                rhs: rhs_l.s.clone(),
                op: "matmul",
            });
        }

        let (b, m, n, k) = (batching, m, n, k);
        let lhs_stride = lhs_l.nd_stride();
        let rhs_stride = rhs_l.nd_stride();
        let rank = lhs_l.n_dims();
        let lhs_cs = lhs_stride[rank - 1];
        let lhs_rs = lhs_stride[rank - 2];

        let rhs_cs = rhs_stride[rank - 1];
        let rhs_rs = rhs_stride[rank - 2];

        let a_skip: usize = match lhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * lhs_l.shape()[1] => stride,
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
            [s1, stride] if s1 == stride * rhs_l.shape()[1] => stride,
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
        //  let mut dst_f16 = vec![f16::zero(); b * m * n];
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
        let l_dim = lhs_l.shape();
        let r_dim: &[usize] = rhs_l.shape();
        let dim = l_dim.len();
        if dim < 2 || r_dim.len() != dim {
            return Err(GError::ShapeMismatchBinaryOp {
                lhs: lhs_l.s.clone(),
                rhs: rhs_l.s.clone(),
                op: "matmul",
            });
        }
        let m = l_dim[dim - 2];
        let k = l_dim[dim - 1];
        let k2 = r_dim[dim - 2];
        let n = r_dim[dim - 1];
        //  let mut c_dim = l_dim[..dim - 2].to_vec();
        // c_dim.extend(&[m, n]);
        //  let c_n_dims = c_dim.len();
        //  let c_shape = Shape::from_vec(c_dim);
        let batching: usize = l_dim[..dim - 2].iter().product();
        let batching_b: usize = r_dim[..dim - 2].iter().product();
        if k != k2 || batching != batching_b {
            return Err(GError::ShapeMismatchBinaryOp {
                lhs: lhs_l.s.clone(),
                rhs: rhs_l.s.clone(),
                op: "matmul",
            });
        }

        let (b, m, n, k) = (batching, m, n, k);
        let lhs_stride = lhs_l.nd_stride();
        let rhs_stride = rhs_l.nd_stride();
        let rank = lhs_l.n_dims();
        let lhs_cs = lhs_stride[rank - 1];
        let lhs_rs = lhs_stride[rank - 2];

        let rhs_cs = rhs_stride[rank - 1];
        let rhs_rs = rhs_stride[rank - 2];

        let a_skip: usize = match lhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * lhs_l.shape()[1] => stride,
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
            [s1, stride] if s1 == stride * rhs_l.shape()[1] => stride,
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
        let dst_cs: usize = dst_strides[1];
        let mut dst_f16 = vec![f16::zero(); b * m * n];
        let num_threads = num_cpus::get();
        let parallelism = if num_threads > 1 {
            Parallelism::Rayon(num_threads)
        } else {
            Parallelism::None
        };
        let rhs_f16: Vec<f16> = rhs.par_iter().map(|e| f16::from_f32(*e)).collect();
        for step in 0..b {
            let lhs_p = &lhs[step * a_skip..];
            let rhs_p = &rhs_f16[step * b_skip..];
            let dst_p = &mut dst_f16[step * c_skip..];
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
                    /* alpha: T = */ f16::zero(),
                    /* beta: T = */ f16::one(),
                    /* conj_dst: bool = */ false,
                    /* conj_lhs: bool = */ false,
                    /* conj_rhs: bool = */ false,
                    parallelism,
                )
            }
        }

        dst.par_iter_mut()
            .zip(dst_f16.par_iter())
            .for_each(|(d, s)| *d = s.to_f32());
        let time2 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        println!("{} time:{} ms", Self::OP, time2 - time1);
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
        inp_d: &Dim,
        k: &[f32],
        k_d: &Dim,
        dst: &mut [f32],
        dst_d: &Dim,
    ) -> GResult<()> {
        self.f(inp, inp_d, k, k_d, dst, dst_d)
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
            _ => {
                todo!()
            }
        }
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

trait Map {
    const OP: &'static str;
    fn f<T: TensorType>(&self, inp: &[T], inp_d: &Dim, dst: &mut [T], dst_d: &Dim) -> GResult<()>;

    fn f_f32(&self, inp: &[f32], inp_d: &Dim, dst: &mut [f32], dst_d: &Dim) -> GResult<()>;

    fn map(&self, dev1: &CpuDevice, d1: &Dim, dst: &mut CpuDevice, d3: &Dim) -> GResult<()> {
        match (dev1, dst) {
            (CpuDevice::F16(v1), CpuDevice::F16(d)) => {
                self.f(v1.as_slice(), d1, d.as_slice_mut(), d3)
            }
            (CpuDevice::F32(v1), CpuDevice::F32(d)) => {
                self.f_f32(v1.as_slice(), d1, d.as_slice_mut(), d3)
            }
            _ => {
                todo!()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat;

    //  [[ F32(7.0), F32(10.0), F32(14.0)],
    //  [ F32(15.0), F32(22.0), F32(32.0)],
    //  [ F32(15.0), F32(22.0), F32(32.0)]]
    // [7.0, 10.0, 14.0, 15.0, 22.0, 32.0, 15.0, 22.0, 32.0]
    #[test]
    fn test_matmul_f32() {
        let a = mat(&[[1.0f32, 2.0], [3.0f32, 4.0], [3.0f32, 4.0]]);
        let b = mat(&[[1.0f32, 2.0, 4.0], [3.0f32, 4.0, 5.0]]);

        // let ne = [
        //     a.ggml_shape()[1],
        //     b.ggml_shape()[1],
        //     a.ggml_shape()[2],
        //     b.ggml_shape()[3],
        // ];
        //println!("{:?}", ne);
        let v = vec![0.0f32; 9];
        let mut d = Tensor::from_vec(v, 2, Shape::from_array([3, 3]));
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
    }

    #[test]
    fn test_matmul_f16() {
        let m1 = mat(&[
            [f16::from_f32(0.123), f16::from_f32(0.456)],
            [f16::from_f32(-0.123), f16::from_f32(9.63)],
            [f16::from_f32(7.996), f16::from_f32(7.9687)],
        ]);

        let m2 = mat(&[[1.0f32, 2.0, 4.0], [3.0f32, 4.0, 5.0]]);
        let v = vec![0.0f32; 9];
        let mut d = Tensor::from_vec(v, 2, Shape::from_array([3, 3]));
        // MatMul.map(
        //     &m1.device(),
        //     &m1.dim,
        //     m2.device(),
        //     &m2.dim,
        //     d.device_mut(),
        //     &d.dim,
        // );
        galois_matmul(&m1, &m2, &mut d).unwrap();
        println!("{:?}", d);
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
