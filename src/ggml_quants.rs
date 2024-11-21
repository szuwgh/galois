use crate::{error::GResult, GGmlType};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use half::f16;
use std::ops::Sub;
pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;
pub const QK_K: usize = 256;

#[inline(always)]
pub(crate) unsafe fn sum_i16_pairs_float(x: __m256i) -> __m256 {
    let ones = _mm256_set1_epi16(1);
    let summed_pairs = _mm256_madd_epi16(ones, x);
    _mm256_cvtepi32_ps(summed_pairs)
}

#[inline(always)]
pub(crate) unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
    let dot = _mm256_maddubs_epi16(ax, sy);
    sum_i16_pairs_float(dot)
}

#[inline(always)]
pub(crate) unsafe fn hsum_float_8(x: __m256) -> f32 {
    let res = _mm256_extractf128_ps(x, 1);
    let res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    let res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

#[inline(always)]
pub(crate) unsafe fn bytes_from_nibbles_32(rsi: *const u8) -> __m256i {
    let tmp = _mm_loadu_si128(rsi as *const __m128i);
    let bytes = _mm256_insertf128_si256::<1>(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4));
    let low_mask = _mm256_set1_epi8(0xF);
    _mm256_and_si256(low_mask, bytes)
}

#[inline(always)]
pub(crate) unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    mul_sum_us8_pairs_float(ax, sy)
}

pub trait QuantType: Sized + Clone {
    const BLCK_SIZE: usize;
    type VecDotType: QuantType;
    fn to_f32(src: &[Self], dst: &mut [f32]);

    fn from_f32(src: &[f32], dst: &mut [Self]) -> GResult<()>;

    fn vec_dot(me: &[Self], other: &[Self::VecDotType]) -> f32;

    fn zero() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
}

impl Sub for BlockQ4_0 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_0 / 2],
}

impl BlockQ4_0 {
    pub fn d(&self) -> f16 {
        self.d
    }

    pub fn qs(&self) -> &[u8] {
        &self.qs
    }

    pub fn vec_dot_q8_0(me: &[Self], other: &[BlockQ8_0]) -> f32 {
        assert!(me.len() == other.len());
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for (x, y) in me.iter().zip(other.iter()) {
                let d = _mm256_set1_ps(f16::to_f32(x.d) * f16::to_f32(y.d));
                let bx = bytes_from_nibbles_32(x.qs.as_ptr());
                let off = _mm256_set1_epi8(8);
                let bx = _mm256_sub_epi8(bx, off);
                let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
                let q = mul_sum_i8_pairs_float(bx, by);
                acc = _mm256_fmadd_ps(d, q, acc);
            }
            hsum_float_8(acc)
        }

        // let qk = QK8_0;
        // let nb = n / qk;

        // assert!(n % qk == 0);
        // let mut sumf = 0.0;

        // for i in 0..nb {
        //     let mut sumi: i32 = 0;

        //     for j in 0..(qk / 2) {
        //         let v0 = (me[i].qs[j] & 0x0F) as i32 - 8;
        //         let v1 = (me[i].qs[j] >> 4) as i32 - 8;

        //         sumi += (v0 * other[i].qs[j] as i32) + (v1 * other[i].qs[j + qk / 2] as i32);
        //     }

        //     sumf += sumi as f32 * me[i].d.to_f32() * other[i].d.to_f32();
        // }

        // sumf
    }
}

impl QuantType for BlockQ4_0 {
    const BLCK_SIZE: usize = QK4_0;
    type VecDotType = BlockQ8_0;

    fn from_f32(src: &[f32], dst: &mut [Self]) -> GResult<()> {
        todo!()
    }

    fn vec_dot(me: &[Self], other: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_q8_0(me, other)
    }

    fn to_f32(src: &[Self], dst: &mut [f32]) {
        let k = dst.len();
        let qk = Self::BLCK_SIZE;
        assert!(k % qk == 0);
        let nb = k / qk;
        for i in 0..nb {
            let d = src[i].d.to_f32();
            for j in 0..qk / 2 {
                let vi = src[i].qs[j];
                let vi0 = (vi & 0x0F) as i8;
                let vi1 = (vi >> 4) as i8;

                let v0 = (vi0 - 8) as f32 * d;
                let v1 = (vi1 - 8) as f32 * d;
                // println!(
                //     "d = {:.6}, vi = {}, vi0 = {}, vi1 = {}, v0 = {:.6}, v1 = {:.6}",
                //     d, vi, vi0, vi1, v0, v1,
                // );

                dst[i * qk + j + 0] = v0;
                dst[i * qk + j + qk / 2] = v1;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
#[repr(C)]
pub struct BlockQ6K {
    pub(crate) ql: [u8; QK_K / 2],
    pub(crate) qh: [u8; QK_K / 4],
    pub(crate) scales: [i8; QK_K / 16],
    pub(crate) d: f16,
}

#[derive(Debug, Clone, PartialEq, Copy)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub(crate) d: f16,
    pub(crate) qs: [i8; QK8_0],
}

impl QuantType for BlockQ8_0 {
    const BLCK_SIZE: usize = QK8_0;
    type VecDotType = BlockQ8_0;

    fn vec_dot(me: &[Self], other: &[Self::VecDotType]) -> f32 {
        todo!()
    }

    fn from_f32(src: &[f32], dst: &mut [Self]) -> GResult<()> {
        let k = src.len();
        if k % Self::BLCK_SIZE != 0 {
            panic!("{k} is not divisible by {}", Self::BLCK_SIZE);
        };
        let nb = k / Self::BLCK_SIZE;
        if dst.len() != nb {
            panic!(
                "size mismatch {} {} {}",
                src.len(),
                dst.len(),
                Self::BLCK_SIZE
            )
        }
        for (i, ys) in dst.iter_mut().enumerate() {
            let mut amax = 0f32;
            let xs = &src[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            for &x in xs.iter() {
                amax = amax.max(x.abs())
            }
            let d = amax / ((1 << 7) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            for (y, &x) in ys.qs.iter_mut().zip(xs.iter()) {
                *y = f32::round(x * id) as i8
            }
        }
        Ok(())
    }

    fn to_f32(src: &[Self], dst: &mut [f32]) {
        todo!()
    }
}
