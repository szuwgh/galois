use crate::op::vec_dot_f16;
use crate::{error::GResult, GGmlType};
use byteorder::{ByteOrder, LittleEndian};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use cudarc::driver::DeviceRepr;
use cudarc::driver::ValidAsZeroBits;
use half::f16;
use std::ops::Sub;

pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;
pub const QK_K: usize = 256;

#[inline(always)]
unsafe fn get_scale_shuffle_k4(i: usize) -> __m256i {
    const K_SHUFFLE: [u8; 256] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
        11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    ];
    _mm256_loadu_si256((K_SHUFFLE.as_ptr() as *const __m256i).add(i))
}

#[inline(always)]
unsafe fn mm256_set_m128i(a: __m128i, b: __m128i) -> __m256i {
    _mm256_insertf128_si256(_mm256_castsi128_si256(b), a, 1)
}

#[inline(always)]
unsafe fn get_scale_shuffle(i: usize) -> __m128i {
    const K_SHUFFLE: [u8; 128] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
        7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13,
        13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    ];
    _mm_loadu_si128((K_SHUFFLE.as_ptr() as *const __m128i).add(i))
}

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

pub trait QuantType: Sized + Clone + Sync {
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

impl QuantType for f16 {
    const BLCK_SIZE: usize = 1;
    type VecDotType = f16;

    fn from_f32(src: &[f32], dst: &mut [Self]) -> GResult<()> {
        for i in 0..dst.len() {
            dst[i] = f16::from_f32(src[i]);
        }
        Ok(())
    }

    fn to_f32(src: &[Self], dst: &mut [f32]) {
        todo!()
    }

    fn vec_dot(me: &[Self], other: &[Self::VecDotType]) -> f32 {
        let mut res = 0.0f32;
        unsafe {
            vec_dot_f16(me.as_ptr(), other.as_ptr(), &mut res, me.len());
        }
        res
    }
}

unsafe impl DeviceRepr for BlockQ4_0 {}
unsafe impl ValidAsZeroBits for BlockQ4_0 {}

#[derive(Default, Debug, Clone, PartialEq, Copy)]
#[repr(C)]
pub(crate) struct BlockQ4_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_0 / 2],
}

impl BlockQ4_0 {
    pub(crate) fn d(&self) -> f16 {
        self.d
    }

    pub(crate) fn qs(&self) -> &[u8] {
        &self.qs
    }

    #[cfg(target_feature = "avx")]
    pub(crate) fn vec_dot_q8_0(me: &[Self], other: &[BlockQ8_0]) -> f32 {
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
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[BlockQ8_0]) -> f32 {
        let qk = QK8_0;
        if n % QK8_0 != 0 {
            panic!("vec_dot_q4_0_q8_0: {n} is not divisible by {qk}")
        }
        // Generic implementation.
        let mut sumf = 0f32;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let mut sum_i = 0;
            for j in 0..qk / 2 {
                let v0 = (xs.qs[j] & 0x0F) as i32 - 8;
                let v1 = (xs.qs[j] >> 4) as i32 - 8;
                sum_i += v0 * ys.qs[j] as i32 + v1 * ys.qs[j + qk / 2] as i32
            }
            sumf += sum_i as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
        }
        sumf
    }
}

impl QuantType for BlockQ4_0 {
    const BLCK_SIZE: usize = QK4_0;
    type VecDotType = BlockQ8_0;

    fn from_f32(src: &[f32], dst: &mut [Self]) -> GResult<()> {
        todo!()
    }

    fn vec_dot(me: &[Self], other: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx")]
        return Self::vec_dot_q8_0(me, other);

        return Self::vec_dot_unopt(me.len(), me, other);
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

                dst[i * qk + j + 0] = v0;
                dst[i * qk + j + qk / 2] = v1;
            }
        }
    }
}

pub(super) fn nearest_int(v: f32) -> i32 {
    v.round() as i32
}

pub(super) unsafe fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: *const f32,
    ls: *mut i8,
    rmse_type: i32,
) -> f32 {
    let mut max = 0f32;
    let mut amax = 0f32;
    for i in 0..n {
        let x = *x.add(i);
        let ax = x.abs();
        if ax > amax {
            amax = ax;
            max = x;
        }
    }
    if amax == 0. {
        // all zero
        for i in 0..n {
            *ls.add(i) = 0;
        }
        return 0.;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for i in 0..n {
            let x = *x.add(i);
            let l = nearest_int(iscale * x);
            *ls.add(i) = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let weight_type = rmse_type % 2;
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let x = *x.add(i);
        let l = nearest_int(iscale * x);
        let l = l.clamp(-nmax, nmax - 1);
        *ls.add(i) = (l + nmax) as i8;
        let w = if weight_type == 1 { x * x } else { 1.0 };
        let l = l as f32;
        sumlx += w * x * l;
        suml2 += w * l * l;
    }
    let mut scale = sumlx / suml2;
    let mut best = scale * sumlx;
    for _itry in 0..3 {
        let iscale = 1.0 / scale;
        let mut slx = 0f32;
        let mut sl2 = 0f32;
        let mut changed = false;
        for i in 0..n {
            let x = *x.add(i);
            let l = nearest_int(iscale * x);
            let l = l.clamp(-nmax, nmax - 1);
            if l + nmax != *ls.add(i) as i32 {
                changed = true;
            }
            let w = if weight_type == 1 { x * x } else { 1f32 };
            let l = l as f32;
            slx += w * x * l;
            sl2 += w * l * l;
        }
        if !changed || sl2 == 0.0 || slx * slx <= best * sl2 {
            break;
        }
        for i in 0..n {
            let x = *x.add(i);
            let l = nearest_int(iscale * x);
            *ls.add(i) = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
        }
        sumlx = slx;
        suml2 = sl2;
        scale = sumlx / suml2;
        best = scale * sumlx;
    }
    for _itry in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let x = *x.add(i);
            let w = if weight_type == 1 { x * x } else { 1. };
            let l = *ls.add(i) as i32 - nmax;
            let mut slx = sumlx - w * x * l as f32;
            if slx > 0. {
                let mut sl2 = suml2 - w * l as f32 * l as f32;
                let new_l = nearest_int(x * sl2 / slx);
                let new_l = new_l.clamp(-nmax, nmax - 1);
                if new_l != l {
                    slx += w * x * new_l as f32;
                    sl2 += w * new_l as f32 * new_l as f32;
                    if sl2 > 0. && slx * slx * suml2 > sumlx * sumlx * sl2 {
                        *ls.add(i) = (nmax + new_l) as i8;
                        sumlx = slx;
                        suml2 = sl2;
                        scale = sumlx / suml2;
                        best = scale * sumlx;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if rmse_type < 3 {
        return scale;
    }
    for is in -4..4 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1f32 * is as f32) / max;
        let mut sumlx = 0.;
        let mut suml2 = 0.;
        for i in 0..n {
            let x = *x.add(i);
            let l = nearest_int(iscale * x);
            let l = l.clamp(-nmax, nmax - 1);
            let w = if weight_type == 1 { x * x } else { 1. };
            let l = l as f32;
            sumlx += w * x * l;
            suml2 += w * l * l;
        }
        if suml2 > 0. && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let x = *x.add(i);
                let l = nearest_int(iscale * x);
                *ls.add(i) = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

#[derive(Debug, Clone, PartialEq, Copy)]
// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/k_quants.h#L82
#[repr(C)]
pub(crate) struct BlockQ4K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () = assert!(QK_K / 2 + K_SCALE_SIZE + 2 * 2 == std::mem::size_of::<BlockQ4K>());

#[cfg(target_feature = "avx")]
#[inline(always)]
pub(crate) fn vec_dot_q4k_q8k(n: usize, xs: &[BlockQ4K], ys: &[BlockQ8K]) -> GResult<f32> {
    if n % QK_K != 0 {
        panic!("vec_dot_q4k_q8k: {n} is not divisible by {QK_K}")
    }
    let mut utmp = [0u32; 4];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    unsafe {
        let m4 = _mm256_set1_epi8(0xF);

        let mut acc = _mm256_setzero_ps();
        let mut acc_m = _mm_setzero_ps();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            let mut q4 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp[3] as i32,
                utmp[2] as i32,
                utmp[1] as i32,
                utmp[0] as i32,
            ));

            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = mm256_set_m128i(sc128, sc128);

            let mut sumi = _mm256_setzero_si256();

            for j in 0..QK_K / 64 {
                let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                let q4bits = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4l = _mm256_and_si256(q4bits, m4);
                let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                let q8l = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let p16l = _mm256_maddubs_epi16(q4l, q8l);
                let p16l = _mm256_madd_epi16(scale_l, p16l);
                sumi = _mm256_add_epi32(sumi, p16l);

                let q8h = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let p16h = _mm256_maddubs_epi16(q4h, q8h);
                let p16h = _mm256_madd_epi16(scale_h, p16h);
                sumi = _mm256_add_epi32(sumi, p16h);
            }

            let vd = _mm256_set1_ps(d);
            acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
        }

        let acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        let acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

        Ok(hsum_float_8(acc) + _mm_cvtss_f32(acc_m))
    }
}

impl QuantType for BlockQ4K {
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;
    fn from_f32(src: &[f32], dst: &mut [Self]) -> GResult<()> {
        todo!()
    }

    fn vec_dot(me: &[Self], other: &[Self::VecDotType]) -> f32 {
        todo!()
    }

    fn to_f32(src: &[Self], dst: &mut [f32]) {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
#[repr(C)]
pub(crate) struct BlockQ6K {
    pub(crate) ql: [u8; QK_K / 2],
    pub(crate) qh: [u8; QK_K / 4],
    pub(crate) scales: [i8; QK_K / 16],
    pub(crate) d: f16,
}

impl QuantType for BlockQ6K {
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;
    fn from_f32(src: &[f32], dst: &mut [Self]) -> GResult<()> {
        if src.len() != dst.len() * Self::BLCK_SIZE {
            panic!(
                "quantize_row_q6k: size mismatch {} {} {}",
                src.len(),
                dst.len(),
                Self::BLCK_SIZE
            )
        }
        let mut l = [0i8; QK_K];
        let mut scales = [0f32; QK_K / 16];
        let mut x = src.as_ptr();
        let l = l.as_mut_ptr();
        unsafe {
            for y in dst.iter_mut() {
                let mut max_scale = 0f32;
                let mut max_abs_scale = 0f32;
                for (ib, scale_) in scales.iter_mut().enumerate() {
                    let scale = make_qx_quants(16, 32, x.add(16 * ib), l.add(16 * ib), 1);
                    *scale_ = scale;
                    let abs_scale = scale.abs();
                    if abs_scale > max_abs_scale {
                        max_abs_scale = abs_scale;
                        max_scale = scale
                    }
                }

                let iscale = -128f32 / max_scale;
                y.d = f16::from_f32(1.0 / iscale);

                for (y_scale, scale) in y.scales.iter_mut().zip(scales.iter()) {
                    *y_scale = nearest_int(iscale * scale).min(127) as i8
                }

                for (j, &y_scale) in y.scales.iter().enumerate() {
                    let d = y.d.to_f32() * y_scale as f32;
                    if d == 0. {
                        continue;
                    }
                    for ii in 0..16 {
                        let ll = nearest_int(*x.add(16 * j + ii) / d).clamp(-32, 31);
                        *l.add(16 * j + ii) = (ll + 32) as i8
                    }
                }

                let mut ql = y.ql.as_mut_ptr();
                let mut qh = y.qh.as_mut_ptr();

                for j in (0..QK_K).step_by(128) {
                    for l_idx in 0..32 {
                        let q1 = *l.add(j + l_idx) & 0xF;
                        let q2 = *l.add(j + l_idx + 32) & 0xF;
                        let q3 = *l.add(j + l_idx + 64) & 0xF;
                        let q4 = *l.add(j + l_idx + 96) & 0xF;
                        *ql.add(l_idx) = (q1 | (q3 << 4)) as u8;
                        *ql.add(l_idx + 32) = (q2 | (q4 << 4)) as u8;
                        *qh.add(l_idx) = ((*l.add(j + l_idx) >> 4)
                            | ((*l.add(j + l_idx + 32) >> 4) << 2)
                            | ((*l.add(j + l_idx + 64) >> 4) << 4)
                            | ((*l.add(j + l_idx + 96) >> 4) << 6))
                            as u8;
                    }
                    ql = ql.add(64);
                    qh = qh.add(32);
                }

                x = x.add(QK_K)
            }
        }
        Ok(())
    }

    fn to_f32(src: &[Self], dst: &mut [f32]) {}

    fn vec_dot(xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        // let n = xs.len();
        // let qk = QK_K;
        // if n % qk != 0 {
        //     panic!("vec_dot_q6k_8k: {n} is not divisible by {qk}")
        // }
        assert!(xs.len() == ys.len());

        unsafe {
            let m4 = _mm256_set1_epi8(0xF);
            let m2 = _mm256_set1_epi8(3);
            let m32s = _mm256_set1_epi8(32);
            let mut acc = _mm256_setzero_ps();
            for (x, y) in xs.iter().zip(ys.iter()) {
                let d = y.d * x.d.to_f32();
                let mut q4 = x.ql.as_ptr();
                let mut qh = x.qh.as_ptr();
                let mut q8 = y.qs.as_ptr();

                let scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
                let mut sumi = _mm256_setzero_si256();

                for j in 0..QK_K / 128 {
                    let is = j * 4;
                    let scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is));
                    let scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
                    let scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
                    let scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));

                    let q4bits1 = _mm256_loadu_si256(q4 as *const __m256i);
                    q4 = q4.add(32);
                    let q4bits2 = _mm256_loadu_si256(q4 as *const __m256i);
                    q4 = q4.add(32);
                    let q4bits_h = _mm256_loadu_si256(qh as *const __m256i);
                    qh = qh.add(32);

                    let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m2), 4);
                    let q4h_1 =
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 2), m2), 4);
                    let q4h_2 =
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 4), m2), 4);
                    let q4h_3 =
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 6), m2), 4);

                    let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                    let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                    let q4_2 =
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
                    let q4_3 =
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

                    let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);

                    let q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
                    let q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
                    let q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
                    let q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

                    let p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                    let p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                    let p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                    let p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

                    let p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                    let p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                    let p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                    let p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

                    let p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
                    let p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
                    let p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
                    let p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
                }
                acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
            }
            hsum_float_8(acc)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub(crate) struct BlockQ8K {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK_K],
    pub(crate) bsums: [i16; QK_K / 16],
}

impl QuantType for BlockQ8K {
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;
    fn from_f32(src: &[f32], dst: &mut [Self]) -> GResult<()> {
        let k = src.len();
        if k % QK_K != 0 {
            panic!("quantize_row_q8k: {k} is not divisible by {QK_K}")
        }
        for (i, y) in dst.iter_mut().enumerate() {
            let mut max = 0f32;
            let mut amax = 0f32;
            let xs = &src[i * QK_K..(i + 1) * QK_K];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            if amax == 0f32 {
                y.d = 0f32;
                y.qs.fill(0)
            } else {
                let iscale = -128f32 / max;
                for (j, q) in y.qs.iter_mut().enumerate() {
                    // ggml uses nearest_int with bit magic here, maybe we want the same
                    // but we would have to test and benchmark it.
                    let v = (iscale * xs[j]).round();
                    *q = v.min(127.) as i8
                }
                for j in 0..QK_K / 16 {
                    let mut sum = 0i32;
                    for ii in 0..16 {
                        sum += y.qs[j * 16 + ii] as i32
                    }
                    y.bsums[j] = sum as i16
                }
                y.d = 1.0 / iscale
            }
        }
        Ok(())
    }

    fn to_f32(src: &[Self], dst: &mut [f32]) {
        todo!()
    }

    fn vec_dot(me: &[Self], other: &[Self::VecDotType]) -> f32 {
        todo!()
    }
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

unsafe impl ValidAsZeroBits for BlockQ8_1 {}
unsafe impl DeviceRepr for BlockQ8_1 {}

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_1 {
    pub(crate) d: f16,
    pub(crate) s: f16,
    pub(crate) qs: [i8; QK8_1],
}

impl QuantType for BlockQ8_1 {
    const BLCK_SIZE: usize = QK8_1;
    type VecDotType = BlockQ8_0;

    fn from_f32(src: &[f32], dst: &mut [Self]) -> GResult<()> {
        todo!()
    }

    fn vec_dot(me: &[Self], other: &[Self::VecDotType]) -> f32 {
        todo!()
    }

    fn to_f32(src: &[Self], dst: &mut [f32]) {
        todo!()
    }
}
