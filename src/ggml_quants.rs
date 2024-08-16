use half::f16;

pub const QK4_0: usize = 32;

pub trait QuantType: Sized {
    const BLCK_SIZE: usize;
    fn to_f32(src: &[Self], dst: &mut [f32]);
}

#[derive(Debug, Clone, PartialEq, Copy)]
#[repr(C)]
pub struct BlockV1_Q4_0 {
    pub(crate) d: f32,
    pub(crate) qs: [u8; QK4_0 / 2],
}

impl BlockV1_Q4_0 {
    pub fn d(&self) -> f32 {
        self.d
    }
}

impl QuantType for BlockV1_Q4_0 {
    const BLCK_SIZE: usize = QK4_0;
    fn to_f32(src: &[Self], dst: &mut [f32]) {
        let k = dst.len();
        let qk = Self::BLCK_SIZE;
        assert!(k % qk == 0);
        let nb = k / qk;
        for i in 0..nb {
            let d = src[i].d;
            for j in (0..qk).step_by(2) {
                let vi = src[i].qs[j / 2];
                let vi0 = (vi & 0x0F) as i8;
                let vi1 = (vi >> 4) as i8;

                let v0 = (vi0 - 8) as f32 * d;
                let v1 = (vi1 - 8) as f32 * d;
                // println!(
                //     "d = {:.6}, vi = {}, vi0 = {}, vi1 = {}, v0 = {:.6}, v1 = {:.6}",
                //     d, vi, vi0, vi1, v0, v1,
                // );

                dst[i * qk + j + 0] = v0;
                dst[i * qk + j + 1] = v1;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
#[repr(C)]
pub struct Block_Q4_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_0 / 2],
}

impl Block_Q4_0 {
    pub fn d(&self) -> f16 {
        self.d
    }
}
