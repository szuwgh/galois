pub const QK4_0: usize = 32;

#[derive(Debug, Clone, PartialEq, Copy)]
#[repr(C)]
pub(crate) struct BlockQ4_0 {
    pub(crate) d: f32,
    pub(crate) qs: [u8; QK4_0 / 2],
}
