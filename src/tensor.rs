use crate::DTensor;
use half::f16;

pub enum DType {
    U8,
    I8,
    I16,
    U16,
    U32,
    I64,
    F16,
    F32,
    F64,
    U64,
}

pub enum Tensor {
    U8(DTensor<u8>),
    I8(DTensor<i8>),
    I16(DTensor<i16>),
    U16(DTensor<u16>),
    F16(DTensor<f16>),
    F32(DTensor<f32>),
    I32(DTensor<i32>),
    U32(DTensor<u32>),
    I64(DTensor<i64>),
    F64(DTensor<f64>),
    U64(DTensor<u64>),
}
