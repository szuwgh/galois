use std::string::FromUtf16Error;

use crate::op;
use crate::FromF32;
use crate::FromF64;
use crate::Shape;
use crate::TensorIter;
use crate::ToUsize;
use crate::UnaryOp;
use crate::{Tensor, TensorType};
use half::f16;
use num_traits::ToPrimitive;

// #[derive(Debug, Clone)]
// pub enum Tensor {
//     U8(Tensor<u8>),
//     U16(Tensor<u16>),
//     U32(Tensor<u32>),
//     U64(Tensor<u64>),
//     I8(Tensor<i8>),
//     I16(Tensor<i16>),
//     I32(Tensor<i32>),
//     I64(Tensor<i64>),
//     F16(Tensor<f16>),
//     F32(Tensor<f32>),
//     F64(Tensor<f64>),
// }

// impl Tensor {
//     pub fn dtype(&self) -> DType {
//         return match self {
//             Tensor::U8(_) => DType::U8,
//             Tensor::I8(_) => DType::I8,
//             Tensor::I16(_) => DType::I16,
//             Tensor::U16(_) => DType::U16,
//             Tensor::F16(_) => DType::F16,
//             Tensor::F32(_) => DType::F32,
//             Tensor::I32(_) => DType::I32,
//             Tensor::U32(_) => DType::U32,
//             Tensor::I64(_) => DType::I64,
//             Tensor::F64(_) => DType::F64,
//             Tensor::U64(_) => DType::U64,
//         };
//     }

//     pub fn shape(&self) -> &Shape {
//         return match self {
//             Tensor::U8(t) => t.shape(),
//             Tensor::I8(t) => t.shape(),
//             Tensor::I16(t) => t.shape(),
//             Tensor::U16(t) => t.shape(),
//             Tensor::F16(t) => t.shape(),
//             Tensor::F32(t) => t.shape(),
//             Tensor::I32(t) => t.shape(),
//             Tensor::U32(t) => t.shape(),
//             Tensor::I64(t) => t.shape(),
//             Tensor::F64(t) => t.shape(),
//             Tensor::U64(t) => t.shape(),
//         };
//     }

//     pub fn from_raw_data(ptr: *mut u8, length: usize, s: Shape, t: DType) -> Self {
//         match t {
//             DType::U8 => Tensor::U8(Tensor::<u8>::from_raw_data(
//                 ptr as _,
//                 length / ::std::mem::size_of::<u8>(),
//                 s,
//             )),
//             DType::U16 => Tensor::U16(Tensor::<u16>::from_raw_data(
//                 ptr as _,
//                 length / ::std::mem::size_of::<u16>(),
//                 s,
//             )),
//             DType::U32 => Tensor::U32(Tensor::<u32>::from_raw_data(
//                 ptr as _,
//                 length / ::std::mem::size_of::<u32>(),
//                 s,
//             )),
//             DType::U64 => Tensor::U64(Tensor::<u64>::from_raw_data(
//                 ptr as _,
//                 length / ::std::mem::size_of::<u64>(),
//                 s,
//             )),
//             DType::I8 => Tensor::I8(Tensor::<i8>::from_raw_data(
//                 ptr as _,
//                 length / ::std::mem::size_of::<i8>(),
//                 s,
//             )),
//             DType::I16 => Tensor::I16(Tensor::<i16>::from_raw_data(
//                 ptr as _,
//                 length / ::std::mem::size_of::<i16>(),
//                 s,
//             )),
//             DType::I32 => Tensor::I32(Tensor::<i32>::from_raw_data(
//                 ptr as _,
//                 length / ::std::mem::size_of::<i32>(),
//                 s,
//             )),
//             DType::I64 => Tensor::I64(Tensor::<i64>::from_raw_data(
//                 ptr as _,
//                 length / ::std::mem::size_of::<i64>(),
//                 s,
//             )),
//             DType::F16 => Tensor::F16(Tensor::<f16>::from_raw_data(
//                 ptr as _,
//                 length / ::std::mem::size_of::<f16>(),
//                 s,
//             )),
//             DType::F32 => Tensor::F32(Tensor::<f32>::from_raw_data(
//                 ptr as _,
//                 length / ::std::mem::size_of::<f32>(),
//                 s,
//             )),
//             DType::F64 => Tensor::F64(Tensor::<f64>::from_raw_data(
//                 ptr as _,
//                 length / ::std::mem::size_of::<f64>(),
//                 s,
//             )),
//         }
//     }
// }

// impl PartialEq<Tensor> for Tensor {
//     fn eq(&self, other: &Tensor) -> bool {
//         return match (self, other) {
//             (Tensor::U8(t1), Tensor::U8(t2)) => t1 == t2,
//             (Tensor::I8(t1), Tensor::I8(t2)) => t1 == t2,
//             (Tensor::I16(t1), Tensor::I16(t2)) => t1 == t2,
//             (Tensor::U16(t1), Tensor::U16(t2)) => t1 == t2,
//             (Tensor::F16(t1), Tensor::F16(t2)) => t1 == t2,
//             (Tensor::F32(t1), Tensor::F32(t2)) => t1 == t2,
//             (Tensor::I32(t1), Tensor::I32(t2)) => t1 == t2,
//             (Tensor::U32(t1), Tensor::U32(t2)) => t1 == t2,
//             (Tensor::I64(t1), Tensor::I64(t2)) => t1 == t2,
//             (Tensor::F64(t1), Tensor::F64(t2)) => t1 == t2,
//             (Tensor::U64(t1), Tensor::U64(t2)) => t1 == t2,
//             _ => false,
//         };
//     }
// }
