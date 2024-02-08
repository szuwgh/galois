use std::string::FromUtf16Error;

use crate::op;
use crate::DTensorIter;
use crate::FromF32;
use crate::FromF64;
use crate::Shape;
use crate::ToUsize;
use crate::UnaryOp;
use crate::{DTensor, TensorType};
use half::f16;
use num_traits::ToPrimitive;
#[derive(Debug)]
pub enum DType {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
}

#[macro_export]
macro_rules! impl_tousize {
    ($($e:ident),*) => {
        $(impl ToUsize for $e {
            fn as_usize(&self) -> usize {
                *self as usize
            }
        })*
    };
}

#[macro_export]
macro_rules! impl_fromf32 {
    ($($e:ident),*) => {
        $(impl FromF32 for $e {
            fn from_f32(a: f32) -> Self {
                a as $e
            }
        })*
    };
}

#[macro_export]
macro_rules! impl_fromf64 {
    ($($e:ident),*) => {
        $(impl FromF64 for $e {
            fn from_f64(a: f64) -> Self {
                a as $e
            }
        })*
    };
}

#[macro_export]
macro_rules! impl_no_unary_op {
    ($($e:ident),*) => {
        $(impl UnaryOp for $e {
                fn _exp(&self) -> Self {
                   todo!()
                }
                fn _ln(&self) -> Self {
                    todo!()
                }
                fn _sin(&self) -> Self {
                    todo!()
                }
                fn _cos(&self) -> Self {
                    todo!()
                }
                fn _tanh(&self) -> Self {
                    todo!()
                }
                fn _neg(&self) -> Self {
                    todo!()
                }
                fn _recip(&self) -> Self {
                    todo!()
                }
                fn _sqr(&self) -> Self {
                    todo!()
                }
                fn _sqrt(&self) -> Self {
                    todo!()
                }
        })*
    };
}

impl ToUsize for f16 {
    fn as_usize(&self) -> usize {
        self.to_usize().unwrap()
    }
}

impl FromF32 for f16 {
    fn from_f32(a: f32) -> Self {
        f16::from_f32(a)
    }
}

impl FromF64 for f16 {
    fn from_f64(a: f64) -> Self {
        f16::from_f64(a)
    }
}

impl_tousize!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_fromf32!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_fromf64!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_no_unary_op!(u8, u16, u32, u64, i8, i16, i32, i64);

impl TensorType for u8 {}
impl TensorType for u16 {}
impl TensorType for u32 {}
impl TensorType for u64 {}
impl TensorType for i8 {}
impl TensorType for i16 {}
impl TensorType for i32 {}
impl TensorType for i64 {}
impl TensorType for f16 {}
impl TensorType for f32 {}
impl TensorType for f64 {}

#[derive(Debug, Clone)]
pub enum Tensor {
    U8(DTensor<u8>),
    U16(DTensor<u16>),
    U32(DTensor<u32>),
    U64(DTensor<u64>),
    I8(DTensor<i8>),
    I16(DTensor<i16>),
    I32(DTensor<i32>),
    I64(DTensor<i64>),
    F16(DTensor<f16>),
    F32(DTensor<f32>),
    F64(DTensor<f64>),
}

impl Tensor {
    pub fn dtype(&self) -> DType {
        return match self {
            Tensor::U8(_) => DType::U8,
            Tensor::I8(_) => DType::I8,
            Tensor::I16(_) => DType::I16,
            Tensor::U16(_) => DType::U16,
            Tensor::F16(_) => DType::F16,
            Tensor::F32(_) => DType::F32,
            Tensor::I32(_) => DType::I32,
            Tensor::U32(_) => DType::U32,
            Tensor::I64(_) => DType::I64,
            Tensor::F64(_) => DType::F64,
            Tensor::U64(_) => DType::U64,
        };
    }

    pub fn shape(&self) -> &Shape {
        return match self {
            Tensor::U8(t) => t.shape(),
            Tensor::I8(t) => t.shape(),
            Tensor::I16(t) => t.shape(),
            Tensor::U16(t) => t.shape(),
            Tensor::F16(t) => t.shape(),
            Tensor::F32(t) => t.shape(),
            Tensor::I32(t) => t.shape(),
            Tensor::U32(t) => t.shape(),
            Tensor::I64(t) => t.shape(),
            Tensor::F64(t) => t.shape(),
            Tensor::U64(t) => t.shape(),
        };
    }

    pub fn from_raw_data(ptr: *mut u8, length: usize, s: Shape, t: DType) -> Self {
        match t {
            DType::U8 => Tensor::U8(DTensor::<u8>::from_raw_data(
                ptr as _,
                length / ::std::mem::size_of::<u8>(),
                s,
            )),
            DType::U16 => Tensor::U16(DTensor::<u16>::from_raw_data(
                ptr as _,
                length / ::std::mem::size_of::<u16>(),
                s,
            )),
            DType::U32 => Tensor::U32(DTensor::<u32>::from_raw_data(
                ptr as _,
                length / ::std::mem::size_of::<u32>(),
                s,
            )),
            DType::U64 => Tensor::U64(DTensor::<u64>::from_raw_data(
                ptr as _,
                length / ::std::mem::size_of::<u64>(),
                s,
            )),
            DType::I8 => Tensor::I8(DTensor::<i8>::from_raw_data(
                ptr as _,
                length / ::std::mem::size_of::<i8>(),
                s,
            )),
            DType::I16 => Tensor::I16(DTensor::<i16>::from_raw_data(
                ptr as _,
                length / ::std::mem::size_of::<i16>(),
                s,
            )),
            DType::I32 => Tensor::I32(DTensor::<i32>::from_raw_data(
                ptr as _,
                length / ::std::mem::size_of::<i32>(),
                s,
            )),
            DType::I64 => Tensor::I64(DTensor::<i64>::from_raw_data(
                ptr as _,
                length / ::std::mem::size_of::<i64>(),
                s,
            )),
            DType::F16 => Tensor::F16(DTensor::<f16>::from_raw_data(
                ptr as _,
                length / ::std::mem::size_of::<f16>(),
                s,
            )),
            DType::F32 => Tensor::F32(DTensor::<f32>::from_raw_data(
                ptr as _,
                length / ::std::mem::size_of::<f32>(),
                s,
            )),
            DType::F64 => Tensor::F64(DTensor::<f64>::from_raw_data(
                ptr as _,
                length / ::std::mem::size_of::<f64>(),
                s,
            )),
        }
    }
}

impl PartialEq<Tensor> for Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        return match (self, other) {
            (Tensor::U8(t1), Tensor::U8(t2)) => t1 == t2,
            (Tensor::I8(t1), Tensor::I8(t2)) => t1 == t2,
            (Tensor::I16(t1), Tensor::I16(t2)) => t1 == t2,
            (Tensor::U16(t1), Tensor::U16(t2)) => t1 == t2,
            (Tensor::F16(t1), Tensor::F16(t2)) => t1 == t2,
            (Tensor::F32(t1), Tensor::F32(t2)) => t1 == t2,
            (Tensor::I32(t1), Tensor::I32(t2)) => t1 == t2,
            (Tensor::U32(t1), Tensor::U32(t2)) => t1 == t2,
            (Tensor::I64(t1), Tensor::I64(t2)) => t1 == t2,
            (Tensor::F64(t1), Tensor::F64(t2)) => t1 == t2,
            (Tensor::U64(t1), Tensor::U64(t2)) => t1 == t2,
            _ => false,
        };
    }
}

// mod tests {
//     use super::*;
//     use crate::{arr, mat};

//     #[test]
//     fn test_tensor_add() {
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

//         let t1 = Tensor::F64(m1);
//         let t2 = Tensor::F64(m2);

//         let m3 = t1 + &t2;
//         println!("m3:{:?}", m3);
//     }
// }
