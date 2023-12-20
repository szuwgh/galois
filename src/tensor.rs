use crate::op;
use crate::DTensorIter;
use crate::Shape;
use crate::{DTensor, TensorType};
use half::f16;
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

pub enum TensorValue<'a> {
    U8(&'a mut u8),
    U16(&'a mut u16),
    U32(&'a mut u32),
    U64(&'a mut u64),
    I8(&'a mut i8),
    I16(&'a mut i16),
    I32(&'a mut i32),
    I64(&'a mut i64),
    F16(&'a mut f16),
    F32(&'a mut f32),
    F64(&'a mut f64),
}

pub enum TensorIter<'a> {
    U8(DTensorIter<'a, u8>),
    U16(DTensorIter<'a, u16>),
    U32(DTensorIter<'a, u32>),
    U64(DTensorIter<'a, u64>),
    I8(DTensorIter<'a, i8>),
    I16(DTensorIter<'a, i16>),
    I32(DTensorIter<'a, i32>),
    I64(DTensorIter<'a, i64>),
    F16(DTensorIter<'a, f16>),
    F32(DTensorIter<'a, f32>),
    F64(DTensorIter<'a, f64>),
}

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

impl<'a> Iterator for TensorIter<'a> {
    type Item = TensorValue<'a>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        return match self {
            TensorIter::U8(t) => Some(TensorValue::U8(t.next()?)),
            TensorIter::I8(t) => Some(TensorValue::I8(t.next()?)),
            TensorIter::I16(t) => Some(TensorValue::I16(t.next()?)),
            TensorIter::U16(t) => Some(TensorValue::U16(t.next()?)),
            TensorIter::F16(t) => Some(TensorValue::F16(t.next()?)),
            TensorIter::F32(t) => Some(TensorValue::F32(t.next()?)),
            TensorIter::I32(t) => Some(TensorValue::I32(t.next()?)),
            TensorIter::U32(t) => Some(TensorValue::U32(t.next()?)),
            TensorIter::I64(t) => Some(TensorValue::I64(t.next()?)),
            TensorIter::F64(t) => Some(TensorValue::F64(t.next()?)),
            TensorIter::U64(t) => Some(TensorValue::U64(t.next()?)),
        };
    }
}

impl Tensor {
    pub fn shape(&self) -> &Shape {
        op::shape(self)
    }

    pub fn reshape(&self, d: Shape) -> Tensor {
        op::reshape(self, d)
    }

    pub fn iter<'a>(&'a self) -> TensorIter<'a> {
        return match self {
            Tensor::U8(t) => TensorIter::U8(t.iter()),
            Tensor::I8(t) => TensorIter::I8(t.iter()),
            Tensor::I16(t) => TensorIter::I16(t.iter()),
            Tensor::U16(t) => TensorIter::U16(t.iter()),
            Tensor::F16(t) => TensorIter::F16(t.iter()),
            Tensor::F32(t) => TensorIter::F32(t.iter()),
            Tensor::I32(t) => TensorIter::I32(t.iter()),
            Tensor::U32(t) => TensorIter::U32(t.iter()),
            Tensor::I64(t) => TensorIter::I64(t.iter()),
            Tensor::F64(t) => TensorIter::F64(t.iter()),
            Tensor::U64(t) => TensorIter::U64(t.iter()),
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

mod tests {
    use super::*;
    use crate::{arr, mat};

    #[test]
    fn test_tensor_add() {
        let m1 = mat(&[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]);
        println!("m1 dim:{:?}", m1.dim);
        println!("m1 {:?}", m1);
        println!("m1 stride:{:?}", m1.dim.stride);
        let m2 = arr(&[1.0, 2.0, 3.0]);

        let t1 = Tensor::F64(m1);
        let t2 = Tensor::F64(m2);

        let m3 = t1 + &t2;
        println!("m3:{:?}", m3);
    }
}
