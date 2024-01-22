use crate::tensor::Tensor;
// use crate::DTensor;
// use crate::Shape;

// macro_rules! binary_op {
//     ($trt:ident, $mth:ident, $mth1:ident,$mth2:ident,$mth3:ident,$mth4:ident) => {
//         impl std::ops::$trt<Tensor> for Tensor {
//             type Output = Tensor;
//             fn $mth(self, rhs: Tensor) -> Self::Output {
//                 $mth1(self, rhs)
//             }
//         }

//         impl std::ops::$trt<&Tensor> for Tensor {
//             type Output = Tensor;
//             fn $mth(self, rhs: &Tensor) -> Self::Output {
//                 $mth2(self, rhs)
//             }
//         }

//         impl std::ops::$trt<Tensor> for &Tensor {
//             type Output = Tensor;
//             fn $mth(self, rhs: Tensor) -> Self::Output {
//                 $mth3(self, rhs)
//             }
//         }

//         impl std::ops::$trt<&Tensor> for &Tensor {
//             type Output = Tensor;
//             fn $mth(self, rhs: &Tensor) -> Self::Output {
//                 $mth4(self, rhs)
//             }
//         }
//     };
// }

// macro_rules! op {
//     ($f:ident, $ty1:ty, $ty2:ty, $p:tt) => {
//        pub(crate) fn $f(lhs: $ty1, rhs:$ty2) -> Tensor {
//             return match (lhs, rhs) {
//                 (Tensor::U8(t1), Tensor::U8(t2)) => Tensor::U8(t1 $p t2),
//                 (Tensor::I8(t1), Tensor::I8(t2)) => Tensor::I8(t1 $p t2),
//                 (Tensor::I16(t1), Tensor::I16(t2)) => Tensor::I16(t1 $p t2),
//                 (Tensor::U16(t1), Tensor::U16(t2)) => Tensor::U16(t1 $p t2),
//                 (Tensor::F16(t1), Tensor::F16(t2)) => Tensor::F16(t1 $p t2),
//                 (Tensor::F32(t1), Tensor::F32(t2)) => Tensor::F32(t1 $p t2),
//                 (Tensor::I32(t1), Tensor::I32(t2)) => Tensor::I32(t1 $p t2),
//                 (Tensor::U32(t1), Tensor::U32(t2)) => Tensor::U32(t1 $p t2),
//                 (Tensor::I64(t1), Tensor::I64(t2)) => Tensor::I64(t1 $p t2),
//                 (Tensor::F64(t1), Tensor::F64(t2)) => Tensor::F64(t1 $p t2),
//                 (Tensor::U64(t1), Tensor::U64(t2)) => Tensor::U64(t1 $p t2),
//                 _ => {
//                     panic!("types do not match");
//                 }
//             };
//         }
//     };
// }

// macro_rules! method {
//     ($f:ident, $re:ty) => {
//         pub(crate) fn $f(t1: &Tensor) -> $re {
//             return match t1 {
//                 Tensor::U8(t) => t.$f(),
//                 Tensor::I8(t) => t.$f(),
//                 Tensor::I16(t) => t.$f(),
//                 Tensor::U16(t) => t.$f(),
//                 Tensor::F16(t) => t.$f(),
//                 Tensor::F32(t) => t.$f(),
//                 Tensor::I32(t) => t.$f(),
//                 Tensor::U32(t) => t.$f(),
//                 Tensor::I64(t) => t.$f(),
//                 Tensor::F64(t) => t.$f(),
//                 Tensor::U64(t) => t.$f(),
//             };
//         }
//     };

//     ($f:ident, $re:ty, $p1:ty) => {
//         pub(crate) fn $f(t1: &Tensor, p1: $p1) -> $re {
//             return match t1 {
//                 Tensor::U8(t) => Tensor::U8(t.$f(p1)),
//                 Tensor::I8(t) => Tensor::I8(t.$f(p1)),
//                 Tensor::I16(t) => Tensor::I16(t.$f(p1)),
//                 Tensor::U16(t) => Tensor::U16(t.$f(p1)),
//                 Tensor::F16(t) => Tensor::F16(t.$f(p1)),
//                 Tensor::F32(t) => Tensor::F32(t.$f(p1)),
//                 Tensor::I32(t) => Tensor::I32(t.$f(p1)),
//                 Tensor::U32(t) => Tensor::U32(t.$f(p1)),
//                 Tensor::I64(t) => Tensor::I64(t.$f(p1)),
//                 Tensor::F64(t) => Tensor::F64(t.$f(p1)),
//                 Tensor::U64(t) => Tensor::U64(t.$f(p1)),
//             };
//         }
//     };

//     ($f:ident, $re:ty, $p1:ty, $p2:ty) => {
//         pub(crate) fn $f(t1: &Tensor, p1: $p1, p2: $p2) -> $re {
//             return match t1 {
//                 Tensor::U8(t) => Tensor::U8(t.$f(p1, p2)),
//                 Tensor::I8(t) => Tensor::I8(t.$f(p1, p2)),
//                 Tensor::I16(t) => Tensor::I16(t.$f(p1, p2)),
//                 Tensor::U16(t) => Tensor::U16(t.$f(p1, p2)),
//                 Tensor::F16(t) => Tensor::F16(t.$f(p1, p2)),
//                 Tensor::F32(t) => Tensor::F32(t.$f(p1, p2)),
//                 Tensor::I32(t) => Tensor::I32(t.$f(p1, p2)),
//                 Tensor::U32(t) => Tensor::U32(t.$f(p1, p2)),
//                 Tensor::I64(t) => Tensor::I64(t.$f(p1, p2)),
//                 Tensor::F64(t) => Tensor::F64(t.$f(p1, p2)),
//                 Tensor::U64(t) => Tensor::U64(t.$f(p1, p2)),
//             };
//         }
//     };
// }

// op!(add1, Tensor, Tensor, +);
// op!(add2, Tensor, &Tensor, +);
// op!(add3, &Tensor, Tensor, +);
// op!(add4, &Tensor, &Tensor, +);

// op!(sub1, Tensor, Tensor, -);
// op!(sub2, Tensor, &Tensor, -);
// op!(sub3, &Tensor, Tensor, -);
// op!(sub4, &Tensor, &Tensor, -);

// binary_op!(Add, add, add1, add2, add3, add4);
// binary_op!(Sub, sub, sub1, sub2, sub3, sub4);

// method!(shape, &Shape);
// method!(reshape, Tensor, Shape);

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
