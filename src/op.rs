use crate::tensor::Tensor;
use crate::Shape;

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        shape(self)
    }

    pub fn reshape(&self, d: Shape) -> Tensor {
        reshape(self, d)
    }
}

macro_rules! binary_op {
    ($trt:ident, $mth:ident) => {
        impl std::ops::$trt<&Tensor> for Tensor {
            type Output = Tensor;
            fn $mth(self, rhs: &Tensor) -> Self::Output {}
        }

        impl std::ops::$trt<Tensor> for Tensor {
            type Output = Tensor;
            fn $mth(self, rhs: Tensor) -> Self::Output {
                self + rhs
            }
        }

        impl std::ops::$trt<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $mth(self, rhs: &Tensor) -> Self::Output {
                self + rhs
            }
        }

        impl std::ops::$trt<Tensor> for &Tensor {
            type Output = Tensor;
            fn $mth(self, rhs: Tensor) -> Self::Output {
                self + rhs
            }
        }
    };
}

macro_rules! op {
    ($f:ident, $ty1:ty, $ty2:ty) => {
        fn $f(t1: &Tensor) -> Tensor {
            return match (self, rhs) {
                (Tensor::U8(t1), Tensor::U8(t2)) => Tensor::U8(t1 + t2),
                (Tensor::I8(t1), Tensor::I8(t2)) => Tensor::I8(t1 + t2),
                (Tensor::I16(t1), Tensor::I16(t2)) => Tensor::I16(t1 + t2),
                (Tensor::U16(t1), Tensor::U16(t2)) => Tensor::U16(t1 + t2),
                (Tensor::F16(t1), Tensor::F16(t2)) => Tensor::F16(t1 + t2),
                (Tensor::F32(t1), Tensor::F32(t2)) => Tensor::F32(t1 + t2),
                (Tensor::I32(t1), Tensor::I32(t2)) => Tensor::I32(t1 + t2),
                (Tensor::U32(t1), Tensor::U32(t2)) => Tensor::U32(t1 + t2),
                (Tensor::I64(t1), Tensor::I64(t2)) => Tensor::I64(t1 + t2),
                (Tensor::F64(t1), Tensor::F64(t2)) => Tensor::F64(t1 + t2),
                (Tensor::U64(t1), Tensor::U64(t2)) => Tensor::U64(t1 + t2),
                _ => {
                    panic!("types do not match");
                }
            };
        }
    };
}

macro_rules! method {
    ($f:ident, $re:ty) => {
        fn $f(t1: &Tensor) -> $re {
            return match t1 {
                Tensor::U8(t) => t.$f(),
                Tensor::I8(t) => t.$f(),
                Tensor::I16(t) => t.$f(),
                Tensor::U16(t) => t.$f(),
                Tensor::F16(t) => t.$f(),
                Tensor::F32(t) => t.$f(),
                Tensor::I32(t) => t.$f(),
                Tensor::U32(t) => t.$f(),
                Tensor::I64(t) => t.$f(),
                Tensor::F64(t) => t.$f(),
                Tensor::U64(t) => t.$f(),
            };
        }
    };

    ($f:ident, $re:ty, $p1:ty) => {
        fn $f(t1: &Tensor, p1: $p1) -> $re {
            return match t1 {
                Tensor::U8(t) => Tensor::U8(t.$f(p1)),
                Tensor::I8(t) => Tensor::I8(t.$f(p1)),
                Tensor::I16(t) => Tensor::I16(t.$f(p1)),
                Tensor::U16(t) => Tensor::U16(t.$f(p1)),
                Tensor::F16(t) => Tensor::F16(t.$f(p1)),
                Tensor::F32(t) => Tensor::F32(t.$f(p1)),
                Tensor::I32(t) => Tensor::I32(t.$f(p1)),
                Tensor::U32(t) => Tensor::U32(t.$f(p1)),
                Tensor::I64(t) => Tensor::I64(t.$f(p1)),
                Tensor::F64(t) => Tensor::F64(t.$f(p1)),
                Tensor::U64(t) => Tensor::U64(t.$f(p1)),
            };
        }
    };

    ($f:ident, $re:ty, $p1:ty, $p2:ty) => {
        fn $f(t1: &Tensor, p1: $p1, p2: $p2) -> $re {
            return match t1 {
                Tensor::U8(t) => Tensor::U8(t.$f(p1, p2)),
                Tensor::I8(t) => Tensor::I8(t.$f(p1, p2)),
                Tensor::I16(t) => Tensor::I16(t.$f(p1, p2)),
                Tensor::U16(t) => Tensor::U16(t.$f(p1, p2)),
                Tensor::F16(t) => Tensor::F16(t.$f(p1, p2)),
                Tensor::F32(t) => Tensor::F32(t.$f(p1, p2)),
                Tensor::I32(t) => Tensor::I32(t.$f(p1, p2)),
                Tensor::U32(t) => Tensor::U32(t.$f(p1, p2)),
                Tensor::I64(t) => Tensor::I64(t.$f(p1, p2)),
                Tensor::F64(t) => Tensor::F64(t.$f(p1, p2)),
                Tensor::U64(t) => Tensor::U64(t.$f(p1, p2)),
            };
        }
    };
}

method!(shape, &[usize]);
method!(reshape, Tensor, Shape);
