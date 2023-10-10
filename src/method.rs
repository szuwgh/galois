use super::broadcast::general_broadcasting;
use super::Tensor;

pub trait Distance {}

#[macro_export]
macro_rules! dim_max {
    ($d1:expr, $d2:expr) => {
        <$d1 as DimMax<$d2>>::Output
    };
}

fn convert_iopsf<A: Clone, B: Clone>(f: impl Fn(A, B) -> A) -> impl FnMut((&mut A, &B)) {
    move |(x, y)| *x = f(x.clone(), y.clone())
}

fn clone_opsf<A: Clone, B: Clone, C>(f: impl Fn(A, B) -> C) -> impl FnMut((&mut A, &B)) -> C {
    move |(x, y)| f(x.clone(), y.clone())
}

macro_rules! impl_binary_op {
    ($trt:ident, $mth:ident) => {
        impl<A> std::ops::$trt<&Tensor<A>> for Tensor<A>
        where
            A: std::ops::$trt<A, Output = A> + Clone,
        {
            type Output = Tensor<A>;
            fn $mth(self, rhs: &Tensor<A>) -> Self::Output {
                if self.shape() == rhs.shape() {
                    self.iter().zip(rhs.iter()).ops(convert_iopsf(A::$mth));
                    self
                } else {
                    let (lhs, rhs2) =
                        general_broadcasting::<A>(&self.as_ref(), &rhs.as_ref()).unwrap();
                    if lhs.shape() == self.shape() {
                        self.iter().zip(rhs2.iter()).ops(convert_iopsf(A::$mth));
                        self
                    } else {
                        lhs.iter()
                            .zip(rhs2.iter())
                            .map(clone_opsf(A::$mth))
                            .collect_tensor(lhs.dim.clone())
                    }
                }
            }
        }

        impl<A> std::ops::$trt<Tensor<A>> for Tensor<A>
        where
            A: std::ops::$trt<A, Output = A> + Clone,
        {
            type Output = Tensor<A>;
            fn $mth(self, rhs: Tensor<A>) -> Self::Output {
                if self.shape() == rhs.shape() {
                    self.iter().zip(rhs.iter()).ops(convert_iopsf(A::$mth));
                    self
                } else {
                    let (lhs, rhs2) =
                        general_broadcasting::<A>(&self.as_ref(), &rhs.as_ref()).unwrap();
                    if lhs.shape() == self.shape() {
                        self.iter().zip(rhs2.iter()).ops(convert_iopsf(A::$mth));
                        self
                    } else {
                        lhs.iter()
                            .zip(rhs2.iter())
                            .map(clone_opsf(A::$mth))
                            .collect_tensor(lhs.dim.clone())
                    }
                }
            }
        }

        impl<A> std::ops::$trt<&Tensor<A>> for &Tensor<A>
        where
            A: std::ops::$trt<A, Output = A> + Clone,
        {
            type Output = Tensor<A>;
            fn $mth(self, rhs: &Tensor<A>) -> Self::Output {
                if self.shape() == rhs.shape() {
                    self.iter()
                        .zip(rhs.iter())
                        .map(clone_opsf(A::$mth))
                        .collect_tensor(rhs.dim.clone())
                } else {
                    let (lhs, rhs2) =
                        general_broadcasting::<A>(&self.as_ref(), &rhs.as_ref()).unwrap();
                    lhs.iter()
                        .zip(rhs2.iter())
                        .map(clone_opsf(A::$mth))
                        .collect_tensor(lhs.dim.clone())
                }
            }
        }

        impl<A> std::ops::$trt<Tensor<A>> for &Tensor<A>
        where
            A: std::ops::$trt<A, Output = A> + Clone,
        {
            type Output = Tensor<A>;
            fn $mth(self, rhs: Tensor<A>) -> Self::Output {
                if self.shape() == rhs.shape() {
                    self.iter()
                        .zip(rhs.iter())
                        .map(clone_opsf(A::$mth))
                        .collect_tensor(rhs.dim.clone())
                } else {
                    let (lhs, rhs2) =
                        general_broadcasting::<A>(&self.as_ref(), &rhs.as_ref()).unwrap();
                    lhs.iter()
                        .zip(rhs2.iter())
                        .map(clone_opsf(A::$mth))
                        .collect_tensor(lhs.dim.clone())
                }
            }
        }
    };
}

impl_binary_op!(Add, add); // +
impl_binary_op!(Sub, sub); // -
impl_binary_op!(Mul, mul); // *
impl_binary_op!(Div, div); // /
impl_binary_op!(Rem, rem); // %
impl_binary_op!(BitAnd, bitand); // &
impl_binary_op!(BitOr, bitor); // |
impl_binary_op!(BitXor, bitxor); // ^
impl_binary_op!(Shl, shl); // <<
impl_binary_op!(Shr, shr); // >>

mod tests {
    use super::super::{arr, mat};
    use super::*;
    #[test]
    fn test_add() {
        let m1 = mat(&[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]);
        println!("m1 dim:{:?}", m1.dim);
        println!("m1 {:?}", m1);
        println!("m1 stride:{:?}", m1.stride);
        let m2 = arr(&[1.0, 2.0, 3.0]);

        let m3 = m1 + &m2;
        println!("m3:{:?}", m3);
    }

    #[test]
    fn test_sub() {
        let m1 = mat(&[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]);
        println!("m1 dim:{:?}", m1.dim);
        println!("m1 {:?}", m1);
        println!("m1 stride:{:?}", m1.stride);
        let m2 = arr(&[1.0, 2.0, 3.0]);

        let m3 = &m1 - &m2;
        println!("m3:{:?}", m3);
    }

    #[test]
    fn test_mul() {
        let m1 = mat(&[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]);
        println!("m1 dim:{:?}", m1.dim);
        println!("m1 {:?}", m1);
        println!("m1 stride:{:?}", m1.stride);
        let m2 = arr(&[1.0, 2.0, 3.0]);

        let m3 = m1 * &m2;
        println!("m3:{:?}", m3);
    }

    #[test]
    fn test_div() {
        let m1 = mat(&[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]);
        println!("m1 dim:{:?}", m1.dim);
        println!("m1 {:?}", m1);
        println!("m1 stride:{:?}", m1.stride);
        let m2 = arr(&[1.0, 2.0, 3.0]);

        let m3 = m1 / &m2;
        println!("m3:{:?}", m3);
    }

    #[test]
    fn test_rem() {
        let m1 = mat(&[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]);
        println!("m1 dim:{:?}", m1.dim);
        println!("m1 {:?}", m1);
        println!("m1 stride:{:?}", m1.stride);
        let m2 = arr(&[1.0, 2.0, 3.0]);

        let m3 = m1 % m2;
        println!("m3:{:?}", m3);
    }
}
