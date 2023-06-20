pub struct Axis(pub usize);

impl Axis {
    pub fn index(&self) -> usize {
        self.0
    }
}

pub trait Dimension: Clone {
    fn slice(&self) -> &[usize];

    fn size(&self) -> usize;

    fn dim(&self) -> usize;

    fn strides(&self) -> Self;

    fn zero() -> Self;

    type AxisDimension: Dimension;
    fn select_axis(&self, a: Axis) -> Self::AxisDimension;
}

impl Dimension for [usize; 1] {
    type AxisDimension = [usize; 1];
    fn select_axis(&self, a: Axis) -> Self::AxisDimension {
        [0; 1]
    }

    fn slice(&self) -> &[usize] {
        self
    }

    fn size(&self) -> usize {
        self.iter().fold(1, |s, &a| s * a as usize)
    }

    // [a, b, c] => strides [b * c, c, 1]
    fn strides(&self) -> [usize; 1] {
        let mut x = Self::zero();
        let s = self.slice().iter().rev();
        let mut prod = 1;
        let mut temp = 1;
        for (m, dim) in x.iter_mut().rev().zip(s) {
            prod *= temp;
            *m = prod;
            temp = *dim;
        }
        x
    }

    fn zero() -> Self {
        [0; 1]
    }

    fn dim(&self) -> usize {
        self.len()
    }
}

// macro_rules! select_axis_impl_array {
//     ($n:expr) => {
//         impl SelectAxis for [usize; $n] {
//             type AxisDimension = [usize; $n - 1];
//             fn select_axis(&self, a: Axis) -> Self::AxisDimension {
//                 let mut dst = [0; $n - 1];
//                 let src = self.slice();
//                 dst[..a.index()].copy_from_slice(&src[..a.index()]);
//                 dst[a.index()..].copy_from_slice(&src[a.index() + 1..]);
//                 dst
//             }
//         }
//     };
// }

macro_rules! dimension_impl_array {
    ($n:expr) => {
        impl Dimension for [usize; $n] {
            type AxisDimension = [usize; $n - 1];
            fn select_axis(&self, a: Axis) -> Self::AxisDimension {
                let mut dst = [0; $n - 1];
                let src = self.slice();
                dst[..a.index()].copy_from_slice(&src[..a.index()]);
                dst[a.index()..].copy_from_slice(&src[a.index() + 1..]);
                dst
            }

            fn slice(&self) -> &[usize] {
                self
            }

            fn size(&self) -> usize {
                self.iter().fold(1, |s, &a| s * a as usize)
            }

            // [a, b, c] => strides [b * c, c, 1]
            fn strides(&self) -> [usize; $n] {
                let mut x = Self::zero();
                let s = self.slice().iter().rev();
                let mut prod = 1;
                let mut temp = 1;
                for (m, dim) in x.iter_mut().rev().zip(s) {
                    prod *= temp;
                    *m = prod;
                    temp = *dim;
                }
                x
            }

            fn zero() -> Self {
                [0; $n]
            }

            fn dim(&self) -> usize {
                self.len()
            }
        }
    };
}

// select_axis_impl_array!(2);
// select_axis_impl_array!(3);
// select_axis_impl_array!(4);
// select_axis_impl_array!(5);
// select_axis_impl_array!(6);
// select_axis_impl_array!(7);
// select_axis_impl_array!(8);
// select_axis_impl_array!(9);
// select_axis_impl_array!(10);
// select_axis_impl_array!(11);

//dimension_impl_array!(1);
dimension_impl_array!(2);
dimension_impl_array!(3);
dimension_impl_array!(4);
dimension_impl_array!(5);
dimension_impl_array!(6);
dimension_impl_array!(7);
dimension_impl_array!(8);
dimension_impl_array!(9);
dimension_impl_array!(10);
dimension_impl_array!(11);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_axis() {
        let d = [4usize, 3, 2];
        let d2 = d.select_axis(Axis(0));
        println!("{:?}", d2);
    }
}
