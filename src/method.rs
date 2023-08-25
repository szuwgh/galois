use super::broadcast::general_broadcasting;
use super::dimension::DynDim;
use super::{Dimension, TenRef, TensorData};

pub trait Distance {}

#[macro_export]
macro_rules! dim_max {
    ($d1:expr, $d2:expr) => {
        <$d1 as DimMax<$d2>>::Output
    };
}

pub trait DimMax<Other: Dimension> {
    type Output: Dimension;
}

impl<D: Dimension> DimMax<D> for D {
    type Output = D;
}

fn convert_opsf<A: Clone, B: Clone>(f: impl Fn(A, B) -> A) -> impl FnMut(&mut A, &B) {
    move |x, y| *x = f(x.clone(), y.clone())
}

impl<A, D> core::ops::Add<TenRef<A, D>> for TenRef<A, D>
where
    A: core::ops::Add<A, Output = A> + Clone,
    D: Dimension,
{
    type Output = TenRef<A, D>;
    fn add(mut self, rhs: TenRef<A, D>) -> Self::Output {
        self.zip(&rhs, convert_opsf(A::add));
        self
    }
}

impl<A, D1, D2> core::ops::Add<&TensorData<A, D2>> for TensorData<A, D1>
where
    A: core::ops::Add<A, Output = A> + Clone,
    D1: Dimension + DimMax<D2>,
    D2: Dimension,
{
    type Output = TensorData<A, <D1 as DimMax<D2>>::Output>;
    fn add(self, rhs: &TensorData<A, D2>) -> Self::Output {
        //  general_broadcasting::<_, _, _, [usize; 4]>(&self.as_ref(), &m2.as_ref());
        // let _ = self.as_ref() + rhs.as_ref();
        // self
        todo!()
    }
}

impl<A, D1, D2> core::ops::Add<&TenRef<A, D2>> for TensorData<A, D1>
where
    A: core::ops::Add<A, Output = A> + Clone,
    D1: Dimension + DimMax<D2>,
    D2: Dimension,
{
    type Output = TensorData<A, <D1 as DimMax<D2>>::Output>;
    fn add(self, rhs: &TenRef<A, D2>) -> Self::Output {
        //直接计算
        if self.dim.as_slice() == rhs.dim.as_slice() {
        } else {
            // let (t3, t4) =
            //     general_broadcasting::<_, _, _, <D1 as DimMax<D2>>::Output>(&self.as_ref(), rhs);
        }

        todo!()
    }
}

impl<A, D> core::ops::Add<A> for TenRef<A, D>
where
    D: Dimension,
{
    type Output = TenRef<A, D>;
    fn add(self, rhs: A) -> Self::Output {
        todo!();
        // if self.dim() == rhs.dim() && self.shape() == rhs.shape() {
        // Self::Output::with_dim()
        //}
    }
}

impl<A, D> core::ops::Sub<TenRef<A, D>> for TenRef<A, D>
where
    D: Dimension,
{
    type Output = TenRef<A, D>;
    fn sub(self, rhs: TenRef<A, D>) -> Self::Output {
        // if self.dim() == rhs.dim() && self.shape() == rhs.shape() {
        //     Self::Output::with_dim()
        // }
        todo!();
    }
}

impl<A, D> core::ops::Sub<A> for TenRef<A, D>
where
    D: Dimension,
{
    type Output = TenRef<A, D>;
    fn sub(self, rhs: A) -> Self::Output {
        todo!();
        // if self.dim() == rhs.dim() && self.shape() == rhs.shape() {
        // Self::Output::with_dim()
        //}
    }
}

macro_rules! impl_broadcast_distinct_fixed {
    ($smaller:ty, $larger:ty) => {
        impl DimMax<$larger> for $smaller {
            type Output = $larger;
        }

        impl DimMax<$smaller> for $larger {
            type Output = $larger;
        }
    };
}

impl_broadcast_distinct_fixed!([usize; 1], [usize; 2]);
impl_broadcast_distinct_fixed!([usize; 1], [usize; 3]);
impl_broadcast_distinct_fixed!([usize; 1], [usize; 4]);
impl_broadcast_distinct_fixed!([usize; 1], [usize; 5]);
impl_broadcast_distinct_fixed!([usize; 1], DynDim);
impl_broadcast_distinct_fixed!([usize; 2], DynDim);
impl_broadcast_distinct_fixed!([usize; 3], DynDim);
impl_broadcast_distinct_fixed!([usize; 4], DynDim);
impl_broadcast_distinct_fixed!([usize; 5], DynDim);
impl_broadcast_distinct_fixed!([usize; 6], DynDim);
impl_broadcast_distinct_fixed!([usize; 7], DynDim);
mod tests {
    use super::super::mat;
    use super::*;
    #[test]
    fn test_add() {
        let m1 = mat(&[[1.0, 2.0], [3.0, 4.0]]);
        let m2 = mat(&[[1.0, 2.0], [3.0, 4.0]]);
        let m3 = m1 + &m2;
        // println!("m3:{:?}", m1);
        println!("m3:{:?}", m3);
        // let v = m1.as_slice();
        // let t = m1.as_ref();
        // println!("v:{:?}", m);
    }
}
