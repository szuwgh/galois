use super::broadcast::general_broadcasting;
use super::shape::Shape;
use super::{TenRef, Tensor};

pub trait Distance {}

#[macro_export]
macro_rules! dim_max {
    ($d1:expr, $d2:expr) => {
        <$d1 as DimMax<$d2>>::Output
    };
}

fn convert_opsf<A: Clone, B: Clone>(f: impl Fn(A, B) -> A) -> impl FnMut(&mut A, &B) {
    move |x, y| *x = f(x.clone(), y.clone())
}

impl<A> core::ops::Add<TenRef<A>> for TenRef<A>
where
    A: core::ops::Add<A, Output = A> + Clone,
{
    type Output = TenRef<A>;
    fn add(mut self, rhs: TenRef<A>) -> Self::Output {
        self.zip(&rhs, convert_opsf(A::add));
        self
    }
}

impl<A> core::ops::Add<&Tensor<A>> for Tensor<A>
where
    A: core::ops::Add<A, Output = A> + Clone,
{
    type Output = Tensor<A>;
    fn add(self, rhs: &Tensor<A>) -> Self::Output {
        //  general_broadcasting::<_, _, _, [usize; 4]>(&self.as_ref(), &m2.as_ref());
        // let _ = self.as_ref() + rhs.as_ref();
        // self
        todo!()
    }
}

mod tests {
    // use super::super::mat;
    use super::*;
    #[test]
    fn test_add() {
        // let m1 = mat(&[[1.0, 2.0], [3.0, 4.0]]);
        // let m2 = mat(&[[1.0, 2.0], [3.0, 4.0]]);
        // let m3 = m1 + &m2;
        // // println!("m3:{:?}", m1);
        // println!("m3:{:?}", m3);
        // let v = m1.as_slice();
        // let t = m1.as_ref();
        // println!("v:{:?}", m);
    }
}
