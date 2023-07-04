use super::{Dimension, TenRef, TensorData};

impl<A, D> core::ops::Add<TenRef<A, D>> for TenRef<A, D>
where
    D: Dimension,
{
    type Output = TensorData<A, D>;
    fn add(self, rhs: TenRef<A, D>) -> Self::Output {
        todo!();
        // if self.dim() == rhs.dim() && self.shape() == rhs.shape() {
        // Self::Output::with_dim()
        //}
    }
}

impl<A, D> core::ops::Add<&TenRef<A, D>> for TenRef<A, D>
where
    D: Dimension,
{
    type Output = TensorData<A, D>;
    fn add(self, rhs: &TenRef<A, D>) -> Self::Output {
        //   let c = TensorData::with_dim(D);
        todo!();
        // if self.dim() == rhs.dim() && self.shape() == rhs.shape() {
        // Self::Output::with_dim()
        //}
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
