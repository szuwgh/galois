use super::{Dimension, TenRef};

pub trait Add<Rhs = Self> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
}

impl<A, D> Add<TenRef<A, D>> for TenRef<A, D>
where
    D: Dimension,
{
    type Output = TenRef<A, D>;
    fn add(self, rhs: TenRef<A, D>) -> Self::Output {
        todo!();
        // if self.dim() == rhs.dim() && self.shape() == rhs.shape() {
        // Self::Output::with_dim()
        //}
    }
}
