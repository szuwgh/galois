use super::{BaseData, TensorData};

pub trait Add<Rhs = Self> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
}

// impl<A, S, D> Add<ArrayData<S, D>> for ArrayData<S, D>
// where
//     S: BaseData<Elem = A>,
//     D: Dimension,
// {
//     type Output = ArrayData<S, D>;
//     fn add(self, rhs: ArrayData<S, D>) -> Self::Output {
//         // if self.dim() == rhs.dim() && self.shape() == rhs.shape() {
//         // Self::Output::with_dim()
//         //}
//     }
// }
