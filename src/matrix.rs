pub type Matrix<S, D> = ArrayBase<S, D>;

pub struct OwnedPtr<P> {
    ptr: *const P,
    len: usize,
    cap: usize,
}

struct ArrayBase<S, D> {
    data: S,
    dim: D,
}

impl<S, D> Matrix<S, D> {
    fn new() {}

    fn zeros() -> Self {}
}
