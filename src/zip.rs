use crate::{dimension::Dimension, TensorData, TensorIter};

pub struct Zip<'a, A, D: Dimension> {
    a: TensorIter<'a, A, D>,
    b: TensorIter<'a, A, D>,
}

impl<'a, A, D: Dimension> Zip<'a, A, D> {
    fn map<F>(f: F)
    where
        F: FnMut(&A, &A) -> A,
    {
        //  for
    }

    // fn collect() -> TensorData<A, D> {}
}

pub struct Map<I, F> {
    pub(crate) iter: I,
    f: F,
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test] //[3, 2]
    fn test_select_axis() {
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 2, 3, 4];
        let c: Vec<i32> = a.iter().zip(b.iter()).map(|(t1, t2)| *t1 + *t2).collect();

        // a.iter().zip(b.iter()).for_each(|(t1, t2)| {
        //     c = *t1 + *t2;
        // });

        println!("c:{:?}", c);
    }
}
