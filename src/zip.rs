use crate::{Shape, TensorItem, TensorType};
use crate::{Tensor, TensorIter};

pub struct Zip<'a> {
    a: TensorIter<'a>,
    b: TensorIter<'a>,
}

impl<'a> Zip<'a> {
    pub fn new(a: TensorIter<'a>, b: TensorIter<'a>) -> Zip<'a> {
        Self { a: a, b: b }
    }

    pub fn ops<F>(self, mut f: F)
    where
        F: FnMut((TensorItem<'a>, TensorItem<'a>)),
    {
        for t in self.into_iter() {
            f(t);
        }
    }

    pub fn map<F>(self, f: F) -> Map<Self, F>
    where
        F: FnMut((TensorItem<'a>, TensorItem<'a>)) -> TensorItem<'a>,
    {
        Map::new(self, f)
    }
}

impl<'a> Iterator for Zip<'a> {
    type Item = (TensorItem<'a>, TensorItem<'a>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let x = self.a.next()?;
        let y = self.b.next()?;
        Some((x, y))
    }
}

pub struct Map<I: Iterator, F> {
    pub(crate) iter: I,
    f: F,
}

impl<B, I: Iterator, F> Map<I, F>
where
    F: FnMut(I::Item) -> B,
{
    pub fn new(i: I, f: F) -> Map<I, F> {
        Self { iter: i, f: f }
    }
    // pub fn collect_tensor(self) -> Vec<B> {
    //   let v = self.collect();
    //   //  Tensor::from_vec(v, dim)
    // }
}

impl<B, I: Iterator, F> Iterator for Map<I, F>
where
    F: FnMut(I::Item) -> B,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.next().map(&mut self.f)
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test] //[3, 2]
    fn test_select_axis() {
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 2, 3, 4];
        //   let c: Vec<i32> = a.iter().zip(b.iter()).map(f)

        // a.iter().zip(b.iter()).for_each(|(t1, t2)| {
        //     c = *t1 + *t2;
        // });

        //println!("c:{:?}", c);
    }
    use crate::mat;
    #[test] //[3, 2]
    fn test_zip() {
        let m1 = mat(&[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]);
        let m2 = mat(&[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]);

        // m1.view()
        //     .iter()
        //     .for_each(|x| if *x > 1.0 { *x = 1.0 } else { *x = 0.0 });

        println!("m3:{:?}", m1);
    }
}
