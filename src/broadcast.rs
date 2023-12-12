use super::error::{GError, ShapeErrorKind};
use crate::{error::GResult, shape::Shape, DTensor};

#[macro_export]
macro_rules! copy {
    ($des:expr, $src:expr) => {
        copy_slice($des, $src)
    };
}

fn copy_slice<T: Copy>(des: &mut [T], src: &[T]) -> usize {
    let l = if des.len() < src.len() {
        des.len()
    } else {
        src.len()
    };
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), des.as_mut_ptr(), l);
    }
    l
}

fn upcast(to: &Shape, from: &Shape, stride: &Shape) -> Option<Shape> {
    let mut new_stride = to.clone();

    if to.dim() < from.dim() {
        return None;
    }

    {
        let mut new_stride_iter = new_stride.as_slice_mut().iter_mut().rev();
        for ((er, es), dr) in from
            .as_slice()
            .iter()
            .rev()
            .zip(stride.as_slice().iter().rev())
            .zip(new_stride_iter.by_ref())
        {
            if *dr == *er {
                *dr = *es;
            } else if *er == 1 {
                *dr = 0
            } else {
                return None;
            }
        }

        for dr in new_stride_iter {
            *dr = 0;
        }
    }
    Some(new_stride)
}

// 广播主要发生在两种情况，一种情况是如果两个张量的维数不相等，但是它们的后缘维度的轴长相符。所谓后缘维度（trailing dimension）是指，
// 从末尾开始算起的维度。另外一种情况是，如果两个张量的后缘维度不同，则有一方的长度为1
// https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
pub fn general_broadcasting<A>(
    t1: &DTensor<A>,
    t2: &DTensor<A>,
) -> GResult<(DTensor<A>, DTensor<A>)> {
    let (d1, d2) = (t1.dim(), t2.dim());
    let k = if d1 > d2 { d1 - d2 } else { d2 - d1 };
    let slice1 = t1.dim.as_slice();
    let slice2 = t2.dim.as_slice();
    let mut output = Shape::zero(slice1.len());
    let output_slice = output.as_slice_mut();
    if copy!(output_slice, slice1) != slice1.len() {
        panic!("copy dimension error");
    }
    for (out, s2) in &mut output_slice[k..].iter_mut().zip(slice2.iter()) {
        if *out != *s2 {
            if *out == 1 {
                *out = *s2
            } else if *s2 != 1 {
                return Err(GError::ShapeError(ShapeErrorKind::IncompatibleShape));
            }
        }
    }
    let broadcast_strides3 = match upcast(&output, &t1.dim, &t1.stride) {
        Some(st) => st,
        None => return Err(GError::ShapeError(ShapeErrorKind::IncompatibleShape)),
    };

    let broadcast_strides4 = match upcast(&output, &t2.dim, &t2.stride) {
        Some(st) => st,
        None => return Err(GError::ShapeError(ShapeErrorKind::IncompatibleShape)),
    };

    Ok((
        DTensor {
            data: t1.data.as_ref(),
            dim: output.clone(),
            stride: broadcast_strides3,
        },
        DTensor {
            data: t2.data.as_ref(),
            dim: output.clone(),
            stride: broadcast_strides4,
        },
    ))
}

mod tests {
    use crate::DTensor;

    use super::super::{arr, cube, mat};
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
        println!("m2 dim:{:?}", m2.dim);
        println!("m2 {:?}", m2);
        println!("m2 stride:{:?}", m2.stride);
        let (out1, out2) = general_broadcasting::<_>(&m1.as_ref(), &m2.as_ref()).unwrap();

        println!("out2:{:?}", out2);
        println!("stride:{:?}", out2.stride);

        for i in out2.iter() {
            println!("i:{}", *i);
        }
    }
}
