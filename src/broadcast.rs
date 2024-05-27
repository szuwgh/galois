use super::error::{GError, ShapeErrorKind};
use crate::shape::Dim;
use crate::TensorType;
use crate::{error::GResult, shape::Shape, Tensor};
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

pub(crate) fn upcast(to: &Shape, from: &Shape, stride: &[usize]) -> Option<Box<[usize]>> {
    let mut new_stride = to.0.clone();

    if to.dim() < from.dim() {
        return None;
    }
    {
        let mut new_stride_iter = new_stride.iter_mut().rev();
        for ((er, es), dr) in from
            .as_slice()
            .iter()
            .rev()
            .zip(stride.iter().rev())
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

pub fn broadcasting_binary_op<A>(t1: &Shape, t2: &Shape) -> GResult<Shape> {
    let (d1, d2) = (t1.dim(), t2.dim());
    let k = if d1 > d2 { d1 - d2 } else { d2 - d1 };
    let slice1 = t1.as_slice();
    let slice2 = t2.as_slice();
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
    Ok(output)
}

pub fn broadcasting_matmul_op<A>(lhs: &Shape, rhs: &Shape) -> GResult<(Shape, Shape)> {
    let lhs_dims = lhs.as_slice();
    let rhs_dims = rhs.as_slice();
    if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
        panic!("only 2d matrixes are supported {lhs:?} {rhs:?}")
    }
    let (m, lhs_k) = (lhs_dims[lhs_dims.len() - 2], lhs_dims[lhs_dims.len() - 1]);
    let (rhs_k, n) = (rhs_dims[rhs_dims.len() - 2], rhs_dims[rhs_dims.len() - 1]);
    if lhs_k != rhs_k {
        panic!("different inner dimensions in broadcast matmul {lhs:?} {rhs:?}")
    }

    let lhs_b = Shape::from_slice(&lhs_dims[..lhs_dims.len() - 2]);
    let rhs_b = Shape::from_slice(&rhs_dims[..rhs_dims.len() - 2]);
    let bcast = broadcasting_binary_op::<A>(&lhs_b, &rhs_b)?;
    let bcast_dims = bcast.as_slice();

    let bcast_lhs = [bcast_dims, &[m, lhs_k]].concat();
    let bcast_rhs = [bcast_dims, &[rhs_k, n]].concat();
    Ok((Shape::from_vec(bcast_lhs), Shape::from_vec(bcast_rhs)))
}

// 广播主要发生在两种情况，一种情况是如果两个张量的维数不相等，但是它们的后缘维度的轴长相符。所谓后缘维度（trailing dimension）是指，
// 从末尾开始算起的维度。另外一种情况是，如果两个张量的后缘维度不同，则有一方的长度为1
// https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
pub fn general_broadcasting<A>(
    t1: &Tensor<A>,
    t2: &Tensor<A>,
) -> GResult<(Tensor<A>, Tensor<A>)>
where
    A: TensorType,
{
    let (d1, d2) = (t1.size(), t2.size());
    let k = if d1 > d2 { d1 - d2 } else { d2 - d1 };
    let slice1 = t1.dim.shape().as_slice();
    let slice2 = t2.dim.shape().as_slice();
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
    let broadcast_strides3 = match upcast(&output, &t1.dim.shape(), &t1.dim.stride()) {
        Some(st) => st,
        None => return Err(GError::ShapeError(ShapeErrorKind::IncompatibleShape)),
    };

    let broadcast_strides4 = match upcast(&output, &t2.dim.shape(), &t2.dim.stride) {
        Some(st) => st,
        None => return Err(GError::ShapeError(ShapeErrorKind::IncompatibleShape)),
    };

    Ok((
        Tensor {
            data: t1.data.as_ref(),
            dim: Dim {
                s: output.clone(),
                stride: broadcast_strides3,
            },
        },
        Tensor {
            data: t2.data.as_ref(),
            dim: Dim {
                s: output.clone(),
                stride: broadcast_strides4,
            },
        },
    ))
}

mod tests {
    use crate::Tensor;

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
        println!("m1 stride:{:?}", m1.dim.stride);
        let m2 = arr(&[1.0, 2.0, 3.0]);
        println!("m2 dim:{:?}", m2.dim);
        println!("m2 {:?}", m2);
        println!("m2 stride:{:?}", m2.dim.stride);
        let (out1, out2) = general_broadcasting::<_>(&m1.view(), &m2.view()).unwrap();

        println!("out2:{:?}", out2);
        println!("stride:{:?}", out2.dim.stride);

        for i in out2.iter() {
            println!("i:{}", *i);
        }
    }
}
