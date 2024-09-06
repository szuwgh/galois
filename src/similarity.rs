use crate::CpuDevice;
use crate::Device;
use crate::GResult;
use crate::Tensor;
use crate::TensorType;
use std::ops::Sub;

trait UnaryOp {
    fn op_powi(&self, n: i32) -> Self;
    fn to_f32(&self) -> f32;
}
trait BinaryOp {
    fn sub(&self, other: &Self) -> Self;
}

impl UnaryOp for f32 {
    fn op_powi(&self, n: i32) -> Self {
        self.powi(n)
    }

    fn to_f32(&self) -> f32 {
        *self
    }
}

impl BinaryOp for f32 {
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
}

pub trait Similarity {
    // 欧式距离/欧几里得距离
    fn euclidean(&self, other: &Self) -> f32;
    // 曼哈顿距离
    fn manhattan(&self) -> f32;
    // 余弦相似度
    fn cosine(&self) -> f32;
    // 切比雪夫距离
    fn chebyshev(&self) -> f32;
}

trait Map {
    fn f<T: BinaryOp + UnaryOp>(&self, left: &[T], right: &[T]) -> f32;

    fn map(&self, left: &CpuDevice, right: &CpuDevice) -> f32 {
        match (left, right) {
            // (CpuDevice::F16(v1), CpuDevice::F16(d)) => self.f(v1.as_slice(), d.as_slice()),
            (CpuDevice::F32(v1), CpuDevice::F32(d)) => self.f(v1.as_slice(), d.as_slice()),
            _ => {
                todo!()
            }
        }
    }
}

struct Euclidean;

impl Map for Euclidean {
    fn f<T: BinaryOp + UnaryOp>(&self, left: &[T], right: &[T]) -> f32 {
        left.iter()
            .zip(right.iter())
            .map(|(x, y)| x.sub(y).op_powi(2).to_f32())
            .sum::<f32>()
            .sqrt()
    }
}

impl Similarity for Tensor {
    fn euclidean(&self, other: &Self) -> f32 {
        match (self.device(), other.device()) {
            (Device::Cpu(a), Device::Cpu(b)) => Euclidean.map(a, b),
            _ => {
                todo!()
            }
        }
    }

    fn manhattan(&self) -> f32 {
        todo!()
    }

    fn cosine(&self) -> f32 {
        todo!()
    }

    fn chebyshev(&self) -> f32 {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::Shape;

    use super::*;

    #[test]
    fn test_euclidean() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 1, Shape::from_array([3]));
        println!("a{:?}", a);
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], 1, Shape::from_array([3]));
        println!("a{:?}", b);
        let d = a.euclidean(&b);
        println!("d:{}", d)
    }
}
