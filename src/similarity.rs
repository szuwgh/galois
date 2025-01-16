use crate::CpuStorageSlice;
use crate::GResult;
use crate::Storage;
use crate::Tensor;
use crate::TensorProto;
use crate::TensorType;
use core::arch::x86_64::*;
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

    fn f32(&self, left: &[f32], right: &[f32]) -> f32;

    fn map(&self, left: &CpuStorageSlice, right: &CpuStorageSlice) -> f32 {
        match (left, right) {
            // (CpuStorageSlice::F16(v1), CpuStorageSlice::F16(d)) => self.f(v1.as_slice(), d.as_slice()),
            (CpuStorageSlice::F32(v1), CpuStorageSlice::F32(d)) => {
                self.f32(v1.as_slice(), d.as_slice())
            }
            _ => {
                todo!()
            }
        }
    }
}

struct Euclidean;

impl Map for Euclidean {
    #[cfg(not(target_feature = "avx2"))]
    fn f32(&self, left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(right.iter()) // 将两个切片的元素配对
            .map(|(l, r)| (l - r).powi(2)) // 计算差值的平方
            .sum::<f32>() // 求和
            .sqrt() // 计算平方根
    }

    #[cfg(target_feature = "avx2")]
    fn f32(&self, left: &[f32], right: &[f32]) -> f32 {
        unsafe {
            let len = left.len();
            let mut i = 0;
            // 初始化一个 AVX 向量用于存储平方和
            let mut sum = _mm256_setzero_ps();
            while i + 8 <= len {
                let left_chunk = _mm256_loadu_ps(left.as_ptr().add(i));
                let right_chunk = _mm256_loadu_ps(right.as_ptr().add(i));

                let diff = _mm256_sub_ps(left_chunk, right_chunk);

                let diff_squared = _mm256_mul_ps(diff, diff);

                sum = _mm256_add_ps(sum, diff_squared);

                i += 8;
            }
            let mut tail_sum = 0.0;
            while i < len {
                let diff = left[i] - right[i];
                tail_sum += diff * diff;
                i += 1;
            }
            let mut sum_array = [0.0; 8];
            _mm256_storeu_ps(sum_array.as_mut_ptr(), sum);
            let simd_sum: f32 = sum_array.iter().sum();
            (simd_sum + tail_sum).sqrt()
        }
    }

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
        match (self.storage(), other.storage()) {
            (Storage::Cpu(a), Storage::Cpu(b)) => Euclidean.map(a, b),
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
    use super::*;
    use crate::Device;
    use crate::Shape;

    #[test]
    fn test_euclidean() {
        let a = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 1.0f32, 2.0, 3.0, 1.0f32, 2.0, 3.0, 1.0f32, 2.0, 3.0,
            ],
            1,
            Shape::from_array([12]),
            &Device::Cpu,
        )
        .unwrap();
        //println!("a{:?}", a);
        let b = Tensor::from_vec(
            vec![
                4.0f32, 5.0, 6.0, 4.0f32, 5.0, 6.0, 4.0f32, 5.0, 6.0, 4.0f32, 5.0, 6.0,
            ],
            1,
            Shape::from_array([12]),
            &Device::Cpu,
        )
        .unwrap();
        // println!("a{:?}", b);
        let d = a.euclidean(&b);
        println!("d:{}", d)
    }
}
