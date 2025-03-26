use crate::CudaDevice;
use crate::GResult;

pub fn init_cuda_function(dev: &CudaDevice) -> GResult<()> {
    dev.load_ptx("dequantize_block_f16_to_f32", galois_kernels::MATMUL)?;
    dev.load_ptx("dequantize_block_f32_to_f16", galois_kernels::MATMUL)?;
    dev.load_ptx("mul_mat_p021_f16_f32", galois_kernels::MATMUL)?;
    dev.load_ptx("dequantize_mul_mat_vec_q4_0", galois_kernels::MATMUL)?;
    dev.load_ptx("rms_norm_f32", galois_kernels::RMS_NORM)?;
    dev.load_ptx("mul_f32", galois_kernels::MUL)?;
    dev.load_ptx("quantize_q8_1", galois_kernels::MATMUL)?;
    dev.load_ptx("mul_mat_q4_0", galois_kernels::MATMUL)?;
    dev.load_ptx("rope_f32", galois_kernels::ROPE)?;
    dev.load_ptx("cpy_f32_f16", galois_kernels::CPY)?;
    dev.load_ptx("cpy_f32_f32", galois_kernels::CPY)?;
    dev.load_ptx("scale_f32", galois_kernels::SCALE)?;
    dev.load_ptx("add_f32", galois_kernels::ADD)?;
    dev.load_ptx("add_f32", galois_kernels::ADD)?;
    dev.load_ptx("soft_max_f32", galois_kernels::SOFTMAX)?;
    dev.load_ptx("silu_f32", galois_kernels::UNARY)?;
    Ok(())
}
