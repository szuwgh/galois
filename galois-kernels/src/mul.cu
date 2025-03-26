#define CUDA_MUL_BLOCK_SIZE 256

extern "C" __global__ void mul_f32(const float *x, const float *y, float *dst, const int kx, const int ky)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= kx)
    {
        return;
    }
    dst[i] = x[i] * y[i % ky];
}
