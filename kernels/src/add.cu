#include <cuda_fp16.h>

extern "C" __global__ void add_f32(const float *x, const float *y, float *dst, const int kx, const int ky)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= kx)
    {
        return;
    }
    dst[i] = x[i] + y[i % ky];
}

extern "C" __global__ void add_f16_f32_f16(const half *x, const float *y, half *dst, const int k)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k)
    {
        return;
    }
    dst[i] = __hadd(x[i], __float2half(y[i]));
}