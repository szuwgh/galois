extern "C" __global__ void silu_f32(const float *x, float *dst, const int k)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k)
    {
        return;
    }
    dst[i] = x[i] / (1.0f + expf(-x[i]));
}
