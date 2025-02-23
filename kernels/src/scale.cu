extern "C" __global__ void scale_f32(const float *x, float *dst, const float scale, const int k)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k)
    {
        return;
    }

    dst[i] = scale * x[i];
}
