// the CUDA soft max implementation differs from the CPU implementation
// instead of doubles floats are used
extern "C" __global__ void soft_max_f32(const float *x, float *dst, const int ncols)
{
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;

    float max_val = -INFINITY;

    for (int col = tid; col < ncols; col += block_size)
    {
        const int i = row * ncols + col;
        max_val = max(max_val, x[i]);
    }

    // find the max value in the block
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, mask, 32));
    }

    float tmp = 0.f;

    for (int col = tid; col < ncols; col += block_size)
    {
        const int i = row * ncols + col;
        const float val = expf(x[i] - max_val);
        tmp += val;
        dst[i] = val;
    }

    // sum up partial sums
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    const float inv_tmp = 1.f / tmp;

    for (int col = tid; col < ncols; col += block_size)
    {
        const int i = row * ncols + col;
        dst[i] *= inv_tmp;
    }
}
