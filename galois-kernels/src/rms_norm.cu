#define WARP_SIZE 32

static __device__ __forceinline__ float warp_reduce_sum(float x)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

extern "C" __global__ void rms_norm_f32(const float *x, float *dst, const int ncols, const float eps, int block_size)
{
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size)
    {
        const float xi = x[row * ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE)
    {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0)
        {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size)
    {
        dst[row * ncols + col] = scale * x[row * ncols + col];
    }
}
