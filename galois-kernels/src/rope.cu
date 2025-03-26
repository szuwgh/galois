#include "iostream"

extern "C" __global__ void rope_f32(const float *x, float *dst, const int ncols, const int32_t *pos, const float freq_scale,
                                    const int p_delta_rows, const float theta_scale, bool has_pos)
{
    const int col = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (col >= ncols)
    {
        return;
    }

    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int i = row * ncols + col;
    const int i2 = row / p_delta_rows;

    const int p = has_pos ? pos[i2] : 0;
    const float p0 = p * freq_scale;
    const float theta = p0 * powf(theta_scale, col / 2);
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + 1];

    dst[i + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[i + 1] = x0 * sin_theta + x1 * cos_theta;
}