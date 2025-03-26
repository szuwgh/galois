#include <cuda_fp16.h>
#include <stdint.h>

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
#define GGML_CUDA_DMMV_X 32
#define WARP_SIZE 32

#if CUDART_VERSION >= 11100
#define GGML_CUDA_ASSUME(x) __builtin_assume(x)
#else
#define GGML_CUDA_ASSUME(x)
#endif // CUDART_VERSION >= 11100

typedef float dfloat;
typedef float2 dfloat2;

typedef struct
{
    half d;                // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

static __device__ __forceinline__ void dequantize_q4_0(const void *vx, const int ib, const int iqs, dfloat2 &v)
{
    const block_q4_0 *x = (const block_q4_0 *)vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {8.0f, 8.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
#endif // GGML_CUDA_F16
}

extern "C" __global__ void dequantize_mul_mat_vec_q4_0(const void *__restrict__ vx, const dfloat *__restrict__ y, float *__restrict__ dst, const int ncols, const int nrows)
{
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    int qr = QR4_0;
    int qk = QK4_0;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= nrows)
    {
        return;
    }

    const int tid = threadIdx.x;

    const int iter_stride = 2 * GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk / 2;

// partial sum for each thread
#ifdef GGML_CUDA_F16
    half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#else
    float tmp = 0.0f;
#endif // GGML_CUDA_F16

    for (int i = 0; i < ncols; i += iter_stride)
    {
        const int col = i + vals_per_iter * tid;
        const int ib = (row * ncols + col) / qk; // x block index
        const int iqs = (col % qk) / qr;         // x quant index
        const int iybs = col - col % qk;         // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2)
        {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_q4_0(vx, ib, iqs + j / qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_CUDA_F16
            tmp += __hmul2(v, {y[iybs + iqs + j / qr + 0],
                               y[iybs + iqs + j / qr + y_offset]});
#else
            tmp += v.x * y[iybs + iqs + j / qr + 0];
            tmp += v.y * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_CUDA_F16
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0)
    {
#ifdef GGML_CUDA_F16
        dst[row] = tmp.x + tmp.y;
#else
        dst[row] = tmp;
#endif // GGML_CUDA_F16
    }
}

#define VDR_Q4_0_Q8_1_MMQ 4

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
typedef struct
{
    half2 ds;         // ds.x = delta, ds.y = sum
    int8_t qs[QK8_0]; // quants
} block_q8_1;

extern "C" __global__ void quantize_q8_1(const float *__restrict__ x, void *__restrict__ vy, const int kx, const int kx_padded)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;

    if (ix >= kx_padded)
    {
        return;
    }

    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    const int i_padded = iy * kx_padded + ix;

    block_q8_1 *y = (block_q8_1 *)vy;

    const int ib = i_padded / QK8_1;  // block index
    const int iqs = i_padded % QK8_1; // quant index

    const float xi = ix < kx ? x[iy * kx + ix] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0)
    {
        return;
    }

    reinterpret_cast<half &>(y[ib].ds.x) = d;
    reinterpret_cast<half &>(y[ib].ds.y) = sum;
}

typedef void (*allocate_tiles_cuda_t)(int **x_ql, half2 **x_dm, int **x_qh, int **x_sc);
typedef void (*load_tiles_cuda_t)(
    const void *__restrict__ vx, int *__restrict__ x_ql, half2 *__restrict__ x_dm, int *__restrict__ x_qh,
    int *__restrict__ x_sc, const int &i_offset, const int &i_max, const int &k, const int &blocks_per_row);
typedef float (*vec_dot_q_mul_mat_cuda_t)(
    const int *__restrict__ x_ql, const half2 *__restrict__ x_dm, const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const half2 *__restrict__ y_ms, const int &i, const int &j, const int &k);

template <int vdr>
static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int *v, const int *u, const float &d4, const half2 &ds8)
{

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i)
    {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = __dp4a(vi0, u[2 * i + 0], sumi);
        sumi = __dp4a(vi1, u[2 * i + 1], sumi);
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8 * vdr / QI4_0) * ds8f.y);
#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1_mul_mat(
    const int *__restrict__ x_ql, const half2 *__restrict__ x_dm, const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const half2 *__restrict__ y_ds, const int &i, const int &j, const int &k)
{

    const int kyqs = k % (QI8_1 / 2) + QI8_1 * (k / (QI8_1 / 2));
    const float *x_dmf = (float *)x_dm;

    int u[2 * VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l)
    {
        u[2 * l + 0] = y_qs[j * WARP_SIZE + (kyqs + l) % WARP_SIZE];
        u[2 * l + 1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_0) % WARP_SIZE];
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>(&x_ql[i * (WARP_SIZE + 1) + k], u, x_dmf[i * (WARP_SIZE / QI4_0) + i / QI4_0 + k / QI4_0],
                                                     y_ds[j * (WARP_SIZE / QI8_1) + (2 * k / QI8_1) % (WARP_SIZE / QI8_1)]);
}

static __device__ __forceinline__ int get_int_from_int8_aligned(const int8_t *x8, const int &i32)
{
    return *((int *)(x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

template <int qk, int qr, int qi, bool need_sum, typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles, int vdr, vec_dot_q_mul_mat_cuda_t vec_dot>
static __device__ __forceinline__ void mul_mat_q(
    const void *__restrict__ vx, const void *__restrict__ vy, float *__restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst)
{

    const block_q_t *x = (const block_q_t *)vx;
    const block_q8_1 *y = (const block_q8_1 *)vy;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    const int blocks_per_warp = WARP_SIZE / qi;

    const int &ncols_dst = ncols_y;

    const int row_dst_0 = blockIdx.x * mmq_y;
    const int &row_x_0 = row_dst_0;

    const int col_dst_0 = blockIdx.y * mmq_x;
    const int &col_y_0 = col_dst_0;

    int *tile_x_ql = nullptr;
    half2 *tile_x_dm = nullptr;
    int *tile_x_qh = nullptr;
    int *tile_x_sc = nullptr;

    allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

    __shared__ int tile_y_qs[mmq_x * WARP_SIZE];
    __shared__ half2 tile_y_ds[mmq_x * WARP_SIZE / QI8_1];

    float sum[mmq_y / WARP_SIZE][mmq_x / nwarps] = {0.0f};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp)
    {

        load_tiles(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                   threadIdx.y, nrows_x - row_x_0 - 1, threadIdx.x, blocks_per_row_x);

#pragma unroll
        for (int ir = 0; ir < qr; ++ir)
        {
            const int kqs = ir * WARP_SIZE + threadIdx.x;
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps)
            {
                const int col_y_eff = min(col_y_0 + threadIdx.y + i, ncols_y - 1); // to prevent out-of-bounds memory accesses

                const block_q8_1 *by0 = &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) + kbxd];

                const int index_y = (threadIdx.y + i) * WARP_SIZE + kqs % WARP_SIZE;
                tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1)
            {
                const int ids = (ids0 + threadIdx.y * QI8_1 + threadIdx.x / (WARP_SIZE / QI8_1)) % mmq_x;
                const int kby = threadIdx.x % (WARP_SIZE / QI8_1);
                const int col_y_eff = min(col_y_0 + ids, ncols_y - 1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                const half2 *dsi_src = &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) + ir * (WARP_SIZE / QI8_1) + kby].ds;
                half2 *dsi_dst = &tile_y_ds[ids * (WARP_SIZE / QI8_1) + kby];
                if (need_sum)
                {
                    *dsi_dst = *dsi_src;
                }
                else
                {
                    float *dfi_dst = (float *)dsi_dst;
                    *dfi_dst = __low2half(*dsi_src);
                }
            }

            __syncthreads();

            // #pragma unroll // unrolling this loop causes too much register pressure
            for (int k = ir * WARP_SIZE / qr; k < (ir + 1) * WARP_SIZE / qr; k += vdr)
            {
#pragma unroll
                for (int j = 0; j < mmq_x; j += nwarps)
                {
#pragma unroll
                    for (int i = 0; i < mmq_y; i += WARP_SIZE)
                    {
                        sum[i / WARP_SIZE][j / nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs, tile_y_ds,
                            threadIdx.x + i, threadIdx.y + j, k);
                    }
                }
            }

            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps)
    {
        const int col_dst = col_dst_0 + j + threadIdx.y;

        if (col_dst >= ncols_dst)
        {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE)
        {
            const int row_dst = row_dst_0 + threadIdx.x + i;

            if (row_dst >= nrows_dst)
            {
                continue;
            }

            dst[col_dst * nrows_dst + row_dst] = sum[i / WARP_SIZE][j / nwarps];
        }
    }
}

#define MMQ_X_Q4_0_RDNA2 64
#define MMQ_Y_Q4_0_RDNA2 128
#define NWARPS_Q4_0_RDNA2 8
#define MMQ_X_Q4_0_RDNA1 64
#define MMQ_Y_Q4_0_RDNA1 64
#define NWARPS_Q4_0_RDNA1 8
#define MMQ_X_Q4_0_AMPERE 64
#define MMQ_Y_Q4_0_AMPERE 128
#define NWARPS_Q4_0_AMPERE 4
#define MMQ_X_Q4_0_PASCAL 64
#define MMQ_Y_Q4_0_PASCAL 64
#define NWARPS_Q4_0_PASCAL 8

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t *x8, const int &i32)
{
    const uint16_t *x16 = (uint16_t *)(x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] << 0;
    x32 |= x16[1] << 16;

    return x32;
}

template <int mmq_y>
static __device__ __forceinline__ void allocate_tiles_q4_0(int **x_ql, half2 **x_dm, int **x_qh, int **x_sc)
{

    __shared__ int tile_x_qs[mmq_y * (WARP_SIZE) + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE / QI4_0) + mmq_y / QI4_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *)tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles_q4_0(
    const void *__restrict__ vx, int *__restrict__ x_ql, half2 *__restrict__ x_dm, int *__restrict__ x_qh,
    int *__restrict__ x_sc, const int &i_offset, const int &i_max, const int &k, const int &blocks_per_row)
{

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset < nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k < WARP_SIZE);

    const int kbx = k / QI4_0;
    const int kqsx = k % QI4_0;

    const block_q4_0 *bx0 = (block_q4_0 *)vx;

    float *x_dmf = (float *)x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps)
    {
        int i = i0 + i_offset;

        if (need_check)
        {
            i = min(i, i_max);
        }

        const block_q4_0 *bxi = bx0 + i * blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0)
    {
        int i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;

        if (need_check)
        {
            i = min(i, i_max);
        }

        const block_q4_0 *bxi = bx0 + i * blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE / QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

extern "C" __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
__launch_bounds__(WARP_SIZE *NWARPS_Q4_0_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    mul_mat_q4_0(
        const void *__restrict__ vx, const void *__restrict__ vy, float *__restrict__ dst,
        const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst, bool need_check)
{

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x = MMQ_X_Q4_0_RDNA2;
    const int mmq_y = MMQ_Y_Q4_0_RDNA2;
    const int nwarps = NWARPS_Q4_0_RDNA2;
#else
    const int mmq_x = MMQ_X_Q4_0_RDNA1;
    const int mmq_y = MMQ_Y_Q4_0_RDNA1;
    const int nwarps = NWARPS_Q4_0_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
              load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x = MMQ_X_Q4_0_AMPERE;
    const int mmq_y = MMQ_Y_Q4_0_AMPERE;
    const int nwarps = NWARPS_Q4_0_AMPERE;

    if (need_check)
    {
        mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
                  load_tiles_q4_0<mmq_y, nwarps, true>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
    else
    {
        mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
                  load_tiles_q4_0<mmq_y, nwarps, false>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x = MMQ_X_Q4_0_PASCAL;
    const int mmq_y = MMQ_Y_Q4_0_PASCAL;
    const int nwarps = NWARPS_Q4_0_PASCAL;

    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
              load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    (void)vec_dot_q4_0_q8_1_mul_mat;
    assert(false);
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

extern "C" __global__ void mul_mat_p021_f16_f32(
    const void *__restrict__ vx, const float *__restrict__ y, float *__restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y)
{

    const half *x = (const half *)vx;

    const int row_x = blockDim.y * blockIdx.y + threadIdx.y;
    const int channel = blockDim.z * blockIdx.z + threadIdx.z;
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x)
    {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x)
        {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x * nchannels_x * ncols_x + channel_x * ncols_x + col_x;
        const float xi = __half2float(x[ix]);

        const int row_y = col_x;

        // y is not transposed but permuted
        const int iy = channel * nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel * nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0)
    {
        dst[idst] = tmp;
    }
}

static __device__ void convert_f32(const void *vx, const int ib, const int iqs, dfloat2 &v)
{
    const float *x = (const float *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x = x[ib + iqs + 0];
    v.y = x[ib + iqs + 1];
}

static __device__ void convert_f16(const void *vx, const int ib, const int iqs, dfloat2 &v)
{
    const half *x = (const half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x = x[ib + iqs + 0];
    v.y = x[ib + iqs + 1];
}

extern "C" __global__ void dequantize_block_f32_to_f16(const void *__restrict__ vx, half *__restrict__ y, const int k, int qk, int qr)
{
    const int i = blockDim.x * blockIdx.x + 2 * threadIdx.x;

    if (i >= k)
    {
        return;
    }

    const int ib = i / qk;         // block index
    const int iqs = (i % qk) / qr; // quant index
    const int iybs = i - i % qk;   // y block start index
    const int y_offset = qr == 1 ? 1 : qk / 2;

    // dequantize
    dfloat2 v;
    convert_f32(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x;
    y[iybs + iqs + y_offset] = v.y;
}

extern "C" __global__ void dequantize_block_f16_to_f32(const void *__restrict__ vx, float *__restrict__ y, const int k, int qk, int qr)
{
    const int i = blockDim.x * blockIdx.x + 2 * threadIdx.x;

    if (i >= k)
    {
        return;
    }

    const int ib = i / qk;         // block index
    const int iqs = (i % qk) / qr; // quant index
    const int iybs = i - i % qk;   // y block start index
    const int y_offset = qr == 1 ? 1 : qk / 2;

    // dequantize
    dfloat2 v;
    convert_f16(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x;
    y[iybs + iqs + y_offset] = v.y;
}
