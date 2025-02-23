#include <cuda_fp16.h>
#include <stdint.h>

static __device__ void cpy_1_f32_f16(const char *cxi, char *cdsti)
{
    const float *xi = (const float *)cxi;
    half *dsti = (half *)cdsti;

    *dsti = __float2half(*xi);
}

static __device__ void cpy_1_f32_f32(const char *cxi, char *cdsti)
{
    const float *xi = (const float *)cxi;
    float *dsti = (float *)cdsti;

    *dsti = *xi;
}

struct F32Dim
{
    int ne00;
    int ne01;
    int nb00;
    int nb01;
    int nb02;
};

struct F16Dim
{
    int ne10;
    int ne11;
    int nb10;
    int nb11;
    int nb12;
};

// typedef void (*cpy_kernel_t)(const char *cx, char *cdst);

// static __device__ void cpy(const char *cx, char *cdst, const int ne,
//                            F32Dim dim1,
//                            F16Dim dim2, cpy_kernel_t cpy_1)
// {
//     const int ne00 = dim1.ne00;
//     const int ne01 = dim1.ne01;
//     const int nb00 = dim1.nb00;
//     const int nb01 = dim1.nb01;
//     const int nb02 = dim1.nb02;

//     const int ne10 = dim2.ne10;
//     const int ne11 = dim2.ne11;
//     const int nb10 = dim2.nb10;
//     const int nb11 = dim2.nb11;
//     const int nb12 = dim2.nb12;

//     const int i = blockDim.x * blockIdx.x + threadIdx.x;

//     if (i >= ne)
//     {
//         return;
//     }

//     // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
//     // then combine those indices with the corresponding byte offsets to get the total offsets
//     const int i02 = i / (ne00 * ne01);
//     const int i01 = (i - i02 * ne01 * ne00) / ne00;
//     const int i00 = i - i02 * ne01 * ne00 - i01 * ne00;
//     const int x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02;

//     const int i12 = i / (ne10 * ne11);
//     const int i11 = (i - i12 * ne10 * ne11) / ne10;
//     const int i10 = i - i12 * ne10 * ne11 - i11 * ne10;
//     const int dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;

//     cpy_1(cx + x_offset, cdst + dst_offset);
//     // cpy_1_f32_f16(cx + x_offset, cdst + dst_offset);
// }

extern "C" __global__ void cpy_f32_f16(const char *cx, char *cdst, const int ne,
                                       F32Dim dim1,
                                       F16Dim dim2)
{
    // cpy(cx, cdst, ne,
    //     dim1,
    //     dim2, cpy_1_f32_f16);
    const int ne00 = dim1.ne00;
    const int ne01 = dim1.ne01;
    const int nb00 = dim1.nb00;
    const int nb01 = dim1.nb01;
    const int nb02 = dim1.nb02;

    const int ne10 = dim2.ne10;
    const int ne11 = dim2.ne11;
    const int nb10 = dim2.nb10;
    const int nb11 = dim2.nb11;
    const int nb12 = dim2.nb12;

    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= ne)
    {
        return;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int i02 = i / (ne00 * ne01);
    const int i01 = (i - i02 * ne01 * ne00) / ne00;
    const int i00 = i - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02;

    const int i12 = i / (ne10 * ne11);
    const int i11 = (i - i12 * ne10 * ne11) / ne10;
    const int i10 = i - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;

    cpy_1_f32_f16(cx + x_offset, cdst + dst_offset);
}

extern "C" __global__ void cpy_f32_f32(const char *cx, char *cdst, const int ne,
                                       F32Dim dim1,
                                       F16Dim dim2)
{
    const int ne00 = dim1.ne00;
    const int ne01 = dim1.ne01;
    const int nb00 = dim1.nb00;
    const int nb01 = dim1.nb01;
    const int nb02 = dim1.nb02;

    const int ne10 = dim2.ne10;
    const int ne11 = dim2.ne11;
    const int nb10 = dim2.nb10;
    const int nb11 = dim2.nb11;
    const int nb12 = dim2.nb12;

    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= ne)
    {
        return;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int i02 = i / (ne00 * ne01);
    const int i01 = (i - i02 * ne01 * ne00) / ne00;
    const int i00 = i - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02;

    const int i12 = i / (ne10 * ne11);
    const int i11 = (i - i12 * ne10 * ne11) / ne10;
    const int i10 = i - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12;

    cpy_1_f32_f32(cx + x_offset, cdst + dst_offset);
}