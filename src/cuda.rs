use crate::ggml_quants::BlockQ4_0;
use crate::ggml_quants::BlockQ8_1;
use crate::ggml_quants::QuantType;
use crate::CpuStorageSlice;
use crate::Dim;
use crate::GGmlType;
use crate::TensorType;
use crate::{error::GResult, CpuStorageView, F16};
use core::ffi::c_void;
use cudarc::cublas::result::gemm_ex;
use cudarc::cublas::result::set_stream;
use cudarc::cublas::result::sgemm;
use cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_16F;
use cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
use cudarc::cublas::sys::cudaDataType_t::CUDA_R_16F;
use cudarc::driver::CudaFunction;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::DeviceRepr;
use cudarc::driver::DeviceSlice;
use cudarc::driver::LaunchAsync;
use cudarc::driver::LaunchConfig;
use cudarc::driver::ValidAsZeroBits;
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
use cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice;
use cudarc::runtime::sys::lib;

const WARP_SIZE: usize = 32;

use core::net;
use std::sync::Arc;
pub struct CudaStorage {
    slice: CudaStorageSlice,
    device: CudaDevice,
}

impl CudaStorage {
    pub fn slice(&self) -> &CudaStorageSlice {
        &self.slice
    }

    pub fn slice_mut(&mut self) -> &mut CudaStorageSlice {
        &mut self.slice
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn offset<'a>(&'a self, offset: usize) -> CudaStorageView<'a> {
        return match &self.slice {
            CudaStorageSlice::Q4_0(v) => CudaStorageView {
                slice: CudaStorageSliceView::Q4_0(v.slice(offset..)),
                device: self.device.clone(),
            },
            CudaStorageSlice::F16(v) => CudaStorageView {
                slice: CudaStorageSliceView::F16(v.slice(offset..)),
                device: self.device.clone(),
            },
            CudaStorageSlice::F32(v) => CudaStorageView {
                slice: CudaStorageSliceView::F32(v.slice(offset..)),
                device: self.device.clone(),
            },
            CudaStorageSlice::I32(v) => CudaStorageView {
                slice: CudaStorageSliceView::I32(v.slice(offset..)),
                device: self.device.clone(),
            },
        };
    }
}

impl CudaStorage {
    pub(crate) fn view<'a>(&'a self) -> CudaStorageView<'a> {
        match &self.slice {
            CudaStorageSlice::Q4_0(v) => {
                let slice = v.slice(..);
                CudaStorageView {
                    slice: CudaStorageSliceView::Q4_0(slice),
                    device: self.device.clone(),
                }
            }
            CudaStorageSlice::F16(v) => {
                let slice = v.slice(..);
                CudaStorageView {
                    slice: CudaStorageSliceView::F16(slice),
                    device: self.device.clone(),
                }
            }
            CudaStorageSlice::F32(v) => {
                let slice = v.slice(..);
                CudaStorageView {
                    slice: CudaStorageSliceView::F32(slice),
                    device: self.device.clone(),
                }
            }
            CudaStorageSlice::I32(v) => {
                let slice = v.slice(..);
                CudaStorageView {
                    slice: CudaStorageSliceView::I32(slice),
                    device: self.device.clone(),
                }
            }
        }
    }
}

pub struct CudaStorageView<'a> {
    slice: CudaStorageSliceView<'a>,
    device: CudaDevice,
}

impl<'a> CudaStorageView<'a> {
    pub fn slice(&self) -> &CudaStorageSliceView {
        &self.slice
    }

    pub fn slice_type(&self) -> GGmlType {
        self.slice.slice_type()
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn offset(&self, offset: usize) -> CudaStorageView<'a> {
        return match &self.slice {
            CudaStorageSliceView::Q4_0(v) => CudaStorageView {
                slice: CudaStorageSliceView::Q4_0(v.slice(offset..)),
                device: self.device.clone(),
            },
            CudaStorageSliceView::F16(v) => CudaStorageView {
                slice: CudaStorageSliceView::F16(v.slice(offset..)),
                device: self.device.clone(),
            },
            CudaStorageSliceView::F32(v) => CudaStorageView {
                slice: CudaStorageSliceView::F32(v.slice(offset..)),
                device: self.device.clone(),
            },
            CudaStorageSliceView::I32(v) => CudaStorageView {
                slice: CudaStorageSliceView::I32(v.slice(offset..)),
                device: self.device.clone(),
            },
        };
    }
}

pub enum CudaStorageSlice {
    Q4_0(CudaSlice<BlockQ4_0>),
    F16(CudaSlice<F16>),
    F32(CudaSlice<f32>),
    I32(CudaSlice<i32>),
}

pub enum CudaStorageSliceView<'a> {
    Q4_0(CudaView<'a, BlockQ4_0>),
    F16(CudaView<'a, F16>),
    F32(CudaView<'a, f32>),
    I32(CudaView<'a, i32>),
}

impl<'a> CudaStorageSliceView<'a> {
    fn slice_type(&self) -> GGmlType {
        match self {
            CudaStorageSliceView::Q4_0(_) => GGmlType::Q4_0,
            CudaStorageSliceView::F16(_) => GGmlType::F16,
            CudaStorageSliceView::F32(_) => GGmlType::F32,
            CudaStorageSliceView::I32(_) => GGmlType::I32,
        }
    }
}

impl CudaStorageSlice {}

#[derive(Clone)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone)]
pub struct CudaDevice {
    id: DeviceId,
    pub(crate) device: Arc<cudarc::driver::CudaDevice>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
}

impl CudaDevice {
    pub fn new(ordinal: usize) -> GResult<Self> {
        let device = cudarc::driver::CudaDevice::new(ordinal)?;
        let blas = cudarc::cublas::CudaBlas::new(device.clone())?;
        Ok(Self {
            id: DeviceId::new(),
            device,
            blas: Arc::new(blas),
        })
    }

    pub(crate) fn from_cpu_storage(&self, c: CpuStorageSlice) -> GResult<CudaStorage> {
        match c {
            CpuStorageSlice::Q4_0(v) => {
                let cuda_slice = self.device.htod_sync_copy(v.as_slice())?;
                Ok(CudaStorage {
                    slice: CudaStorageSlice::Q4_0(cuda_slice),
                    device: self.clone(),
                })
            }
            CpuStorageSlice::Q6K(v) => {
                todo!()
            }
            CpuStorageSlice::F16(v) => {
                todo!()
            }
            CpuStorageSlice::F32(v) => {
                let cuda_slice = self.device.htod_sync_copy(v.as_slice())?;
                Ok(CudaStorage {
                    slice: CudaStorageSlice::F32(cuda_slice),
                    device: self.clone(),
                })
            }
            CpuStorageSlice::I32(v) => {
                todo!()
            }
        }
    }

    pub(crate) fn from_cpu_storage_view(&self, c: CpuStorageView) -> GResult<CudaStorage> {
        match c {
            CpuStorageView::Q4_0(v) => {
                todo!()
            }
            CpuStorageView::Q6K(v) => {
                todo!()
            }
            CpuStorageView::F16(v) => {
                let cuda_slice = self.device.htod_sync_copy(v.as_slice())?;
                Ok(CudaStorage {
                    slice: CudaStorageSlice::F16(cuda_slice),
                    device: self.clone(),
                })
            }
            CpuStorageView::F32(v) => {
                let cuda_slice = self.device.htod_sync_copy(v.as_slice())?;
                Ok(CudaStorage {
                    slice: CudaStorageSlice::F32(cuda_slice),
                    device: self.clone(),
                })
            }
            CpuStorageView::I32(v) => {
                let cuda_slice = self.device.htod_sync_copy(v.as_slice())?;
                Ok(CudaStorage {
                    slice: CudaStorageSlice::I32(cuda_slice),
                    device: self.clone(),
                })
            }
        }
    }

    pub fn load_ptx(&self, module_name: &str, ptx: &'static str) -> GResult<()> {
        let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
        self.device
            .load_ptx(ptx.into(), module_name, &[static_module_name])?;
        Ok(())
    }

    pub fn get_or_load_func(&self, module_name: &str, ptx: &'static str) -> GResult<CudaFunction> {
        let f = self.device.get_func(module_name, module_name).unwrap();
        Ok(f)
    }
}

pub(crate) trait CudaMap2 {
    fn f<T: TensorType + cudarc::driver::DeviceRepr>(
        &self,
        inp0: &CudaView<T>,
        inp0_d: &Dim,
        inp1: &CudaView<T>,
        inp1_d: &Dim,
        dst: &CudaView<T>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        todo!()
    }

    fn f_x_y_y<
        X: TensorType + cudarc::driver::DeviceRepr + ValidAsZeroBits,
        Y: TensorType + cudarc::driver::DeviceRepr + ValidAsZeroBits,
    >(
        &self,
        inp0: &CudaView<X>,
        inp0_d: &Dim,
        inp1: &CudaView<Y>,
        inp1_d: &Dim,
        dst: &CudaView<Y>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        todo!()
    }

    fn f_x_y_x<
        X: TensorType + cudarc::driver::DeviceRepr,
        Y: TensorType + cudarc::driver::DeviceRepr,
    >(
        &self,
        inp0: &CudaView<X>,
        inp0_d: &Dim,
        inp1: &CudaView<Y>,
        inp1_d: &Dim,
        dst: &CudaView<X>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        todo!()
    }

    fn f_q_f32_f32<T: QuantType + TensorType + cudarc::driver::DeviceRepr>(
        &self,
        inp0: &CudaView<T>,
        inp0_d: &Dim,
        inp1: &CudaView<f32>,
        inp1_d: &Dim,
        dst: &CudaView<f32>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        todo!()
    }

    fn map(
        &self,
        dev1: CudaStorageView,
        d1: &Dim,
        dev2: CudaStorageView,
        d2: &Dim,
        dst: &CudaStorageView,
        d3: &Dim,
    ) -> GResult<()> {
        match (dev1.slice(), dev2.slice(), dst.slice()) {
            (
                CudaStorageSliceView::F16(v1),
                CudaStorageSliceView::F16(v2),
                CudaStorageSliceView::F16(d),
            ) => self.f(v1, d1, v2, d2, d, d3, dev1.device()),
            (
                CudaStorageSliceView::F32(v1),
                CudaStorageSliceView::F32(v2),
                CudaStorageSliceView::F32(d),
            ) => self.f(v1, d1, v2, d2, d, d3, dev1.device()),
            (
                CudaStorageSliceView::Q4_0(v1),
                CudaStorageSliceView::F32(v2),
                CudaStorageSliceView::F32(d),
            ) => self.f_q_f32_f32(v1, d1, v2, d2, d, d3, dev1.device()),
            (
                CudaStorageSliceView::F32(v1),
                CudaStorageSliceView::I32(v2),
                CudaStorageSliceView::F32(d),
            ) => self.f_x_y_x(v1, d1, v2, d2, d, d3, dev1.device()),
            (
                CudaStorageSliceView::F16(v1),
                CudaStorageSliceView::F32(v2),
                CudaStorageSliceView::F32(d),
            ) => self.f_x_y_y(v1, d1, v2, d2, d, d3, dev1.device()),
            _ => {
                println!(
                    "{:?} {:?} {:?}",
                    dev1.slice_type(),
                    dev2.slice_type(),
                    dst.slice_type()
                );
                todo!()
            }
        }
    }
}

pub(crate) trait CudaMap {
    fn f<T: TensorType + cudarc::driver::DeviceRepr>(
        &self,
        inp0: &CudaView<T>,
        inp0_d: &Dim,
        dst: &CudaView<T>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        todo!()
    }

    fn f_x_y<
        X: TensorType + cudarc::driver::DeviceRepr,
        Y: TensorType + cudarc::driver::DeviceRepr,
    >(
        &self,
        inp0: &CudaView<X>,
        inp0_d: &Dim,
        dst: &CudaView<Y>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        todo!()
    }

    fn map(&self, dev1: CudaStorageView, d1: &Dim, dst: &CudaStorageView, d3: &Dim) -> GResult<()> {
        match (dev1.slice(), dst.slice()) {
            (CudaStorageSliceView::F16(v1), CudaStorageSliceView::F16(d)) => {
                self.f(v1, d1, d, d3, dev1.device())
            }
            (CudaStorageSliceView::F32(v1), CudaStorageSliceView::F32(d)) => {
                self.f(v1, d1, d, d3, dev1.device())
            }
            (CudaStorageSliceView::F32(v1), CudaStorageSliceView::F16(d)) => {
                self.f_x_y(v1, d1, d, d3, dev1.device())
            }
            _ => {
                todo!()
            }
        }
    }
}

const MATRIX_ROW_PADDING: usize = 512;
const GGML_CUDA_DMMV_X: usize = 32;
const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;

const MMQ_X_Q4_0_RDNA2: usize = 64;
const MMQ_Y_Q4_0_RDNA2: usize = 128;
const NWARPS_Q4_0_RDNA2: usize = 8;
const MMQ_X_Q4_0_RDNA1: usize = 64;
const MMQ_Y_Q4_0_RDNA1: usize = 64;
const NWARPS_Q4_0_RDNA1: usize = 8;
const MMQ_X_Q4_0_AMPERE: usize = 64;
const MMQ_Y_Q4_0_AMPERE: usize = 128;
const NWARPS_Q4_0_AMPERE: usize = 4;
const MMQ_X_Q4_0_PASCAL: usize = 64;
const MMQ_Y_Q4_0_PASCAL: usize = 64;
const NWARPS_Q4_0_PASCAL: usize = 8;

pub(crate) struct CudaMatMul;

impl CudaMatMul {
    fn mul_mat_f16_f32<
        X: TensorType + cudarc::driver::DeviceRepr,
        Y: TensorType + cudarc::driver::DeviceRepr,
    >(
        &self,
        inp0: &CudaView<X>,
        inp1: &CudaView<Y>,
        dst: &CudaView<Y>,
        ncols_x: usize,
        nrows_x: usize,
        nchannels_x: usize,
        nchannels_y: usize,
        dev: &CudaDevice,
    ) -> GResult<()> {
        let cfg: LaunchConfig = LaunchConfig {
            grid_dim: (1, nrows_x as u32, nchannels_y as u32),
            block_dim: (WARP_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = dev
            .get_or_load_func("mul_mat_p021_f16_f32", kernels::MATMUL)
            .unwrap();
        unsafe {
            func.launch(
                cfg,
                (inp0, inp1, dst, ncols_x, nrows_x, nchannels_x, nchannels_y),
            )
        }?;
        Ok(())
    }

    fn convert_fp32_to_fp16_cuda<Y: TensorType + cudarc::driver::DeviceRepr>(
        &self,
        inp1: &CudaView<Y>,
        dst: &CudaSlice<half::f16>,
        k: usize,
        dev: &CudaDevice,
    ) -> GResult<()> {
        assert!(Y::DTYPE == GGmlType::F32);
        let num_blocks = (k + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
        let cfg: LaunchConfig = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (CUDA_DEQUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let q_f16_func = dev.get_or_load_func("dequantize_block_f32_to_f16", kernels::MATMUL)?;
        unsafe { q_f16_func.launch(cfg, (inp1, dst, k, 1, 1)) }?;
        Ok(())
    }

    fn cuda_cpy_tensor_2d<'a, X: TensorType + cudarc::driver::DeviceRepr>(
        &self,
        dst: &'a CudaView<'a, X>,
        src: &'a CudaView<'a, X>,
        src_d: &Dim,
        i3: usize,
        i2: usize,
        i1_low: usize,
        i1_high: usize,
        dev: &CudaDevice,
    ) -> GResult<()> {
        let ne0 = src_d.dim_0();
        let (nb0, nb1, nb2, nb3) = src_d.stride_4d();
        let ts = X::byte_size();
        let bs = X::blck_size();
        let i1_diff = i1_high - i1_low;
        let x = src.slice((i1_low * nb1 + i2 * nb2 + i3 * nb3)..);
        println!(
            "i1_diff:{},nb0:{},nb1:{},ne0:{},bs:{},ts:{}",
            i1_diff, nb0, nb1, ne0, bs, ts
        );
        if nb0 == 1 && nb1 == ne0 / bs {
            //let dst1 = dst.slice((..i1_diff * nb1));
            unsafe {
                lib().cudaMemcpyAsync(
                    *dst.device_ptr() as *mut _,
                    *x.device_ptr() as *const _,
                    i1_diff * nb1 * ts,
                    cudaMemcpyDeviceToDevice,
                    dev.device.cu_stream().clone() as *mut cudarc::runtime::sys::CUstream_st,
                );
            }
        } else if nb0 == 1 {
            unsafe {
                lib().cudaMemcpy2DAsync(
                    *dst.device_ptr() as *mut _,
                    ts * ne0 / bs,
                    *x.device_ptr() as *const _,
                    nb1 * ts,
                    ts * ne0 / bs,
                    i1_diff,
                    cudaMemcpyDeviceToDevice,
                    dev.device.cu_stream().clone() as *mut cudarc::runtime::sys::CUstream_st,
                );
            }
        } else {
            for i1 in 0..i1_diff {
                let rx = x.slice(i1 * nb1..);
                let rd = dst.slice(i1 * ne0 / bs..);
                unsafe {
                    lib().cudaMemcpy2DAsync(
                        *rd.device_ptr() as *mut _,
                        ts / bs,
                        *rx.device_ptr() as *const _,
                        nb0 * ts,
                        ts / bs,
                        ne0,
                        cudaMemcpyDeviceToDevice,
                        dev.device.cu_stream().clone() as *mut cudarc::runtime::sys::CUstream_st,
                    );
                }
            }
        }
        Ok(())
    }

    fn convert_fp16_to_fp32_cuda<
        X: TensorType + cudarc::driver::DeviceRepr,
        Y: TensorType + cudarc::driver::DeviceRepr,
        A: DevicePtr<X>,
        B: DevicePtr<Y>,
    >(
        &self,
        inp0: &A,
        dst: &B,
        k: usize,
        dev: &CudaDevice,
    ) -> GResult<()>
    where
        for<'a> &'a A: DeviceRepr,
        for<'b> &'b B: DeviceRepr,
    {
        assert!(X::DTYPE == GGmlType::F16);
        assert!(Y::DTYPE == GGmlType::F32);
        let num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
        let cfg: LaunchConfig = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (CUDA_DEQUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let q_f32_func = dev.get_or_load_func("dequantize_block_f16_to_f32", kernels::MATMUL)?;
        unsafe { q_f32_func.launch(cfg, (inp0, dst, k, 1, 1)) }?;
        Ok(())
    }

    fn ggml_cuda_mul_mat_vec_p021<
        X: TensorType + cudarc::driver::DeviceRepr,
        Y: TensorType + cudarc::driver::DeviceRepr,
    >(
        &self,
        inp0: &CudaView<X>,
        inp0_d: &Dim,
        inp1: &CudaView<Y>,
        inp1_d: &Dim,
        dst: &CudaView<Y>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        assert!(inp1_d.dim_1() == 1);
        assert!(inp0_d.is_permuted() && inp1_d.is_permuted());
        assert!(inp0_d.stride_0() <= inp0_d.stride_1() && inp0_d.stride_2() <= inp0_d.stride_3());
        assert!(inp1_d.stride_0() <= inp1_d.stride_1() && inp1_d.stride_2() <= inp1_d.stride_3());
        assert!(X::DTYPE == GGmlType::F16);
        assert!(Y::DTYPE == GGmlType::F32);
        let ne00 = inp0_d.dim_0();
        let ne01 = inp0_d.dim_1();
        let ne02 = inp0_d.dim_2();
        let ne12 = inp1_d.dim_2();
        self.mul_mat_f16_f32(inp0, inp1, dst, ne00, ne01, ne02, ne12, dev)?;
        Ok(())
    }

    fn cuda_op_mul_mat_cublas<
        X: TensorType + cudarc::driver::DeviceRepr,
        Y: TensorType + cudarc::driver::DeviceRepr,
    >(
        &self,
        inp0: &CudaView<X>,
        inp0_d: &Dim,
        inp1: &CudaView<Y>,
        inp1_d: &Dim,
        dst: &CudaView<Y>,
        dst_d: &Dim,
        row_low: usize,
        row_high: usize,
        src1_ncols: usize,
        dev: &CudaDevice,
    ) -> GResult<()> {
        let ne00 = inp0_d.dim_0();
        let ne10 = inp1_d.dim_0();
        let ne0 = dst_d.dim_0();
        let row_diff = row_high - row_low;
        if X::DTYPE == GGmlType::F16 && inp0_d.ggml_is_contiguous() && row_diff == inp0_d.dim_1() {
            // 定义半精度标量
            let alpha_f16: half::f16 = half::f16::from_f32(1.0);
            let beta_f16: half::f16 = half::f16::from_f32(0.0);
            let ne = src1_ncols * ne10;
            let inp1_f16 = dev.device.alloc_zeros::<F16>(ne)?;
            if Y::DTYPE == GGmlType::F32 {
                self.convert_fp32_to_fp16_cuda(inp1, &inp1_f16, ne, dev)?;
            }
            let mut dst_f16 = dev.device.alloc_zeros::<F16>(row_diff * src1_ncols)?;
            unsafe {
                gemm_ex(
                    dev.blas.handle().clone(), //
                    CUBLAS_OP_T,               // A 不转置
                    CUBLAS_OP_N,               // B 不转置
                    row_diff as i32,
                    src1_ncols as i32,
                    ne10 as i32,
                    (&alpha_f16) as *const half::f16 as *const c_void,
                    *inp0.device_ptr() as *const _,
                    CUDA_R_16F,
                    ne00 as i32,
                    *inp1_f16.device_ptr() as *const _,
                    CUDA_R_16F,
                    ne10 as i32,
                    (&beta_f16) as *const half::f16 as *const c_void,
                    *dst_f16.device_ptr_mut() as *mut _,
                    CUDA_R_16F,
                    ne0 as i32,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )?;
            }
            self.convert_fp16_to_fp32_cuda(&dst_f16, dst, row_diff * src1_ncols, dev)?;
        } else {
            println!(
                "cublasSgemm:{},{},{},{},{},{}",
                row_diff, src1_ncols, ne10, ne00, ne10, ne0,
            );
            let inp0_as_f32 = dev.device.alloc_zeros::<f32>(row_diff * ne00)?;
            self.convert_fp16_to_fp32_cuda(inp0, &inp0_as_f32, row_diff * ne00, dev)?;
            let alpha = 1.0f32;
            let beta = 0.0f32;
            unsafe {
                sgemm(
                    dev.blas.handle().clone(),
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    row_diff as i32,
                    src1_ncols as i32,
                    ne10 as i32,
                    &alpha as *const f32 as *const _,
                    *inp0_as_f32.device_ptr() as *const _,
                    ne00 as i32,
                    *inp1.device_ptr() as *const _,
                    ne10 as i32,
                    &beta as *const f32 as *const _,
                    *dst.device_ptr() as *mut _,
                    ne0 as i32,
                )?;
            }
        }

        Ok(())
    }

    fn mul_mat_q4_0_q8_1_cuda<T: TensorType + cudarc::driver::DeviceRepr>(
        &self,
        vx: &CudaView<T>,
        vy: &CudaView<BlockQ8_1>,
        dst: &CudaView<f32>,
        ncols_x: usize,
        nrows_x: usize,
        ncols_y: usize,
        nrows_y: usize,
        nrows_dst: usize,
        dev: &CudaDevice,
    ) -> GResult<()> {
        let mmq_x = MMQ_X_Q4_0_AMPERE;
        let mmq_y = MMQ_Y_Q4_0_AMPERE;
        let nwarps = NWARPS_Q4_0_AMPERE;

        let block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
        let block_num_y = (ncols_y + mmq_x - 1) / mmq_x;

        let func = dev
            .get_or_load_func("mul_mat_q4_0", kernels::MATMUL)
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig {
            grid_dim: (block_num_x as u32, block_num_y as u32, 1),
            block_dim: (WARP_SIZE as u32, nwarps as u32, 1),
            shared_mem_bytes: 0,
        };

        if nrows_x % mmq_y == 0 {
            unsafe {
                func.launch(
                    cfg,
                    (
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, false,
                    ),
                )
            }?;
        } else {
            unsafe {
                func.launch(
                    cfg,
                    (
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, true,
                    ),
                )
            }?;
        }
        Ok(())
    }

    fn convert_to_q8<
        X: TensorType + cudarc::driver::DeviceRepr,
        Y: QuantType + cudarc::driver::DeviceRepr,
    >(
        &self,
        inp1: &CudaView<X>,
        inp1_q: &CudaSlice<Y>,
        ne10: usize,
        nrows1: usize,
        src1_padded_col_size: usize,
        dev: &CudaDevice,
    ) -> GResult<()> {
        let block_num_x =
            (src1_padded_col_size + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
        let q_q8_func = dev.get_or_load_func("quantize_q8_1", kernels::MATMUL)?;
        let cfg = LaunchConfig {
            grid_dim: (block_num_x as u32, nrows1 as u32, 1),
            block_dim: (CUDA_DEQUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { q_q8_func.launch(cfg, (inp1, inp1_q, ne10, src1_padded_col_size)) }?;
        Ok(())
    }
}

impl CudaMap2 for CudaMatMul {
    fn f<T: TensorType + cudarc::driver::DeviceRepr>(
        &self,
        inp0: &CudaView<T>,
        inp0_d: &Dim,
        inp1: &CudaView<T>,
        inp1_d: &Dim,
        dst: &CudaView<T>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        todo!();
        Ok(())
    }

    fn f_x_y_y<
        X: TensorType + cudarc::driver::DeviceRepr + ValidAsZeroBits,
        Y: TensorType + cudarc::driver::DeviceRepr + ValidAsZeroBits,
    >(
        &self,
        inp0: &CudaView<X>,
        inp0_d: &Dim,
        inp1: &CudaView<Y>,
        inp1_d: &Dim,
        dst: &CudaView<Y>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        let (ne00, ne01, ne02, ne03) = inp0_d.dim4();
        let (ne10, ne11, ne12, ne13) = inp1_d.dim4();
        let (ne0, ne1) = dst_d.dim2();
        assert!(ne12 >= ne02 && ne12 % ne02 == 0);
        if inp0_d.is_permuted() && inp1_d.is_permuted() && inp1_d.dim_1() == 1 {
            //self.ggml_cuda_mul_mat_vec_p021(inp0, inp0_d, inp1, inp1_d, dst, dst_d, dev)?;
            todo!();
        } else if !inp0_d.ggml_is_contiguous() && inp1_d.ggml_is_contiguous() && inp1_d.dim_1() == 1
        {
            todo!();
        } else if X::DTYPE == GGmlType::F16 {
            if inp1_d.dim_1() == 1 && inp0_d.dim_0() % GGML_CUDA_DMMV_X == 0 {
                //src1->ne[1] == 1 && src0->ne[0] % GGML_CUDA_DMMV_X == 0
                todo!();
            } else {
                assert!(X::DTYPE == GGmlType::F16);
                assert!(Y::DTYPE == GGmlType::F32);
                assert!(Y::DTYPE == GGmlType::F32);

                let src1_col_0 = 0;
                let row_low = 0;
                let row_high = ne01;
                let i02_divisor = ne12 / ne02;
                let src1_ncols = ne11;
                let src0_is_contiguous = inp0_d.ggml_is_contiguous();
                let src1_is_contiguous = inp1_d.ggml_is_contiguous();

                // let size_src0_ddq = inp0_d.elem_count();
                // let src0_dd = dev.device.alloc_zeros::<X>(size_src0_ddq)?;

                // let size_src0_ddq = inp1_d.elem_count();
                // let src1_dd = dev.device.alloc_zeros::<Y>(size_src0_ddq)?;

                // let inp0_tmp = if !src0_is_contiguous {
                //     &src0_dd.slice(0..)
                // } else {
                //     inp0
                // };

                // let inp1_tmp = if !src1_is_contiguous {
                //     &src1_dd.slice(0..)
                // } else {
                //     inp1
                // };

                let mut src0_dd = None;
                let mut src1_dd = None;

                let inp0_tmp = if !src0_is_contiguous {
                    let size_src0_ddq = inp0_d.elem_count();
                    src0_dd = Some(dev.device.alloc_zeros::<X>(size_src0_ddq)?);
                    &src0_dd.as_ref().unwrap().slice(0..)
                } else {
                    inp0
                };

                let inp1_tmp = if !src1_is_contiguous {
                    let size_src1_ddq = inp1_d.elem_count();
                    src1_dd = Some(dev.device.alloc_zeros::<Y>(size_src1_ddq)?);
                    &src1_dd.as_ref().unwrap().slice(0..)
                } else {
                    inp1
                };

                for i0 in 0..ne13 * ne12 {
                    let i03 = i0 / ne12;
                    let i02 = i0 % ne12;

                    let src0_dd_i = inp0_tmp.slice(((i0 / i02_divisor) * ne01 * ne00)..);
                    let src1_ddf_i = inp1_tmp.slice((i0 * ne11 + src1_col_0) * ne10..);
                    let dst_dd_i = dst.slice(((i0 * ne1 + src1_col_0) * ne0)..);

                    if src1_is_contiguous {
                        todo!();
                    } else {
                        self.cuda_cpy_tensor_2d(
                            &src1_ddf_i,
                            inp1,
                            inp1_d,
                            i03,
                            i02,
                            src1_col_0,
                            src1_col_0 + src1_ncols,
                            dev,
                        )?;
                    }

                    if !src0_is_contiguous && i02 % i02_divisor == 0 {
                        self.cuda_cpy_tensor_2d(
                            &src0_dd_i,
                            inp0,
                            inp0_d,
                            i03,
                            i02 / i02_divisor,
                            row_low,
                            row_high,
                            dev,
                        )?;
                    }

                    self.cuda_op_mul_mat_cublas(
                        &src0_dd_i,
                        inp0_d,
                        &src1_ddf_i,
                        inp1_d,
                        &dst_dd_i,
                        dst_d,
                        row_low,
                        row_high,
                        src1_ncols,
                        dev,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn f_q_f32_f32<T: QuantType + TensorType + cudarc::driver::DeviceRepr>(
        &self,
        inp0: &CudaView<T>,
        inp0_d: &Dim,
        inp1: &CudaView<f32>,
        inp1_d: &Dim,
        dst: &CudaView<f32>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        let (ne00, ne01, ne02, ne03) = inp0_d.dim4();
        let (ne10, ne11, ne12, ne13) = inp1_d.dim4();
        let nrows0 = inp0_d.nrows();
        let nrows1 = inp1_d.nrows();
        let (ne0, ne1) = dst_d.dim2();
        let src0_is_contiguous = inp0_d.ggml_is_contiguous();
        let src1_is_contiguous = inp1_d.ggml_is_contiguous();
        if ne11 == 1 && ne00 % GGML_CUDA_DMMV_X == 0 {
            let nb2 = dst_d.stride_2();
            let nb3 = dst_d.stride_3();
            let row_diff = ne01;
            let nrows = row_diff;
            let block_num_y = nrows as u32;
            let cfg = LaunchConfig {
                grid_dim: (1, block_num_y, 1),
                block_dim: (WARP_SIZE as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let blk_siz = T::BLCK_SIZE;
            let i02_divisor = ne12 / ne02;
            let func = dev.get_or_load_func("dequantize_mul_mat_vec_q4_0", kernels::MATMUL)?;
            for i0 in 0..ne13 * ne12 {
                let src0_dd_i = inp0.slice(((i0 / i02_divisor) * ne01 * ne00 / blk_siz)..);
                let src1_ddf_i = inp1.slice((i0 * ne11 + 0) * ne10..);
                let dst_dd_i = dst.slice(((i0 * ne1 + 0) * ne0)..);
                unsafe {
                    func.clone()
                        .launch(cfg, (&src0_dd_i, &src1_ddf_i, &dst_dd_i, ne00, row_diff))
                }?;
            }
        } else {
            let row_low = 0;
            let row_high = ne01;
            let row_diff = row_high - row_low;
            let src1_ncols = ne11;
            let src1_padded_col_size = if ne10 % MATRIX_ROW_PADDING == 0 {
                ne10
            } else {
                ne10 - ne10 % MATRIX_ROW_PADDING + MATRIX_ROW_PADDING
            };
            let blk_siz = T::BLCK_SIZE;
            let inp1_q: CudaSlice<BlockQ8_1> = dev
                .device
                .alloc_zeros::<BlockQ8_1>(inp1.len() / BlockQ8_1::BLCK_SIZE)?;
            let i02_divisor = ne12 / ne02;

            let nrows_dst = ne0;
            self.convert_to_q8(inp1, &inp1_q, ne10, nrows1, src1_padded_col_size, dev)?;
            for i0 in 0..ne13 * ne12 {
                let i03 = i0 / ne12;
                let i02 = i0 % ne12;

                let src1_ddq_i_offset =
                    (i0 * ne11 + 0) * src1_padded_col_size / BlockQ8_1::BLCK_SIZE;
                let src1_ddf_i = inp1.slice((i0 * ne11 + 0) * ne10..);
                let src0_dd_i = inp0.slice(((i0 / i02_divisor) * ne01 * ne00 / blk_siz)..);
                let src1_ddq_i = inp1_q.slice(src1_ddq_i_offset..);
                let dst_dd_i = dst.slice(((i0 * ne1 + 0) * ne0)..);

                if src1_is_contiguous {
                } else {
                    self.cuda_cpy_tensor_2d(
                        &src1_ddf_i,
                        inp1,
                        inp1_d,
                        i03,
                        i02,
                        0,
                        0 + src1_ncols,
                        dev,
                    )?;
                }

                if !src0_is_contiguous && i02 % i02_divisor == 0 {
                    self.cuda_cpy_tensor_2d(
                        &src0_dd_i,
                        inp0,
                        inp0_d,
                        i03,
                        i02 / i02_divisor,
                        row_low,
                        row_high,
                        dev,
                    )?;
                }

                self.mul_mat_q4_0_q8_1_cuda(
                    &src0_dd_i,
                    &src1_ddq_i,
                    &dst_dd_i,
                    ne00,
                    row_diff,
                    src1_ncols,
                    src1_padded_col_size,
                    nrows_dst,
                    dev,
                )?;
            }
        }

        Ok(())
    }
}

pub(crate) struct CudaRmsNorm {
    pub(crate) eps: f32,
}

impl CudaMap for CudaRmsNorm {
    fn f<T: TensorType + cudarc::driver::DeviceRepr>(
        &self,
        inp0: &CudaView<T>,
        inp0_d: &Dim,
        dst: &CudaView<T>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        let ne00 = inp0_d.dim1();
        let nrows = inp0_d.nrows();
        assert!(ne00 % 32 == 0);
        let func = dev.get_or_load_func("rms_norm_f32", kernels::RMS_NORM)?;
        let (cfg, block_size) = if ne00 < 1024 {
            (
                LaunchConfig {
                    grid_dim: (nrows as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                32,
            )
        } else {
            (
                LaunchConfig {
                    grid_dim: (nrows as u32, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                },
                1024,
            )
        };
        unsafe { func.launch(cfg, (inp0, dst, ne00, self.eps, block_size)) }?;
        Ok(())
    }
}

const CUDA_MUL_BLOCK_SIZE: usize = 256;
const CUDA_ROPE_BLOCK_SIZE: usize = 256;
pub(crate) struct CudaMul;

impl CudaMap2 for CudaMul {
    fn f<T: TensorType + cudarc::driver::DeviceRepr>(
        &self,
        inp0: &CudaView<T>,
        inp0_d: &Dim,
        inp1: &CudaView<T>,
        inp1_d: &Dim,
        dst: &CudaView<T>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        assert!(T::DTYPE == GGmlType::F32);
        let (ne10, ne11) = inp1_d.dim2();
        let kx = inp0_d.elem_count();
        let ky = ne10 * ne11;
        let num_blocks = (kx + CUDA_MUL_BLOCK_SIZE - 1) / CUDA_MUL_BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (CUDA_MUL_BLOCK_SIZE as u32, 1, 1),
            block_dim: (num_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = dev.get_or_load_func("mul_f32", kernels::MUL)?;
        unsafe { func.launch(cfg, (inp0, inp1, dst, kx, ky)) }?;
        Ok(())
    }
}

pub(crate) struct CudaRope {
    pub(crate) n_dims: usize,
    pub(crate) mode: i32,
    pub(crate) n_ctx: i32,
    pub(crate) freq_base: f32,
    pub(crate) freq_scale: f32,
    pub(crate) xpos_base: f32,
    pub(crate) xpos_down: bool,
}

impl CudaRope {
    fn rope_cuda_f32<
        X: TensorType + cudarc::driver::DeviceRepr,
        Y: TensorType + cudarc::driver::DeviceRepr,
    >(
        &self,
        inp0: &CudaView<X>,
        inp1: &CudaView<Y>,
        dst: &CudaView<X>,
        ncols: usize,
        nrows: usize,
        freq_scale: f32,
        p_delta_rows: usize,
        theta_scale: f32,
        dev: &CudaDevice,
    ) -> GResult<()> {
        assert!(ncols % 2 == 0);
        let num_blocks_x = (ncols + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);

        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, num_blocks_x as u32, 1),
            block_dim: (1, CUDA_ROPE_BLOCK_SIZE as u32, 1),
            shared_mem_bytes: 0,
        };
        let func = dev.get_or_load_func("rope_f32", kernels::ROPE)?;
        unsafe {
            func.launch(
                cfg,
                (
                    inp0,
                    dst,
                    ncols,
                    inp1,
                    freq_scale,
                    p_delta_rows,
                    theta_scale,
                    true,
                ),
            )
        }?;
        Ok(())
    }
}

impl CudaMap2 for CudaRope {
    fn f_x_y_x<
        X: TensorType + cudarc::driver::DeviceRepr,
        Y: TensorType + cudarc::driver::DeviceRepr,
    >(
        &self,
        inp0: &CudaView<X>,
        inp0_d: &Dim,
        inp1: &CudaView<Y>,
        inp1_d: &Dim,
        dst: &CudaView<X>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        assert!(X::DTYPE == GGmlType::F32);
        let (ne00, ne01) = inp0_d.dim2();
        let ne2 = dst_d.dim_2();
        let nrows = inp0_d.nrows();

        let n_dims = self.n_dims;
        let mode = self.mode;
        let n_ctx = self.n_ctx;
        let freq_base = self.freq_base;
        let freq_scale = self.freq_scale;
        let xpos_base = self.xpos_base;
        let xpos_down = self.xpos_down;

        let theta_scale = (freq_base as f32).powf(-2.0 / n_dims as f32);

        if (mode & 1) == 0 {
            assert!(Y::DTYPE == GGmlType::I32);
            assert!(inp1_d.dim_0() == ne2);
        }

        let is_neox = mode & 2 != 0;
        let is_glm = mode & 4 != 0;
        if is_glm {
            todo!()
        } else if is_neox {
            todo!()
        } else {
            self.rope_cuda_f32(
                inp0,
                inp1,
                dst,
                ne00,
                nrows,
                freq_scale,
                ne01,
                theta_scale,
                dev,
            )?;
        }

        Ok(())
    }
}

const CUDA_CPY_BLOCK_SIZE: usize = 32;

unsafe impl DeviceRepr for F32Dim {}

#[repr(C)]
struct F32Dim {
    pub ne00: i32,
    pub ne01: i32,
    pub nb00: i32,
    pub nb01: i32,
    pub nb02: i32,
}

unsafe impl DeviceRepr for F16Dim {}

#[repr(C)]
struct F16Dim {
    pub ne10: i32,
    pub ne11: i32,
    pub nb10: i32,
    pub nb11: i32,
    pub nb12: i32,
}

pub(crate) struct CudaCpy;

impl CudaCpy {
    fn cpy_f32_f16_cuda<
        X: TensorType + cudarc::driver::DeviceRepr,
        Y: TensorType + cudarc::driver::DeviceRepr,
    >(
        &self,
        src: &CudaView<X>,
        dst: &CudaView<Y>,
        ne: usize,
        ne00: usize,
        ne01: usize,
        nb00: usize,
        nb01: usize,
        nb02: usize,
        ne10: usize,
        ne11: usize,
        nb10: usize,
        nb11: usize,
        nb12: usize,
        dev: &CudaDevice,
    ) -> GResult<()> {
        assert!(X::DTYPE == GGmlType::F32);
        assert!(Y::DTYPE == GGmlType::F16);
        let num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (CUDA_CPY_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = dev.get_or_load_func("cpy_f32_f16", kernels::CPY)?;
        unsafe {
            func.launch(
                cfg,
                (
                    src,
                    dst,
                    ne,
                    F32Dim {
                        ne00: ne00 as i32,
                        ne01: ne01 as i32,
                        nb00: nb00 as i32,
                        nb01: nb01 as i32,
                        nb02: nb02 as i32,
                    },
                    F16Dim {
                        ne10: ne10 as i32,
                        ne11: ne11 as i32,
                        nb10: nb10 as i32,
                        nb11: nb11 as i32,
                        nb12: nb12 as i32,
                    },
                ),
            )
        }?;
        Ok(())
    }
}

impl CudaMap for CudaCpy {
    fn f_x_y<
        X: TensorType + cudarc::driver::DeviceRepr,
        Y: TensorType + cudarc::driver::DeviceRepr,
    >(
        &self,
        inp0: &CudaView<X>,
        inp0_d: &Dim,
        dst: &CudaView<Y>,
        dst_d: &Dim,
        dev: &CudaDevice,
    ) -> GResult<()> {
        let ne = inp0_d.elem_count(); //ggml_nelements(src0);
        assert!(ne == dst_d.elem_count());

        assert!(inp0_d.elem_count() * X::byte_size() <= usize::MAX);
        assert!(dst_d.elem_count() * Y::byte_size() <= usize::MAX);

        let (ne00, ne01) = inp0_d.dim2();

        assert!(inp0_d.dim_3() == 1);
        let (_nb00, _nb01, _nb02) = inp0_d.stride_3d();
        let (nb00, nb01, nb02) = (
            _nb00 * X::byte_size(),
            _nb01 * X::byte_size(),
            _nb02 * X::byte_size(),
        );

        let (ne10, ne11) = dst_d.dim2();
        assert!(dst_d.dim_3() == 1);

        let (_nb10, _nb11, _nb12) = dst_d.stride_3d();
        let (nb10, nb11, nb12) = (
            _nb10 * Y::byte_size(),
            _nb11 * Y::byte_size(),
            _nb12 * Y::byte_size(),
        );

        self.cpy_f32_f16_cuda(
            inp0, dst, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12, dev,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::init_cuda_function;
    use crate::{op::galois_mul, Device, Shape, Tensor, TensorProto};
    use kernels::RMS_NORM;

    use super::*;
    use crate::multiply_tuple;
    use crate::op::galois_rms_norm;

    #[test]
    fn test_cuda_matmul() {
        let gpu_dev = Device::Gpu(CudaDevice::new(0).unwrap());
        let a = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0],
            2,
            Shape::from_array([2, 2]),
            &gpu_dev,
        )
        .unwrap();

        let b = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0],
            2,
            Shape::from_array([2, 2]),
            &gpu_dev,
        )
        .unwrap();

        let mut c = Tensor::from_vec(
            vec![0.0f32, 0.0, 0.0, 0.0],
            2,
            Shape::from_array([2, 2]),
            &gpu_dev,
        )
        .unwrap();
    }

    #[test]
    fn test_cuda_rmsnorm() {
        let gpu_dev = Device::Gpu(CudaDevice::new(0).unwrap());
        let a = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0],
            2,
            Shape::from_array([2, 2]),
            &gpu_dev,
        )
        .unwrap();

        // let b = Tensor::from_vec(
        //     vec![1.0f32, 2.0, 3.0, 4.0],
        //     2,
        //     Shape::from_array([2, 2]),
        //     &gpu_dev,
        // )
        // .unwrap();

        let mut c = Tensor::from_vec(
            vec![0.0f32, 0.0, 0.0, 0.0],
            2,
            Shape::from_array([2, 2]),
            &gpu_dev,
        )
        .unwrap();

        galois_rms_norm(&a, &mut c, 0.1).unwrap();

        let c1 = c.to_cpu_tensor().unwrap();
        println!("{:?}", unsafe { c1.as_slice::<f32>() });
    }

    #[test]
    fn test_cuda_mul() {
        let cuda_dev = CudaDevice::new(0).unwrap();
        init_cuda_function(&cuda_dev).unwrap();
        let gpu_dev = Device::Gpu(cuda_dev);
        let a = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0],
            2,
            Shape::from_array([2, 2]),
            &gpu_dev,
        )
        .unwrap();

        let b = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0],
            2,
            Shape::from_array([2, 2]),
            &gpu_dev,
        )
        .unwrap();

        let mut c = Tensor::from_vec(
            vec![0.0f32, 0.0, 0.0, 0.0],
            2,
            Shape::from_array([2, 2]),
            &gpu_dev,
        )
        .unwrap();

        galois_mul(&a, &b, &mut c).unwrap();

        let c1 = c.to_cpu_tensor().unwrap();
        println!("{:?}", unsafe { c1.as_slice::<f32>() });
    }
    #[test]
    fn test_multiply_tuple() {
        let c = multiply_tuple!(2, (1, 2, 3));
        let d = multiply_tuple!(2, (4, 5, 6));
        println!("{:?}", c);
        println!("{:?}", d);
    }
}
