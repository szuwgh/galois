use crate::ggml_quants::BlockQ4_0;
use crate::CpuStorageSlice;
use crate::Dim;
use crate::StorageProto;
use crate::StorageView;
use crate::TensorProto;
use crate::TensorType;
use crate::{error::GResult, CpuStorageView, F16};
use cudarc::driver::CudaFunction;
use cudarc::driver::LaunchAsync;
use cudarc::driver::LaunchConfig;
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
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
}

impl CudaStorage {
    pub(crate) fn view<'a>(&'a self) -> CudaStorageView<'a> {
        match &self.slice {
            CudaStorageSlice::Q4_0(v) => {
                let slice = v.slice(0..);
                CudaStorageView {
                    slice: CudaStorageSliceView::Q4_0(slice),
                    device: self.device.clone(),
                }
            }
            CudaStorageSlice::F16(v) => {
                let slice = v.slice(0..);
                CudaStorageView {
                    slice: CudaStorageSliceView::F16(slice),
                    device: self.device.clone(),
                }
            }
            CudaStorageSlice::F32(v) => {
                let slice = v.slice(0..);
                CudaStorageView {
                    slice: CudaStorageSliceView::F32(slice),
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

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }
}

pub enum CudaStorageSlice {
    Q4_0(CudaSlice<BlockQ4_0>),
    F16(CudaSlice<F16>),
    F32(CudaSlice<f32>),
}

pub enum CudaStorageSliceView<'a> {
    Q4_0(CudaView<'a, BlockQ4_0>),
    F16(CudaView<'a, F16>),
    F32(CudaView<'a, f32>),
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
                todo!()
            }
            CpuStorageView::F32(v) => {
                let cuda_slice = self.device.htod_sync_copy(v.as_slice())?;

                Ok(CudaStorage {
                    slice: CudaStorageSlice::F32(cuda_slice),
                    device: self.clone(),
                })
            }
            CpuStorageView::I32(v) => {
                todo!()
            }
        }
    }

    pub fn get_or_load_func(&self, module_name: &str, ptx: &'static str) -> GResult<CudaFunction> {
        // if !self.has_func(module_name, module_name) {
        //     // Leaking the string here is a bit sad but we need a &'static str and this is only
        //     // done once per kernel name.
        //     let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
        //     self.load_ptx(ptx.into(), module_name, &[static_module_name])
        //         .map_err(|cuda| CudaError::Load {
        //             cuda,
        //             module_name: module_name.to_string(),
        //         })
        //         .w()?;
        // }
        let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
        self.device
            .load_ptx(ptx.into(), module_name, &[static_module_name])?;
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
    ) -> GResult<()>;

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
            _ => {
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
    ) -> GResult<()>;

    fn map(&self, dev1: CudaStorageView, d1: &Dim, dst: &CudaStorageView, d3: &Dim) -> GResult<()> {
        match (dev1.slice(), dst.slice()) {
            (CudaStorageSliceView::F16(v1), CudaStorageSliceView::F16(d)) => {
                self.f(v1, d1, d, d3, dev1.device())
            }
            (CudaStorageSliceView::F32(v1), CudaStorageSliceView::F32(d)) => {
                self.f(v1, d1, d, d3, dev1.device())
            }
            _ => {
                todo!()
            }
        }
    }
}

pub(crate) struct CudaMatMul;

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
        let func = dev.get_or_load_func("matmul", kernels::MATMUL)?;
        let mut c_host = [T::zeros(); 4];
        let cfg = LaunchConfig {
            block_dim: (2, 2, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { func.launch(cfg, (inp0, inp1, dst, 2i32)) }?;

        println!("Found {:?}", c_host);
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

// pub fn cuda_matmul<T: TensorProto>(src0: &T, src1: &T, dst: &mut T) -> GResult<()> {
//     let (dst_device, dst_dim) = (dst.storage().view(), dst.dim());
//     match (src0.storage().view(), src1.storage().view(), dst_device) {
//         (StorageView::Gpu(s0), StorageView::Gpu(s1), StorageView::Gpu(mut d)) => {
//             CudaMatMul.map(s0, src0.dim(), s1, src1.dim(), &mut d, dst_dim)?;
//         }
//         _ => {
//             todo!()
//         }
//     }
//     Ok(())
// }

#[cfg(test)]
mod tests {
    use kernels::RMS_NORM;

    use crate::{Device, Shape, Tensor, TensorProto};

    use super::*;
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
}
