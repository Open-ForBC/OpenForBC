from fastapi import APIRouter, Depends

from openforbc.api.dependency import nvidia_gpu, nvidia_mig_gpu
from openforbc.gpu.nvidia.gpu import NvidiaGPU
from openforbc.gpu.nvidia.model import GPUInstanceModel

router = APIRouter()


@router.get("/mode", tags=["gpu", "mig"])
def get_gpu_mig_mode(gpu: NvidiaGPU = Depends(nvidia_gpu)):
    return gpu.get_current_mig_status()


@router.get("/profile", tags=["gpu", "mig"])
def get_gpu_mig_ci_profiles(gpu: NvidiaGPU = Depends(nvidia_mig_gpu)):
    return gpu.get_supported_gpu_instance_profiles()


@router.get("/instance", tags=["gpu", "mig"])
def get_gpu_mig_instances(gpu: NvidiaGPU = Depends(nvidia_mig_gpu)):
    return [GPUInstanceModel.from_raw(instance) for instance in gpu.get_gpu_instances()]


@router.post("/instance", tags=["gpu", "mig"])
def create_mig_gpu_instance(profile_id: int, gpu: NvidiaGPU = Depends(nvidia_mig_gpu)):
    return GPUInstanceModel.from_raw(gpu.create_gpu_instance(profile_id))


@router.get("/instance/{id}", tags=["gpu", "mig"])
def get_gpu_mig_instance_by_id(id: int, gpu: NvidiaGPU = Depends(nvidia_mig_gpu)):
    return GPUInstanceModel.from_raw(gpu.get_gpu_instance_by_id(id))


@router.delete("/instance/{id}", tags=["gpu", "mig"])
def destroy_gpu_mig_instance_by_id(id: int, gpu: NvidiaGPU = Depends(nvidia_mig_gpu)):
    gpu.get_gpu_instance_by_id(id).destroy()
    return {"ok": True}
