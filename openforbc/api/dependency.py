from uuid import UUID

from fastapi import Depends, HTTPException, status

from openforbc.gpu.generic import GPU
from openforbc.gpu.nvidia.gpu import NvidiaGPU
from openforbc.gpu.nvidia.mig import MIGModeStatus


def nvidia_gpu(uuid: UUID) -> NvidiaGPU:
    if not isinstance((gpu := GPU.from_uuid(uuid)), NvidiaGPU):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "not an NVIDIA GPU")

    return gpu


def nvidia_mig_gpu(gpu: NvidiaGPU = Depends(nvidia_gpu)) -> NvidiaGPU:
    if gpu.get_current_mig_status() != MIGModeStatus.ENABLE:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "MIG mode disabled on selected GPU"
        )

    return gpu