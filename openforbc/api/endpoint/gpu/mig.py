# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
from __future__ import annotations

from http.client import HTTPException

from fastapi import APIRouter, Depends, status

from openforbc.api.dependency import nvidia_gpu, nvidia_mig_gpu
from openforbc.gpu.nvidia.gpu import NvidiaGPU  # noqa: TC001
from openforbc.gpu.nvidia.mig import MIGModeStatus  # noqa: TC001
from openforbc.gpu.nvidia.model import ComputeInstanceModel, GPUInstanceModel


router = APIRouter()


@router.get("/mode", tags=["gpu", "mig"])
def get_gpu_mig_mode(gpu: NvidiaGPU = Depends(nvidia_gpu)):
    return gpu.get_current_mig_status()


@router.post("/mode", tags=["gpu", "mig"])
def set_gpu_mig_mode(mode: MIGModeStatus, gpu: NvidiaGPU = Depends(nvidia_gpu)):
    gpu.set_mig_mode(mode)
    return gpu.get_current_mig_status()


@router.get("/ci", tags=["gpu", "mig"])
def get_compute_instances(gpu: NvidiaGPU = Depends(nvidia_mig_gpu)):
    cis: list[ComputeInstanceModel] = []
    for gi in gpu.get_gpu_instances():
        cis.extend(
            ComputeInstanceModel.from_raw(instance)
            for instance in gi.get_compute_instances()
        )

    return cis


@router.get("/gi", tags=["gpu", "mig"])
def get_gpu_mig_instances(gpu: NvidiaGPU = Depends(nvidia_mig_gpu)):
    return [GPUInstanceModel.from_raw(instance) for instance in gpu.get_gpu_instances()]


@router.get("/gi/profile", tags=["gpu", "mig"])
def get_gpu_instance_profiles(gpu: NvidiaGPU = Depends(nvidia_mig_gpu)):
    return gpu.get_supported_gpu_instance_profiles()


@router.get("/gi/profile/{gip_id}/capacity", tags=["gpu", "mig"])
def get_gpu_instance_profile_capacity(
    gip_id: int, gpu: NvidiaGPU = Depends(nvidia_mig_gpu)
):
    try:
        gi_profile = next(
            profile
            for profile in gpu.get_supported_gpu_instance_profiles()
            if profile.id == gip_id
        )
    except StopIteration:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no such gi profile: {gip_id}")

    return gpu.get_gpu_instance_remaining_capacity(gi_profile)


@router.post("/gi", tags=["gpu", "mig"])
def create_gpu_instance(
    gip_id: int, default_ci: bool = True, gpu: NvidiaGPU = Depends(nvidia_mig_gpu)
):
    return GPUInstanceModel.from_raw(gpu.create_gpu_instance(gip_id, default_ci))


@router.get("/gi/{id}", tags=["gpu", "mig"])
def get_gpu_mig_instance_by_id(id: int, gpu: NvidiaGPU = Depends(nvidia_mig_gpu)):
    return GPUInstanceModel.from_raw(gpu.get_gpu_instance_by_id(id))


@router.delete("/gi/{id}", tags=["gpu", "mig"])
def destroy_gpu_mig_instance_by_id(id: int, gpu: NvidiaGPU = Depends(nvidia_mig_gpu)):
    gpu.get_gpu_instance_by_id(id).destroy()
    return {"ok": True}


@router.get("/gi/{gid}/ci", tags=["gpu", "mig"])
def get_gi_compute_instances(gid: int, gpu: NvidiaGPU = Depends(nvidia_mig_gpu)):
    return [
        ComputeInstanceModel.from_raw(ci)
        for ci in gpu.get_gpu_instance_by_id(gid).get_compute_instances()
    ]


@router.get("/gi/{gid}/ci/profile", tags=["gpu", "mig"])
def get_gi_compute_instance_profiles(
    gid: int, gpu: NvidiaGPU = Depends(nvidia_mig_gpu)
):
    return gpu.get_gpu_instance_by_id(gid).get_compute_instance_profiles()


@router.get("/gi/{gid}/ci/profile/{cip_id}/capacity", tags=["gpu", "mig"])
def get_compute_instance_profile_capacity(
    gid: int, cip_id: int, gpu: NvidiaGPU = Depends(nvidia_mig_gpu)
):
    gi = gpu.get_gpu_instance_by_id(gid)
    try:
        cip = next(x for x in gi.get_compute_instance_profiles() if x.id == cip_id)
    except StopIteration:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "no such ci profile {cip_id} for gi {gid}"
        )

    return gi.get_compute_instance_remaining_capacity(cip)


@router.post("/gi/{gid}/ci", tags=["gpu", "mig"])
def create_compute_instance(
    gid: int, cip_id: int, gpu: NvidiaGPU = Depends(nvidia_mig_gpu)
):
    gi = gpu.get_gpu_instance_by_id(gid)
    try:
        ci_profile = next(
            profile
            for profile in gi.get_compute_instance_profiles()
            if profile.id == cip_id
        )
    except StopIteration:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"no such ci profile {cip_id} for gi {gid}"
        )

    return ComputeInstanceModel.from_raw(gi.create_compute_instance(ci_profile))


@router.delete("/gi/{gid}/ci/{cid}", tags=["gpu", "mig"])
def destroy_compute_instance(
    gid: int, cid: int, gpu: NvidiaGPU = Depends(nvidia_mig_gpu)
):
    cis = gpu.get_gpu_instance_by_id(gid).get_compute_instances()
    if (ci := next((ci for ci in cis if ci.id == cid), None)) is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"no ci #{cid} found for gi #{gid}"
        )
    ci.destroy()
    return {"ok": True}
