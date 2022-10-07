# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from openforbc.api.dependency import get_gpu
from openforbc.gpu import GPU
from openforbc.gpu.model import GPUPartitionModel

router = APIRouter()


@router.get("/types", tags=["gpu", "hpartition"])
def list_gpu_supported_types(gpu: GPU = Depends(get_gpu), creatable: bool = False):
    return (
        gpu.get_creatable_hpart_types()
        if creatable
        else gpu.get_supported_hpart_types()
    )


@router.get("/", tags=["gpu", "hpartition"])
def list_gpu_partitions(gpu: GPU = Depends(get_gpu)):
    return [
        GPUPartitionModel.from_raw(partition) for partition in gpu.get_hpartitions()
    ]


@router.post("/", tags=["gpu", "hpartition"])
def create_gpu_partition(type_id: int, gpu: GPU = Depends(get_gpu)):
    if (
        part_type := next(
            (x for x in gpu.get_supported_hpart_types() if x.id == type_id), None
        )
    ) is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unsupported partition type")

    if not next(
        (True for x in gpu.get_creatable_hpart_types() if x.id == type_id), False
    ):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unavailable partition type")

    return GPUPartitionModel.from_raw(gpu.create_hpartition(part_type))


@router.delete("/{partition_uuid}", tags=["gpu", "hpartition"])
def delete_gpu_partition(partition_uuid: UUID, gpu: GPU = Depends(get_gpu)):
    partition = next(
        (x for x in gpu.get_hpartitions() if x.uuid == partition_uuid), None
    )

    if partition is None:
        raise HTTPException(404, "no partition found with such uuid")

    partition.destroy()

    return {"ok": True}
