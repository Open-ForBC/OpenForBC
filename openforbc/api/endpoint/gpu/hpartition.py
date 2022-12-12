# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID  # noqa: TC002

from fastapi import APIRouter, Depends, HTTPException, status

from openforbc.api.dependency import get_gpu
from openforbc.gpu import GPU  # noqa: TC002
from openforbc.gpu.generic import GPUPartitionUse
from openforbc.gpu.model import GPUPartitionModel

if TYPE_CHECKING:
    from typing import Literal

USE: Literal[GPUPartitionUse.HOST_PARTITION] = GPUPartitionUse.HOST_PARTITION
router = APIRouter()


@router.get("/types", tags=["gpu", "hpartition"])
def list_gpu_supported_types(gpu: GPU = Depends(get_gpu), creatable: bool = False):
    return gpu.get_partition_types(USE, creatable)


@router.get("/", tags=["gpu", "hpartition"])
def list_gpu_partitions(gpu: GPU = Depends(get_gpu)):
    return [
        GPUPartitionModel.from_raw(partition) for partition in gpu.get_partitions(USE)
    ]


@router.post("/", tags=["gpu", "hpartition"])
def create_gpu_partition(type_id: int, gpu: GPU = Depends(get_gpu)):
    if (
        part_type := next(
            (x for x in gpu.get_partition_types(USE) if x.id == type_id), None
        )
    ) is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unsupported partition type")

    if not next(
        (True for x in gpu.get_partition_types(USE, True) if x.id == type_id), False
    ):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unavailable partition type")

    return GPUPartitionModel.from_raw(gpu.create_partition(USE, part_type))


@router.delete("/{partition_uuid}", tags=["gpu", "hpartition"])
def delete_gpu_partition(partition_uuid: UUID, gpu: GPU = Depends(get_gpu)):
    partition = next(
        (x for x in gpu.get_partitions(USE) if x.uuid == partition_uuid), None
    )

    if partition is None:
        raise HTTPException(404, "no partition found with such uuid")

    partition.destroy()

    return {"ok": True}
