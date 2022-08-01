from uuid import UUID

from fastapi import APIRouter, Depends, Form, HTTPException, status

from openforbc.api.dependency import get_gpu
from openforbc.api.endpoint.gpu.mig import router as mig_router
from openforbc.gpu import GPU
from openforbc.gpu.model import GPUPartitionModel

router = APIRouter()


@router.get("/types", tags=["gpu"])
def list_gpu_supported_types(gpu: GPU = Depends(get_gpu), creatable: bool = False):
    types = gpu.get_creatable_types() if creatable else gpu.get_supported_types()
    return [type.into_generic() for type in types]


@router.get("/partition", tags=["gpu"])
def list_gpu_partitions(gpu: GPU = Depends(get_gpu)):
    return [GPUPartitionModel.from_raw(partition) for partition in gpu.get_partitions()]


@router.post("/partition", tags=["gpu", "partition"])
def create_gpu_partition(gpu: GPU = Depends(get_gpu), type_id: int = Form()):
    if (
        part_type := next(
            (x for x in gpu.get_supported_types() if x.id == type_id), None
        )
    ) is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unsupported partition type")

    if not next((True for x in gpu.get_creatable_types() if x.id == type_id), False):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unavailable partition type")

    return GPUPartitionModel.from_raw(gpu.create_partition(part_type))


@router.delete("/partition/{partition_uuid}", tags=["gpu", "partition"])
def delete_gpu_partition(partition_uuid: UUID, gpu: GPU = Depends(get_gpu)):
    partition = next(
        (x for x in gpu.get_partitions() if x.uuid == partition_uuid), None
    )

    if partition is None:
        raise HTTPException(404, "no partition found with such uuid")

    partition.destroy()

    return {"ok": True}


router.include_router(mig_router, prefix="/mig")
