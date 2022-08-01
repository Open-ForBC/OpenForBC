from uuid import UUID

from fastapi import APIRouter, Form, HTTPException, status

from openforbc.api.endpoint.gpu.mig import router as mig_router
from openforbc.gpu import GPU

router = APIRouter()


@router.get("/types", tags=["gpu"])
def list_gpu_supported_types(uuid: UUID, creatable: bool = False):
    gpu = GPU.from_uuid(uuid)
    return gpu.get_creatable_types() if creatable else gpu.get_supported_types()


@router.get("/partition", tags=["gpu"])
def list_gpu_partitions(uuid: UUID):
    gpu = GPU.from_uuid(uuid)

    return (
        {"uuid": partition.uuid, "type_id": partition.type.id}
        for partition in gpu.get_partitions()
    )


@router.put("/partition", tags=["gpu", "partition"])
def create_gpu_partition(uuid: UUID, type_id: int = Form()):
    gpu = GPU.from_uuid(uuid)

    if (
        part_type := next(
            (x for x in gpu.get_supported_types() if x.id == type_id), None
        )
    ) is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unsupported partition type")

    if not next((True for x in gpu.get_creatable_types() if x.id == type_id), False):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unavailable partition type")

    partition = gpu.create_partition(part_type)

    return {"ok": True, "uuid": partition.uuid}


@router.delete("/partition/{partition_uuid}", tags=["gpu", "partition"])
def delete_gpu_partition(uuid: UUID, partition_uuid: UUID):
    gpu = GPU.from_uuid(uuid)
    partition = next(
        (x for x in gpu.get_partitions() if x.uuid == partition_uuid), None
    )

    if partition is None:
        raise HTTPException(404, "no partition found with such uuid")

    partition.destroy()

    return {"ok": True}


router.include_router(mig_router, prefix="/mig")
