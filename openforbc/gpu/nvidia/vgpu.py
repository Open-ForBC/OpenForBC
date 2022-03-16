from __future__ import annotations
from enum import Enum
from pynvml import (
    nvmlVgpuTypeGetName,
    nvmlVgpuTypeGetClass,
    nvmlVgpuTypeGetFramebufferSize,
    nvmlVgpuTypeGetGpuInstanceProfileId,
    NVML_HOST_VGPU_MODE_NON_SRIOV,
    NVML_HOST_VGPU_MODE_SRIOV,
)
from typing import TYPE_CHECKING
from uuid import UUID

from openforbc.gpu.generic import GPUPartition, GPUPartitionType, GPUPartitionTechnology
from openforbc.gpu.nvidia.nvml import nvml
from openforbc.sysfs.mdev import MdevSysFsHandle

if TYPE_CHECKING:
    from openforbc.gpu.nvidia.gpu import NvidiaGPU

INVALID_GPU_INSTANCE_PROFILE_ID = 0xFFFFFFFF


class VGPUMode(Enum):
    NON_SRIOV = NVML_HOST_VGPU_MODE_NON_SRIOV
    SRIOV = NVML_HOST_VGPU_MODE_SRIOV


class VGPUCreateException(Exception):
    pass


class VGPUTypeException(Exception):
    pass


class VGPUType(GPUPartitionType):
    def __init__(
        self, id: int, name: str, vgpu_class: str, fb_size: int, gip_id: int
    ) -> None:
        is_mig = gip_id != INVALID_GPU_INSTANCE_PROFILE_ID
        super().__init__(
            name,
            id,
            GPUPartitionTechnology.NVIDIA_VGPU_MIG
            if is_mig
            else GPUPartitionTechnology.NVIDIA_VGPU_TIMESHARED,
            fb_size,
        )

        self.id = id
        self.name = name
        self.vgpu_class = vgpu_class
        self.is_mig = is_mig
        self.gip_id = gip_id

    def __repr__(self) -> str:
        return f"{self.id}: {self.name}{' (MIG)' if self.is_mig else ''}"

    @classmethod
    def from_id(cls, id: int) -> VGPUType:
        with nvml():
            return VGPUType(
                id,
                nvmlVgpuTypeGetName(id).decode(),
                nvmlVgpuTypeGetClass(id).decode(),
                nvmlVgpuTypeGetFramebufferSize(id),
                nvmlVgpuTypeGetGpuInstanceProfileId(id),
            )

    def get_mdev_type(self) -> str:
        return f"nvidia-{self.id}"


class VGPUInstance(GPUPartition):
    def __init__(
        self, sysfs_handle: MdevSysFsHandle, uuid: UUID, type: VGPUType
    ) -> None:
        super().__init__(uuid, type)

        self._sysfs_handle = sysfs_handle
        self.uuid = uuid
        self.type = type

    def __repr__(self) -> str:
        return f"{self.type} {self.uuid} on {self._sysfs_handle.get_parent_gpu()}"

    @classmethod
    def from_sysfs_handle(cls, sysfs_handle: MdevSysFsHandle) -> VGPUInstance:
        UUID
        uuid = sysfs_handle.get_uuid()
        mdev_type = sysfs_handle.get_mdev_type()
        if not mdev_type.startswith("nvidia-"):
            raise VGPUTypeException(f"VGPU type not recognized: {mdev_type}")
        type = VGPUType.from_id(int(mdev_type[len("nvidia-") :]))

        return cls(sysfs_handle, uuid, type)

    @classmethod
    def from_mdev_uuid(cls, uuid: UUID) -> VGPUInstance:
        return cls.from_sysfs_handle(MdevSysFsHandle.from_uuid(uuid))

    def destroy(self) -> None:
        self._sysfs_handle.remove()

    def get_parent_gpu(self) -> NvidiaGPU:
        from openforbc.gpu.nvidia.gpu import NvidiaGPU

        parent_dev = self._sysfs_handle.get_parent_gpu()
        if parent_dev.get_sriov_is_vf():
            parent_dev = parent_dev.get_sriov_physfn()

        return NvidiaGPU.from_pci_bus_id(
            self._sysfs_handle.get_parent_gpu().get_pci_bus_id()
        )
