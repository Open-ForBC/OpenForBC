# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from pynvml import (
    nvmlVgpuInstanceGetGpuInstanceId,
    nvmlVgpuInstanceGetType,
    nvmlVgpuInstanceGetUUID,
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
from openforbc.gpu.nvidia.mig import GPUInstanceProfile
from openforbc.gpu.nvidia.nvml import nvml
from openforbc.sysfs.mdev import MdevSysFsHandle

if TYPE_CHECKING:
    from openforbc.gpu.nvidia.gpu import NvidiaGPU
    from openforbc.gpu.nvidia.nvml import NVMLVGPUInstance

INVALID_GPU_INSTANCE_PROFILE_ID = 0xFFFFFFFF


class VGPUMode(IntEnum):
    NON_SRIOV = NVML_HOST_VGPU_MODE_NON_SRIOV
    SRIOV = NVML_HOST_VGPU_MODE_SRIOV


class VGPUCreateException(Exception):
    pass


class VGPUTypeException(Exception):
    pass


@dataclass
class VGPUType(GPUPartitionType):
    vgpu_class: str
    is_mig: bool
    gip_id: int

    def __init__(
        self, id: int, name: str, memory: int, vgpu_class: str, gip_id: int
    ) -> None:
        is_mig = gip_id != INVALID_GPU_INSTANCE_PROFILE_ID
        super().__init__(
            name,
            id,
            GPUPartitionTechnology.NVIDIA_VGPU_MIG
            if is_mig
            else GPUPartitionTechnology.NVIDIA_VGPU_TIMESHARED,
            memory,
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
            return cls(
                id,
                nvmlVgpuTypeGetName(id),
                nvmlVgpuTypeGetFramebufferSize(id),
                nvmlVgpuTypeGetClass(id),
                nvmlVgpuTypeGetGpuInstanceProfileId(id),
            )

    def get_mdev_type(self) -> str:
        return f"nvidia-{self.id}"


@dataclass
class VGPUInstance:
    type: VGPUType
    uuid: UUID

    _dev: NVMLVGPUInstance = field(repr=False)

    @classmethod
    def from_nvml(cls, nvml_vgpu_instance: NVMLVGPUInstance) -> VGPUInstance:
        return cls(
            VGPUType.from_id(nvmlVgpuInstanceGetType(nvml_vgpu_instance)),
            UUID(nvmlVgpuInstanceGetUUID(nvml_vgpu_instance)),
            nvml_vgpu_instance,
        )

    def get_gpu_instance_id(self) -> int:
        return nvmlVgpuInstanceGetGpuInstanceId(self._dev)


@dataclass
class VGPUMdev(GPUPartition):
    """
    A VGPUMdev represents a VFIO Mediated device with a vGPU type.

    We associate mdevs with GPU partitions in OpenForBC since the usage model
    supports manual usage, in which a use may create a GPU partition without
    starting a VM at the same time.

    This device might have an actual vGPU instance associated with it or NOT.
    Beware that in some cases mdev devices referring to non-actually-creatable
    vGPUs could exist and trying to start a VM with these attached will result
    in a failure.
    """

    type: VGPUType

    _sysfs_handle: MdevSysFsHandle

    # def __init__(
    #     self, sysfs_handle: MdevSysFsHandle, uuid: UUID, type: VGPUType
    # ) -> None:
    #     super().__init__(uuid, type)

    #     self._sysfs_handle = sysfs_handle
    #     self.uuid = uuid
    #     self.type = type

    def __repr__(self) -> str:
        return f"{self.type} {self.uuid} on {self._sysfs_handle.get_parent_gpu()}"

    @classmethod
    def from_sysfs_handle(cls, sysfs_handle: MdevSysFsHandle) -> VGPUMdev:
        UUID
        uuid = sysfs_handle.get_uuid()
        mdev_type = sysfs_handle.get_mdev_type()
        if not mdev_type.startswith("nvidia-"):
            raise VGPUTypeException(f"VGPU type not recognized: {mdev_type}")
        type = VGPUType.from_id(int(mdev_type[len("nvidia-") :]))

        return cls(uuid, type, sysfs_handle)

    @classmethod
    def from_mdev_uuid(cls, uuid: UUID) -> VGPUMdev:
        return cls.from_sysfs_handle(MdevSysFsHandle.from_uuid(uuid))

    def destroy(self) -> None:
        gpu = self.get_parent_gpu()
        self._sysfs_handle.remove()

        if self.type.is_mig:
            gpu.destroy_gpu_instance_maybe(
                GPUInstanceProfile.from_id(self.type.gip_id, gpu)
            )

    def get_parent_gpu(self) -> NvidiaGPU:
        from openforbc.gpu.nvidia.gpu import NvidiaGPU

        parent_dev = self._sysfs_handle.get_parent_gpu()
        if parent_dev.get_sriov_is_vf():
            parent_dev = parent_dev.get_sriov_physfn()

        return NvidiaGPU.from_pci_bus_id(parent_dev.get_pci_bus_id())
