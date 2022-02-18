from __future__ import annotations
from typing import TYPE_CHECKING

from openforbc.sysfs import GPUSysFsHandle, MdevSysFsHandle

if TYPE_CHECKING:
    from ctypes import pointer
    from uuid import UUID

from contextlib import contextmanager
from enum import Enum
from pynvml import (
    NVML_DEVICE_MIG_DISABLE,
    NVML_DEVICE_MIG_ENABLE,
    NVML_HOST_VGPU_MODE_NON_SRIOV,
    NVML_HOST_VGPU_MODE_SRIOV,
    struct_c_nvmlDevice_t,
    nvmlInit,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetHandleByPciBusId,
    nvmlDeviceGetCreatableVgpus,
    nvmlDeviceGetHostVgpuMode,
    nvmlDeviceGetMigMode,
    nvmlDeviceGetName,
    nvmlDeviceGetPciInfo_v3,
    nvmlDeviceGetSupportedVgpus,
    nvmlVgpuTypeGetClass,
    nvmlVgpuTypeGetFramebufferSize,
    nvmlVgpuTypeGetGpuInstanceProfileId,
    nvmlVgpuTypeGetMaxInstances,
    nvmlVgpuTypeGetName,
    nvmlShutdown,
)


@contextmanager
def nvml():
    nvmlInit()
    yield
    nvmlShutdown()


class VGPUType:
    def __init__(
        self, id: int, name: str, vgpu_class: str, fb_size: int, gip_id: int
    ) -> None:
        self.id = id
        self.name = name
        self.vgpu_class = vgpu_class
        self.fb_size = fb_size
        self.is_mig = gip_id != 0xFFFFFFFF
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


class MIGModeStatus(Enum):
    DISABLE = NVML_DEVICE_MIG_DISABLE
    ENABLE = NVML_DEVICE_MIG_ENABLE


class VGPUMode(Enum):
    NON_SRIOV = NVML_HOST_VGPU_MODE_NON_SRIOV
    SRIOV = NVML_HOST_VGPU_MODE_SRIOV


class VGPUCreateException(Exception):
    pass


class VGPUTypeException(Exception):
    pass


class VGPUInstance:
    def __init__(
        self, sysfs_handle: MdevSysFsHandle, uuid: UUID, type: VGPUType
    ) -> None:
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

    def get_parent_gpu(self) -> GPU:
        parent_dev = self._sysfs_handle.get_parent_gpu()
        if parent_dev.get_sriov_is_vf():
            parent_dev = parent_dev.get_sriov_physfn()

        return GPU.from_pci_bus_id(self._sysfs_handle.get_parent_gpu().get_pci_bus_id())


class GPU:
    _ref_count = 0

    @classmethod
    def from_nvml_handle(cls, dev: pointer[struct_c_nvmlDevice_t]) -> GPU:
        return cls(
            dev,
            nvmlDeviceGetName(dev).decode(),
            [VGPUType.from_id(id) for id in nvmlDeviceGetSupportedVgpus(dev)],
        )

    @classmethod
    def from_pci_bus_id(cls, bus_id: str) -> GPU:
        with nvml():
            return cls.from_nvml_handle(nvmlDeviceGetHandleByPciBusId(bus_id.encode()))

    @classmethod
    def get_gpus(cls):
        with nvml():
            return [
                cls.from_nvml_handle(nvmlDeviceGetHandleByIndex(i))
                for i in range(nvmlDeviceGetCount())
            ]

    def __init__(
        self,
        nvml_dev: pointer[struct_c_nvmlDevice_t],
        name: str,
        supported_vgpu_types: list[VGPUType],
    ) -> None:
        if not GPU._ref_count:
            nvmlInit()
        GPU._ref_count += 1

        self._nvml_dev = nvml_dev
        self.name = name
        self.supported_vgpu_types = supported_vgpu_types

    def __del__(self) -> None:
        GPU._ref_count -= 1
        if not GPU._ref_count:
            nvmlShutdown()

    def __repr__(self) -> str:
        return self.name

    def create_vgpu(self, type: VGPUType) -> VGPUInstance:
        sysfs_handle = self.get_sysfs_handle()
        if self.get_vgpu_mode() == VGPUMode.SRIOV:
            if not sysfs_handle.get_sriov_active():
                self.set_sriov_enable(True)

            tmp = sysfs_handle.get_sriov_available_vf()
            if tmp is None:
                raise VGPUCreateException(f"GPU {self} has no available VFs")

            sysfs_handle = tmp

        mdev_type = type.get_mdev_type()

        if not sysfs_handle.get_mdev_type_available(mdev_type):
            raise VGPUCreateException(f"GPU {self} can't create any {type}")

        return VGPUInstance.from_sysfs_handle(sysfs_handle.create_mdev(mdev_type))

    def get_creatable_vgpus(self) -> list[VGPUType]:
        return [
            VGPUType.from_id(id) for id in nvmlDeviceGetCreatableVgpus(self._nvml_dev)
        ]

    def get_created_vgpus(self) -> list[VGPUInstance]:
        vgpus: list[VGPUInstance] = []
        sysfs_handle = GPUSysFsHandle.from_gpu(self)

        if sysfs_handle.get_mdev_supported():
            vgpus.extend(
                VGPUInstance.from_sysfs_handle(dev)
                for dev in sysfs_handle.get_mdev_devices()
            )

        if sysfs_handle.get_sriov_active():
            for vf in sysfs_handle.get_sriov_vfs():
                vgpus.extend(
                    VGPUInstance.from_sysfs_handle(dev) for dev in vf.get_mdev_devices()
                )

        return vgpus

    def get_current_mig_status(self) -> MIGModeStatus:
        current, _ = nvmlDeviceGetMigMode(self._nvml_dev)
        return MIGModeStatus(current)

    def get_pending_mig_status(self) -> MIGModeStatus:
        _, pending = nvmlDeviceGetMigMode(self._nvml_dev)
        return MIGModeStatus(pending)

    def get_pci_id(self) -> str:
        return nvmlDeviceGetPciInfo_v3(self._nvml_dev).busIdLegacy.decode()

    def get_sysfs_handle(self) -> GPUSysFsHandle:
        return GPUSysFsHandle.from_gpu(self)

    def get_vgpu_mode(self) -> VGPUMode:
        mode = nvmlDeviceGetHostVgpuMode(self._nvml_dev)
        return VGPUMode(mode)

    def get_vgpu_max_instances(self, type: VGPUType) -> int:
        return nvmlVgpuTypeGetMaxInstances(self._nvml_dev, type.id)

    def get_vf_available_vgpu(self, vf_num: int, vgpu_type: VGPUType) -> bool:
        path = (
            f"/sys/bus/pci/devices/{self.get_pci_id()}/virtfn{vf_num}"
            f"/mdev_supported_types/nvidia-{vgpu_type.id}/available_instances"
        )
        with open(path) as f:
            return bool(int(f.read()))

    def get_vf_available_vgpus(self, vf_num: int) -> list[VGPUType]:
        return [
            x
            for x in self.get_creatable_vgpus()
            if self.get_vf_available_vgpu(vf_num, x)
        ]

    def set_sriov_enable(self, enable: bool) -> None:
        from subprocess import run

        p = run(
            [
                "/usr/lib/nvidia/sriov-manage",
                "-e" if enable else "-d",
                self.get_pci_id(),
            ]
        )
        p.check_returncode()
