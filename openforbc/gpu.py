from contextlib import contextmanager
from ctypes import pointer
from enum import Enum
from typing import List
from pynvml import (
    nvmlInit,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlShutdown,
)
from pynvml.nvml import (
    NVML_DEVICE_MIG_DISABLE,
    NVML_DEVICE_MIG_ENABLE,
    NVML_HOST_VGPU_MODE_NON_SRIOV,
    NVML_HOST_VGPU_MODE_SRIOV,
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
    struct_c_nvmlDevice_t,
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
    def from_id(cls, id: int) -> "VGPUType":
        with nvml():
            return VGPUType(
                id,
                nvmlVgpuTypeGetName(id).decode(),
                nvmlVgpuTypeGetClass(id).decode(),
                nvmlVgpuTypeGetFramebufferSize(id),
                nvmlVgpuTypeGetGpuInstanceProfileId(id),
            )


class MIGModeStatus(Enum):
    DISABLE = NVML_DEVICE_MIG_DISABLE
    ENABLE = NVML_DEVICE_MIG_ENABLE


class VGPUMode(Enum):
    NON_SRIOV = NVML_HOST_VGPU_MODE_NON_SRIOV
    SRIOV = NVML_HOST_VGPU_MODE_SRIOV


class GPU:
    _ref_count = 0

    @classmethod
    def from_nvml_handle(cls, dev: "pointer[struct_c_nvmlDevice_t]") -> "GPU":
        return cls(
            dev,
            nvmlDeviceGetName(dev).decode(),
            [VGPUType.from_id(id) for id in nvmlDeviceGetSupportedVgpus(dev)],
        )

    @classmethod
    def get_gpus(cls):
        with nvml():
            return [
                cls.from_nvml_handle(nvmlDeviceGetHandleByIndex(i))
                for i in range(nvmlDeviceGetCount())
            ]

    def __init__(
        self,
        nvml_dev: "pointer[struct_c_nvmlDevice_t]",
        name: str,
        supported_vgpu_types: "List[VGPUType]",
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

    def get_creatable_vgpus(self) -> "List[VGPUType]":
        return [
            VGPUType.from_id(id) for id in nvmlDeviceGetCreatableVgpus(self._nvml_dev)
        ]

    def get_current_mig_status(self) -> MIGModeStatus:
        current, _ = nvmlDeviceGetMigMode(self._nvml_dev)
        return MIGModeStatus(current)

    def get_pending_mig_status(self) -> MIGModeStatus:
        _, pending = nvmlDeviceGetMigMode(self._nvml_dev)
        return MIGModeStatus(pending)

    def get_pci_id(self) -> str:
        return nvmlDeviceGetPciInfo_v3(self._nvml_dev).busIdLegacy.decode()

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

    def get_vf_available_vgpus(self, vf_num: int) -> "List[VGPUType]":
        return [
            x
            for x in self.get_creatable_vgpus()
            if self.get_vf_available_vgpu(vf_num, x)
        ]

    def get_vf_num(self) -> int:
        path = f"/sys/bus/pci/devices/{self.get_pci_id()}/sriov_numvfs"
        with open(path) as f:
            return int(f.read())

    def set_vf_enable(self, enable: bool) -> None:
        from subprocess import run

        p = run(
            [
                "/usr/lib/nvidia/sriov-manage",
                "-e" if enable else "-d",
                self.get_pci_id(),
            ]
        )
        p.check_returncode()
