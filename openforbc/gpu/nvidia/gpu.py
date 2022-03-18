from __future__ import annotations
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetCreatableVgpus,
    nvmlDeviceGetMigMode,
    nvmlDeviceGetName,
    nvmlDeviceGetSupportedVgpus,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetHandleByPciBusId,
    nvmlDeviceGetHandleByUUID,
    nvmlDeviceGetHostVgpuMode,
    nvmlDeviceGetPciInfo_v3,
    nvmlDeviceGetUUID,
    nvmlVgpuTypeGetMaxInstances,
    nvmlInit,
    nvmlShutdown,
)
from typing import TYPE_CHECKING
from uuid import UUID

from openforbc.gpu.generic import GPU, GPUPartition, GPUPartitionType
from openforbc.gpu.nvidia.nvml import nvml
from openforbc.gpu.nvidia.vgpu import (
    VGPUCreateException,
    VGPUInstance,
    VGPUMode,
    VGPUType,
)
from openforbc.gpu.nvidia.mig import MIGModeStatus
from openforbc.pci import PCIID
from openforbc.sysfs.gpu import GPUSysFsHandle

if TYPE_CHECKING:
    from openforbc.gpu.nvidia.nvml import NVMLDevice
    from typing import Sequence


class NvidiaGPU(GPU):
    _ref_count = 0

    @classmethod
    def from_nvml_handle_uuid(cls, dev: NVMLDevice, uuid: UUID) -> NvidiaGPU:
        return cls(
            dev,
            nvmlDeviceGetName(dev).decode(),
            uuid,
            PCIID.from_int(nvmlDeviceGetPciInfo_v3(dev).pciDeviceId),
            [VGPUType.from_id(id) for id in nvmlDeviceGetSupportedVgpus(dev)],
        )

    @classmethod
    def from_nvml_handle(cls, dev: NVMLDevice) -> NvidiaGPU:
        uuid = nvmlDeviceGetUUID(dev).decode()
        if uuid.startswith("GPU-"):
            uuid = uuid[len("GPU-") :]

        return cls.from_nvml_handle_uuid(dev, uuid)

    @classmethod
    def from_pci_bus_id(cls, bus_id: str) -> NvidiaGPU:
        with nvml():
            return cls.from_nvml_handle(nvmlDeviceGetHandleByPciBusId(bus_id.encode()))

    @classmethod
    def from_uuid(cls, uuid: UUID):
        with nvml():
            return cls.from_nvml_handle_uuid(
                nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode()), uuid
            )

    @classmethod
    def get_gpus(cls) -> list[NvidiaGPU]:
        with nvml():
            return [
                cls.from_nvml_handle(nvmlDeviceGetHandleByIndex(i))
                for i in range(nvmlDeviceGetCount())
            ]

    def __init__(
        self,
        nvml_dev: NVMLDevice,
        name: str,
        uuid: UUID,
        pciid: PCIID,
        supported_vgpu_types: list[VGPUType],
    ) -> None:
        super().__init__(name, uuid, pciid)

        if not NvidiaGPU._ref_count:
            nvmlInit()
        NvidiaGPU._ref_count += 1

        self._nvml_dev = nvml_dev
        self.supported_vgpu_types = supported_vgpu_types

    def __del__(self) -> None:
        NvidiaGPU._ref_count -= 1
        if not NvidiaGPU._ref_count:
            nvmlShutdown()

    def __repr__(self) -> str:
        return self.name

    def get_supported_types(self) -> Sequence[GPUPartitionType]:
        return self.supported_vgpu_types

    def get_creatable_types(self) -> Sequence[GPUPartitionType]:
        return self.get_creatable_vgpus()

    def create_partition(self, type: GPUPartitionType) -> GPUPartition:
        assert isinstance(type, VGPUType)
        return self.create_vgpu(type)

    def get_partitions(self) -> Sequence[GPUPartition]:
        return self.get_created_vgpus()

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
