from __future__ import annotations
from pynvml import (
    NVML_SUCCESS,
    NVML_GPU_INSTANCE_PROFILE_COUNT,
    NVMLError,
    NVMLError_NotSupported,
    nvmlDeviceCreateGpuInstance,
    nvmlDeviceGetCount,
    nvmlDeviceGetCreatableVgpus,
    nvmlDeviceGetMigMode,
    nvmlDeviceGetGpuInstanceProfileInfo,
    nvmlDeviceGetGpuInstances,
    nvmlDeviceGetName,
    nvmlDeviceGetActiveVgpus,
    nvmlDeviceGetSupportedVgpus,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetHandleByPciBusId,
    nvmlDeviceGetHandleByUUID,
    nvmlDeviceGetHostVgpuMode,
    nvmlDeviceGetPciInfo_v3,
    nvmlDeviceGetUUID,
    nvmlDeviceGetGpuInstanceRemainingCapacity,
    nvmlDeviceSetMigMode,
    nvmlVgpuTypeGetMaxInstances,
    nvmlInit,
    nvmlShutdown,
)
from typing import TYPE_CHECKING

from openforbc.gpu.generic import GPU
from openforbc.gpu.nvidia.nvml import NVMLGpuInstance, nvml
from openforbc.gpu.nvidia.vgpu import (
    VGPUCreateException,
    VGPUInstance,
    VGPUMdev,
    VGPUMode,
    VGPUType,
)
from openforbc.gpu.nvidia.mig import GPUInstance, GPUInstanceProfile, MIGModeStatus
from openforbc.pci import PCIID
from openforbc.sysfs.gpu import GPUSysFsHandle

if TYPE_CHECKING:
    from typing import Sequence
    from uuid import UUID

    from openforbc.gpu.generic import GPUPartition, GPUPartitionType
    from openforbc.gpu.nvidia.nvml import NVMLDevice


class NvidiaGPUMIGDisabled(Exception):
    pass


class NvidiaGPU(GPU):
    _ref_count = 0

    @classmethod
    def from_nvml_handle_uuid(cls, dev: NVMLDevice, uuid: UUID) -> NvidiaGPU:
        return cls(
            dev,
            nvmlDeviceGetName(dev),
            uuid,
            PCIID.from_int(nvmlDeviceGetPciInfo_v3(dev).pciDeviceId),
            [VGPUType.from_id(id) for id in nvmlDeviceGetSupportedVgpus(dev)],
        )

    @classmethod
    def from_nvml_handle(cls, dev: NVMLDevice) -> NvidiaGPU:
        uuid = nvmlDeviceGetUUID(dev)
        if uuid.startswith("GPU-"):
            uuid = uuid[len("GPU-") :]
        if uuid.startswith("MIG-"):
            uuid = uuid[len("MIG-") :]

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
        return f"{self.name} @{self.get_pci_id()}"

    def get_supported_types(self) -> Sequence[GPUPartitionType]:
        return self.supported_vgpu_types

    def get_creatable_types(self) -> Sequence[GPUPartitionType]:
        return self.get_creatable_vgpus()

    def create_partition(self, type: GPUPartitionType) -> GPUPartition:
        assert isinstance(type, VGPUType)
        return self.create_vgpu(type)

    def get_partitions(self) -> Sequence[GPUPartition]:
        return self.get_created_mdevs()

    def create_gpu_instance(self, profile_id: int) -> GPUInstance:
        handle = nvmlDeviceCreateGpuInstance(self._nvml_dev, profile_id)
        instance = GPUInstance.from_nvml_handle_parent(handle, self)
        for profile in instance.get_compute_instance_profiles():
            if profile.slice_count == instance.profile.slice_count:
                instance.create_compute_instance(profile)
                break

        return handle

    def create_gpu_instance_maybe(self, vgpu_type: VGPUType) -> None:
        gpu_instances = self.get_gpu_instances()
        vgpu_types = [mdev.type for mdev in self.get_created_mdevs()] + [vgpu_type]
        for vgpu_type in vgpu_types:
            gpu_instance = next(
                (x for x in gpu_instances if x.profile.id == vgpu_type.gip_id), None
            )
            if gpu_instance:
                gpu_instances.remove(gpu_instance)
            else:
                self.create_gpu_instance(vgpu_type.gip_id)

    def create_vgpu(self, type: VGPUType) -> VGPUMdev:
        required_mig_status = (
            MIGModeStatus.ENABLE if type.is_mig else MIGModeStatus.DISABLE
        )

        if self.get_current_mig_status() != required_mig_status:
            if self.get_created_mdevs():
                raise VGPUCreateException(
                    f"vGPU type {type} requires MIG mode change, "
                    "but some mdevs are present"
                )
            self.set_mig_mode(required_mig_status)

        if type.is_mig:
            self.create_gpu_instance_maybe(type)

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

        return VGPUMdev.from_sysfs_handle(sysfs_handle.create_mdev(mdev_type))

    def destroy_gpu_instance_maybe(self, profile: GPUInstanceProfile) -> None:
        gpu_instances = [x for x in self.get_gpu_instances() if x.profile == profile]
        if not gpu_instances:
            return
        mdevs = [
            mdev for mdev in self.get_created_mdevs() if mdev.type.gip_id == profile.id
        ]
        for vgpu in self.get_active_vgpus():
            gpu_instances = [
                instance
                for instance in gpu_instances
                if instance.id != vgpu.get_gpu_instance_id()
            ]
            mdev = next(mdev for mdev in mdevs if mdev.type == vgpu.type)
            mdevs.remove(mdev)

        if len(gpu_instances) > len(mdevs):
            gpu_instances.pop().destroy()

    def get_active_vgpus(self) -> list[VGPUInstance]:
        return [
            VGPUInstance.from_nvml(dev)
            for dev in nvmlDeviceGetActiveVgpus(self._nvml_dev)
        ]

    def get_creatable_vgpus(self) -> list[VGPUType]:
        if not self.get_created_mdevs():
            return self.supported_vgpu_types

        supported_mig_vgpu_types = [
            type for type in self.supported_vgpu_types if type.is_mig
        ]

        return (
            [
                type
                for type in supported_mig_vgpu_types
                if self.get_gpu_instance_remaining_capacity(
                    GPUInstanceProfile.from_id(type.gip_id, self)
                )
            ]
            if self.get_current_mig_status() == MIGModeStatus.ENABLE
            else [
                VGPUType.from_id(id)
                for id in nvmlDeviceGetCreatableVgpus(self._nvml_dev)
            ]
        )

    def get_created_mdevs(self) -> list[VGPUMdev]:
        vgpus: list[VGPUMdev] = []
        sysfs_handle = GPUSysFsHandle.from_gpu(self)

        if sysfs_handle.get_mdev_supported():
            vgpus.extend(
                VGPUMdev.from_sysfs_handle(dev)
                for dev in sysfs_handle.get_mdev_devices()
            )

        if sysfs_handle.get_sriov_active():
            for vf in sysfs_handle.get_sriov_vfs():
                vgpus.extend(
                    VGPUMdev.from_sysfs_handle(dev) for dev in vf.get_mdev_devices()
                )

        return vgpus

    def get_current_mig_status(self) -> MIGModeStatus:
        current, _ = nvmlDeviceGetMigMode(self._nvml_dev)
        return MIGModeStatus(current)

    def get_pending_mig_status(self) -> MIGModeStatus:
        _, pending = nvmlDeviceGetMigMode(self._nvml_dev)
        return MIGModeStatus(pending)

    def get_gpu_instance_remaining_capacity(self, profile: GPUInstanceProfile) -> int:
        return nvmlDeviceGetGpuInstanceRemainingCapacity(self._nvml_dev, profile.id)

    def get_gpu_instances(self) -> list[GPUInstance]:
        from ctypes import byref, c_uint

        if self.get_current_mig_status() != MIGModeStatus.ENABLE:
            raise NvidiaGPUMIGDisabled

        instances = []
        for profile in range(NVML_GPU_INSTANCE_PROFILE_COUNT):
            try:
                info = nvmlDeviceGetGpuInstanceProfileInfo(self._nvml_dev, profile)
            except NVMLError_NotSupported:
                continue

            InstancesArray = NVMLGpuInstance * info.instanceCount
            c_count = c_uint()
            c_profile_instances = InstancesArray()
            nvmlDeviceGetGpuInstances(
                self._nvml_dev, info.id, c_profile_instances, byref(c_count)
            )

            for i in range(c_count.value):
                instances.append(
                    GPUInstance.from_nvml_handle_parent(c_profile_instances[i], self)
                )

        return instances

    def get_pci_id(self) -> str:
        return nvmlDeviceGetPciInfo_v3(self._nvml_dev).busIdLegacy.lower()

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

    def set_mig_mode(self, mode: MIGModeStatus) -> None:
        status = nvmlDeviceSetMigMode(self._nvml_dev, mode)
        if status != NVML_SUCCESS:
            raise NVMLError(status)

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
