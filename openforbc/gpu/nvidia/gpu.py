# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
"""
NVIDIA GPU partitioning management.

This module allows to partition a NVIDIA GPU using various supported technologies.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING
from uuid import UUID

from pynvml import (
    NVML_SUCCESS,
    NVML_GPU_INSTANCE_PROFILE_COUNT,
    NVMLError,
    NVMLError_NotSupported,  # type: ignore
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
    nvmlDeviceGetGpuInstanceById,
    nvmlDeviceGetGpuInstanceRemainingCapacity,
    nvmlDeviceSetMigMode,
    nvmlVgpuTypeGetMaxInstances,
    nvmlInit,
    nvmlShutdown,
)

from openforbc.error import PermissionException
from openforbc.gpu.generic import GPU
from openforbc.gpu.nvidia.nvml import NVMLGpuInstance, nvml_context
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
    from typing import ClassVar, Sequence

    from openforbc.gpu.generic import GPUvPartition, GPUvPartitionType
    from openforbc.gpu.nvidia.nvml import NVMLDevice

logger = getLogger(__name__)


class NvidiaGPUException(Exception):
    pass


class NvidiaGPUMIGDisabled(NvidiaGPUException):
    """Raised when trying to create a GI on a GPU with MIG mode disabled."""

    pass


class NvidiaGPUMIGCIProfileNotFound(NvidiaGPUException):
    pass


@dataclass
class NvidiaGPU(GPU):
    """
    NvidiaGPU represents an NVIDIA GPU.

    Allows to create/manage/delete both vGPUs and MIG instances.
    """

    _nvml_dev: NVMLDevice
    supported_vgpu_types: list[VGPUType]
    _ref_count: ClassVar[int] = 0

    @classmethod
    def from_nvml_handle_uuid(cls, dev: NVMLDevice, uuid: UUID) -> NvidiaGPU:
        """Create NvidiaGPU instance from NVML handle and its UUID."""
        return cls(
            nvmlDeviceGetName(dev),
            uuid,
            PCIID.from_int(nvmlDeviceGetPciInfo_v3(dev).pciDeviceId),
            dev,
            [VGPUType.from_id(id) for id in nvmlDeviceGetSupportedVgpus(dev)],
        )

    @classmethod
    def from_nvml_handle(cls, dev: NVMLDevice) -> NvidiaGPU:
        """Create NvidiaGPU instance from NVML handle."""
        uuid = nvmlDeviceGetUUID(dev)
        logger.debug("device #%s uuid: %s", str(dev), uuid)
        if uuid.startswith("GPU-"):
            uuid = uuid[len("GPU-") :]
        if uuid.startswith("MIG-"):
            uuid = uuid[len("MIG-") :]

        return cls.from_nvml_handle_uuid(dev, UUID(uuid))

    @classmethod
    def from_pci_bus_id(cls, bus_id: str) -> NvidiaGPU:
        """Create NvidiaGPU instance from the GPU's PCI bus id."""
        with nvml_context():
            return cls.from_nvml_handle(nvmlDeviceGetHandleByPciBusId(bus_id.encode()))

    @classmethod
    def from_uuid(cls, uuid: UUID) -> NvidiaGPU:
        """Create NvidiaGPU instance from GPU UUID ("GPU-<UUUID>")."""
        with nvml_context():
            return cls.from_nvml_handle_uuid(
                nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode()), uuid
            )

    @classmethod
    def get_gpus(cls) -> list[NvidiaGPU]:
        """Get all GPUs connected to this system."""
        logger.info("listing gpus")
        with nvml_context():
            return [
                cls.from_nvml_handle(nvmlDeviceGetHandleByIndex(i))
                for i in range(nvmlDeviceGetCount())
            ]

    def __post_init__(self) -> None:
        """Manage refcount for NVML library."""
        if not NvidiaGPU._ref_count:
            logger.info("initializing NVML")
            nvmlInit()
        NvidiaGPU._ref_count += 1

    def __del__(self) -> None:
        """Handle NvidiaGPU delete."""
        # update refcount and eventually shutdown NVML
        NvidiaGPU._ref_count -= 1
        if not NvidiaGPU._ref_count:
            logger.info("shutting down NVML")
            nvmlShutdown()

    def __repr__(self) -> str:
        """Pretty repr NvidiaGPU."""
        return f"{self.name} @{self.get_pci_bus_id()}"

    def get_supported_types(self) -> Sequence[GPUvPartitionType]:
        """Get GPU supported partition types."""
        return self.supported_vgpu_types

    def get_creatable_types(self) -> Sequence[GPUvPartitionType]:
        """Get GPU creatable partition types."""
        return self.get_creatable_vgpus()

    def create_partition(self, type: GPUvPartitionType) -> GPUvPartition:
        """Create a partition with specified type on this GPU."""
        assert isinstance(type, VGPUType)
        return self.create_vgpu(type)

    def get_partitions(self) -> Sequence[GPUvPartition]:
        """Get created partitions on this GPU."""
        return self.get_created_mdevs()

    def create_gpu_instance(
        self, profile_id: int, default_ci: bool = True
    ) -> GPUInstance:
        """Create a MIG GPU instance on this GPU."""
        logger.info(
            "creating GI with profile #%s on %s", profile_id, self.get_pci_bus_id()
        )
        handle = nvmlDeviceCreateGpuInstance(self._nvml_dev, profile_id)
        logger.debug("got handle #%s for GI", str(handle))
        instance = GPUInstance.from_nvml_handle_parent(handle, self)
        if not default_ci:
            return instance

        for profile in instance.get_compute_instance_profiles():
            if profile.slice_count == instance.profile.slice_count:
                logger.debug("default CIP is #%s", profile.id)
                instance.create_compute_instance(profile)
                return instance

        raise NvidiaGPUMIGCIProfileNotFound(
            f"Default CI profile not found for GIP #{profile_id}"
        )

    def create_gpu_instance_maybe(self, vgpu_type: VGPUType) -> None:
        """Create a MIG GI if needed for specified vGPU type."""
        logger.info("creating GI for vGPU type %s if necessary", vgpu_type)
        gpu_instances = self.get_gpu_instances()
        vgpu_types = [mdev.type for mdev in self.get_created_mdevs()] + [vgpu_type]
        for vgpu_type in vgpu_types:
            gpu_instance = next(
                (x for x in gpu_instances if x.profile.id == vgpu_type.gip_id), None
            )
            if gpu_instance:
                logger.debug(
                    "GI (%s) for vGPU type %s already present", gpu_instance, vgpu_type
                )
                gpu_instances.remove(gpu_instance)
            else:
                logger.debug(
                    "creating GI with profile id %s for vGPU type %s",
                    vgpu_type.gip_id,
                    vgpu_type,
                )
                self.create_gpu_instance(vgpu_type.gip_id)

    def create_vgpu(self, type: VGPUType) -> VGPUMdev:
        """Create vGPU with specified type on this GPU."""
        logger.info("creating %s vGPU", type)
        required_mig_status = (
            MIGModeStatus.ENABLE if type.is_mig else MIGModeStatus.DISABLE
        )

        if self.get_current_mig_status() != required_mig_status:
            logger.info("vGPU %s requires MIG mode change", type)
            if self.get_created_mdevs():
                logger.error("cannot change MIG mode")
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

            if (tmp := sysfs_handle.get_sriov_available_vf()) is None:
                logger.error("no available VFs on %s", self)
                raise VGPUCreateException(f"GPU {self} has no available VFs")

            sysfs_handle = tmp

        logger.info("creating mdev on %s for %s vGPU", sysfs_handle, type)
        mdev_type = type.get_mdev_type()
        logger.debug("using mdev type %s for %s vGPU", mdev_type, type)

        if not sysfs_handle.get_mdev_type_available(mdev_type):
            logger.error(
                "cannot create %s mdev for %s vGPU on %s", mdev_type, type, self
            )
            raise VGPUCreateException(f"GPU {self} can't create any {type}")

        return VGPUMdev.from_sysfs_handle(sysfs_handle.create_mdev(mdev_type))

    def destroy_gpu_instance_maybe(self, profile: GPUInstanceProfile) -> None:
        """Destroy MIG GI with specified profile if not needed anymore."""
        logger.info("destroying GI for %s if not necessary", profile)
        gpu_instances = [x for x in self.get_gpu_instances() if x.profile == profile]
        if not gpu_instances:
            logger.debug("no %s GIs present, not destroying any", profile)
            return

        mdevs = [
            mdev for mdev in self.get_created_mdevs() if mdev.type.gip_id == profile.id
        ]
        logger.debug("GIP %s is used by mdevs: %s", profile, mdevs)
        for vgpu in self.get_active_vgpus():
            logger.debug("removing GIs used by vGPU %s", vgpu)
            gpu_instances = [
                instance
                for instance in gpu_instances
                if instance.id != vgpu.get_gpu_instance_id()
            ]
            mdev = next(mdev for mdev in mdevs if mdev.type == vgpu.type)
            mdevs.remove(mdev)

        logger.debug(
            "%s GIs present with %s mdevs created for GIP %s",
            len(gpu_instances),
            len(mdevs),
            profile,
        )
        if len(gpu_instances) > len(mdevs):
            gpu_instances.pop().destroy()

    def get_active_vgpus(self) -> list[VGPUInstance]:
        """Get active (VM is running) vGPUs."""
        logger.info("getting active vGPUs for %s", self)
        return [
            VGPUInstance.from_nvml(dev)
            for dev in nvmlDeviceGetActiveVgpus(self._nvml_dev)
        ]

    def get_creatable_vgpus(self) -> list[VGPUType]:
        """Get creatable vGPU types for this GPU."""
        logger.info("geting creatable vGPUs for %s", self)
        if not self.get_created_mdevs():
            logger.debug("no mdevs, all types can be created")
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
        """Get created MDEVs on this GPU."""
        logger.info("getting created mdevs for %s", self)
        vgpus: list[VGPUMdev] = []
        sysfs_handle = GPUSysFsHandle.from_gpu(self)

        if sysfs_handle.get_mdev_supported():
            logger.debug("getting direct mdev devices")
            vgpus.extend(
                VGPUMdev.from_sysfs_handle(dev)
                for dev in sysfs_handle.get_mdev_devices()
            )

        if sysfs_handle.get_sriov_active():
            for vf in sysfs_handle.get_sriov_vfs():
                logger.debug("getting mdevs for virtfn #%s", vf)
                vgpus.extend(
                    VGPUMdev.from_sysfs_handle(dev) for dev in vf.get_mdev_devices()
                )

        return vgpus

    def get_current_mig_status(self) -> MIGModeStatus:
        """Get current MIG status."""
        current, _ = nvmlDeviceGetMigMode(self._nvml_dev)
        return MIGModeStatus(current)

    def get_pending_mig_status(self) -> MIGModeStatus:
        """Get pending (needs device reset) MIG status."""
        _, pending = nvmlDeviceGetMigMode(self._nvml_dev)
        return MIGModeStatus(pending)

    def get_gpu_instance_remaining_capacity(self, profile: GPUInstanceProfile) -> int:
        """Get remaining capacity for the specified GI profile."""
        return nvmlDeviceGetGpuInstanceRemainingCapacity(self._nvml_dev, profile.id)

    def get_gpu_instance_by_id(self, id: int) -> GPUInstance:
        """Get a GPUInstance by its ID."""
        return GPUInstance.from_nvml_handle_parent(
            nvmlDeviceGetGpuInstanceById(self._nvml_dev, id), self
        )

    def get_gpu_instances(self) -> list[GPUInstance]:
        """Get all GIs created on this GPU."""
        logger.info("getting GIs for %s", self)
        from ctypes import byref, c_uint

        if self.get_current_mig_status() != MIGModeStatus.ENABLE:
            raise NvidiaGPUMIGDisabled

        instances = []
        for profile in range(NVML_GPU_INSTANCE_PROFILE_COUNT):
            logger.debug("trying with GIP #%s", profile)
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

    def get_pci_bus_id(self) -> str:
        """Get this GPU's PCI bus id."""
        return nvmlDeviceGetPciInfo_v3(self._nvml_dev).busIdLegacy.lower()

    def get_sysfs_handle(self) -> GPUSysFsHandle:
        """Get a GPUSysFsHandle for this GPU."""
        return GPUSysFsHandle.from_gpu(self)

    def get_supported_gpu_instance_profiles(self) -> list[GPUInstanceProfile]:
        """Get all supported MIG GPU instance profiles."""
        logger.info("getting supported GIPs for %s", self)
        profiles = []
        for i in range(NVML_GPU_INSTANCE_PROFILE_COUNT):
            logger.debug("checking support of GIP #%s by %s", i, self)
            try:
                profiles.append(GPUInstanceProfile.from_idx(i, self))
            except NVMLError_NotSupported:
                continue

        return profiles

    def get_vgpu_mode(self) -> VGPUMode:
        """Get the vGPU mode of this GPU."""
        mode = nvmlDeviceGetHostVgpuMode(self._nvml_dev)
        return VGPUMode(mode)

    def get_vgpu_max_instances(self, type: VGPUType) -> int:
        """Get the maximum number of instances for the specified vGPU type."""
        return nvmlVgpuTypeGetMaxInstances(self._nvml_dev, type.id)

    def get_vf_available_vgpu(self, vf_num: int, vgpu_type: VGPUType) -> bool:
        """Return True if a virtual function is available to create a vGPU."""
        path = (
            f"/sys/bus/pci/devices/{self.get_pci_bus_id()}/virtfn{vf_num}"
            f"/mdev_supported_types/nvidia-{vgpu_type.id}/available_instances"
        )
        with open(path) as f:
            return bool(int(f.read()))

    def get_vf_available_vgpus(self, vf_num: int) -> list[VGPUType]:
        """Get available vGPU types on a specific virtual function."""
        return [
            x
            for x in self.get_creatable_vgpus()
            if self.get_vf_available_vgpu(vf_num, x)
        ]

    def set_mig_mode(self, mode: MIGModeStatus) -> None:
        """Change the MIG mode of this GPU."""
        logger.info("setting MIG mode %s for %s", mode, self)
        status = nvmlDeviceSetMigMode(self._nvml_dev, mode)
        if status != NVML_SUCCESS:
            raise NVMLError(status)

    def set_sriov_enable(self, enable: bool) -> None:
        """Enable or disable SRIOV on this GPU."""
        from subprocess import run, PIPE

        logger.info(
            "%s sriov for %s @ %s",
            "enabling" if enable else "disabling",
            self.pciid,
            self,
        )

        nvmlShutdown()
        p = run(
            [
                "/usr/lib/nvidia/sriov-manage",
                "-e" if enable else "-d",
                self.get_pci_bus_id(),
            ],
            stdout=PIPE,
            stderr=PIPE,
        )
        nvmlInit()
        self._nvml_dev = nvmlDeviceGetHandleByUUID(f"GPU-{self.uuid}".encode())
        logger.debug("sriov-manage: stderr: %s", p.stderr)

        if "Permission denied" in p.stderr.decode():
            raise PermissionException(
                "Elevated permissions needed to modify sriov status"
            )

        p.check_returncode()
