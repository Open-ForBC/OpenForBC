# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
"""
NVIDIA Multi-Instance GPU management.

MIG allows to create physically separate partitions on a supported (Ampere or later) GPU.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from logging import getLogger
from typing import TYPE_CHECKING

from pynvml import (
    NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED,
    NVML_COMPUTE_INSTANCE_PROFILE_COUNT,
    NVML_DEVICE_MIG_DISABLE,
    NVML_DEVICE_MIG_ENABLE,
    NVML_GPU_INSTANCE_PROFILE_COUNT,
    NVMLError_NotSupported,  # type: ignore
    nvmlComputeInstanceDestroy,
    nvmlComputeInstanceGetInfo,
    nvmlDeviceGetDeviceHandleFromMigDeviceHandle,
    nvmlDeviceGetGpuInstanceProfileInfo,
    nvmlGpuInstanceDestroy,
    nvmlGpuInstanceGetInfo,
    nvmlGpuInstanceCreateComputeInstance,
    nvmlGpuInstanceGetComputeInstanceProfileInfo,
    nvmlGpuInstanceGetComputeInstances,
)

from openforbc.gpu.nvidia.nvml import NVMLComputeInstance


if TYPE_CHECKING:
    from openforbc.gpu.nvidia.gpu import NvidiaGPU
    from openforbc.gpu.nvidia.nvml import NVMLGpuInstance


logger = getLogger(__name__)


class NvidiaMIGGIPNotFound(Exception):
    """Raised when a GI profile cannot be found for a GI."""

    pass


class NvidiaMIGCIPNotFound(Exception):
    """Raised when a CI cannot be found on parent GI."""

    pass


class MIGModeStatus(IntEnum):
    """NVIDIA MIG mode status (either enabled or disabled)."""

    DISABLE = NVML_DEVICE_MIG_DISABLE
    ENABLE = NVML_DEVICE_MIG_ENABLE


@dataclass
class GPUInstanceProfile:
    """A MIG GPU instance profile."""

    id: int
    slice_count: int
    memory_size: int
    media_engine: bool

    def __str__(self) -> str:
        """Pretty repr for GIP."""
        return f"{self.slice_count}g.{round(self.memory_size / 1000)}gb" + (
            "+me" if self.slice_count < 7 and self.media_engine else ""
        )

    @classmethod
    def from_idx(cls, idx: int, gpu: NvidiaGPU) -> GPUInstanceProfile:
        """Build a GPUInstanceProfile from its index (NVML_GPU_ISNTANCE_PROFILE_*)."""
        info = nvmlDeviceGetGpuInstanceProfileInfo(gpu._nvml_dev, idx)
        return cls(info.id, info.sliceCount, info.memorySizeMB, bool(info.jpegCount))

    @classmethod
    def from_id(cls, id: int, gpu: NvidiaGPU) -> GPUInstanceProfile:
        """Construct a GPUInstanceProfile from its ID."""
        for i in range(NVML_GPU_INSTANCE_PROFILE_COUNT):
            try:
                info = nvmlDeviceGetGpuInstanceProfileInfo(gpu._nvml_dev, i)
            except NVMLError_NotSupported:
                continue
            if info.id == id:
                return cls(
                    info.id, info.sliceCount, info.memorySizeMB, bool(info.jpegCount)
                )

        raise NvidiaMIGGIPNotFound


@dataclass
class GPUInstance:
    """
    A GPUInstance represents an NVIDIA MIG GPU instance.

    A GPU instance is an hardware partition of a GPU, with dedicated physical
    resources.
    """

    _nvml_dev: NVMLGpuInstance
    id: int
    profile: GPUInstanceProfile
    parent: NvidiaGPU

    def __str__(self) -> str:
        """Pretty repr for GI."""
        return f"{self.profile} #{self.id} @({self.parent})"

    @classmethod
    def from_nvml_handle_parent(
        cls, nvml_device: NVMLGpuInstance, parent: NvidiaGPU
    ) -> GPUInstance:
        """Construct a GPUInstance from its NVML handle and the parent NvidiaGPU."""
        info = nvmlGpuInstanceGetInfo(nvml_device)
        return cls(
            nvml_device,
            info.id,
            GPUInstanceProfile.from_id(info.profileId, parent),
            parent,
        )

    @classmethod
    def from_nvml_handle(cls, dev: NVMLGpuInstance) -> GPUInstance:
        """Construct a GPUInstance from its NVMLHandle."""
        from openforbc.gpu.nvidia.gpu import NvidiaGPU

        return cls.from_nvml_handle_parent(
            dev,
            NvidiaGPU.from_nvml_handle(
                nvmlDeviceGetDeviceHandleFromMigDeviceHandle(dev)
            ),
        )

    def create_compute_instance(
        self, profile: ComputeInstanceProfile
    ) -> ComputeInstance:
        """Create a CI with given profile on this GI."""
        logger.info("creating %s CI on %s", profile, self)
        return ComputeInstance.from_nvml_handle_parent(
            nvmlGpuInstanceCreateComputeInstance(self._nvml_dev, profile.id), self
        )

    def destroy(self) -> None:
        """Destroy this GPU instance."""
        logger.info("destroying GI %s", self)
        for instance in self.get_compute_instances():
            instance.destroy()
        nvmlGpuInstanceDestroy(self._nvml_dev)

    def get_compute_instance_profiles(self) -> list[ComputeInstanceProfile]:
        """Get supported CI profiles."""
        from contextlib import suppress

        logger.info("getting CIPs for GI %s", self)

        profiles = []
        for i in range(NVML_COMPUTE_INSTANCE_PROFILE_COUNT):
            with suppress(NVMLError_NotSupported):
                profiles.append(ComputeInstanceProfile.from_idx(i, self))
        return profiles

    def get_compute_instances(self) -> list[ComputeInstance]:
        """Get children CIs."""
        from ctypes import byref, c_uint

        logger.info("getting CIs for GI %s", self)

        instances = []
        for profile in range(NVML_COMPUTE_INSTANCE_PROFILE_COUNT):
            try:
                info = nvmlGpuInstanceGetComputeInstanceProfileInfo(
                    self._nvml_dev, profile, NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED
                )
            except NVMLError_NotSupported:
                continue

            InstancesArray = NVMLComputeInstance * info.instanceCount
            c_count = c_uint()
            c_profile_instances = InstancesArray()
            nvmlGpuInstanceGetComputeInstances(
                self._nvml_dev, info.id, c_profile_instances, byref(c_count)
            )

            for i in range(c_count.value):
                instances.append(
                    ComputeInstance.from_nvml_handle_parent(
                        c_profile_instances[i], self
                    )
                )

        return instances


@dataclass
class ComputeInstanceProfile:
    """A MIG compute instance profile."""

    id: int
    slice_count: int
    gpu_instance_profile: GPUInstanceProfile

    def __repr__(self) -> str:
        """Pretty repr for ComputeInstanceProfile."""
        return (
            f"{self.slice_count}c."
            if self.slice_count != self.gpu_instance_profile.slice_count
            else ""
        ) + repr(self.gpu_instance_profile)

    @classmethod
    def from_idx(cls, idx: int, gpu_instance: GPUInstance) -> ComputeInstanceProfile:
        """Create ComputeInstanceProfile instance from index and parent GPUInstance."""
        info = nvmlGpuInstanceGetComputeInstanceProfileInfo(
            gpu_instance._nvml_dev, idx, NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED
        )
        return cls(info.id, info.sliceCount, gpu_instance.profile)

    @classmethod
    def from_id(cls, id: int, gpu_instance: GPUInstance) -> ComputeInstanceProfile:
        """Create ComputeInstanceProfile instance from ID and parent GPUInstance."""
        for i in range(NVML_COMPUTE_INSTANCE_PROFILE_COUNT):
            try:
                info = nvmlGpuInstanceGetComputeInstanceProfileInfo(
                    gpu_instance._nvml_dev,
                    i,
                    NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED,
                )
            except NVMLError_NotSupported:
                continue
            if info.id == id:
                return cls(info.id, info.sliceCount, gpu_instance.profile)
        raise NvidiaMIGCIPNotFound


@dataclass
class ComputeInstance:
    """
    A MIG compute instance.

    MIG compute instances can be used to run different workloads on a GPU instance.
    """

    _nvml_dev: NVMLComputeInstance
    id: int
    profile: ComputeInstanceProfile
    parent: GPUInstance

    def __repr__(self) -> str:
        """Pretty repr this CI."""
        return f"{self.profile} #{self.id} @({self.parent})"

    @classmethod
    def from_nvml_handle_parent(
        cls, nvml_device: NVMLComputeInstance, parent: GPUInstance
    ) -> ComputeInstance:
        """Get a ComputeInstance reference from its handle and its parent GPUInstance."""
        info = nvmlComputeInstanceGetInfo(nvml_device)
        return cls(
            nvml_device,
            info.id,
            ComputeInstanceProfile.from_id(info.profileId, parent),
            parent,
        )

    def destroy(self) -> None:
        """Destroy this CI."""
        logger.info("destroying CI %s", self)
        nvmlComputeInstanceDestroy(self._nvml_dev)
