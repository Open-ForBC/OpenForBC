# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
"""Generic GPU partition management."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from typing import Sequence
    from uuid import UUID
    from openforbc.pci import PCIID


@dataclass
class _GPU:
    name: str
    uuid: UUID
    pciid: PCIID


class GPU(_GPU, ABC):
    """A generic GPU."""

    @classmethod
    def get_gpus(cls) -> Sequence[GPU]:
        """Get all GPUs of all vendors available to the system."""
        from openforbc.gpu.nvidia.gpu import NvidiaGPU

        return NvidiaGPU.get_gpus()

    @classmethod
    @abstractmethod
    def from_uuid(cls, uuid: UUID) -> GPU:
        """Get a GPU instance by UUID."""
        from openforbc.gpu.nvidia.gpu import NvidiaGPU

        return NvidiaGPU.from_uuid(uuid)

    @abstractmethod
    def get_supported_vpart_types(self) -> Sequence[GPUvPartitionType]:
        """Get all the VM partition types supported by this GPU."""
        ...

    @abstractmethod
    def get_creatable_vpart_types(self) -> Sequence[GPUvPartitionType]:
        """Get all the VM partition types which can be actually created at this time."""
        ...

    @abstractmethod
    def get_vpartitions(self) -> Sequence[GPUvPartition]:
        """Get all created VM partitons on this GPU."""
        ...

    @abstractmethod
    def create_vpartition(self, type: GPUvPartitionType) -> GPUvPartition:
        """Create a VM partition on this GPU with the specified type."""
        ...

    @abstractmethod
    def get_supported_hpart_types(self) -> Sequence[GPUhPartitionType]:
        """Get all the host partition types supported by this GPU."""
        ...

    @abstractmethod
    def get_creatable_hpart_types(self) -> Sequence[GPUhPartitionType]:
        """Get all the host partition types which are available to create."""
        ...

    @abstractmethod
    def get_hpartitions(self) -> Sequence[GPUhPartition]:
        """Get all created host partitons on this GPU."""
        ...

    @abstractmethod
    def create_hpartition(self, type: GPUhPartitionType) -> GPUhPartition:
        """Create a host partition on this GPU with the specified type."""
        ...


class GPUvPartitionTechnology(str, Enum):
    """
    A GPU partitioning technology for VMs.

    Different GPU vendors have their technologies.

    NVIDIA:
    - NVIDIA vGPU (timeslot scheduled partition)
    - NVIDIA vGPU with MIG (partition with dedicated physical resources)
    """

    NVIDIA_VGPU_MIG = "vgpu+mig"
    NVIDIA_VGPU_TIMESHARED = "vgpu"


class GPUhPartitionTechnology(str, Enum):
    """
    A GPU partitioning technology for host partitions.

    Different GPU vendors have their own technology, but as of now the only supported
    technology is NVIDIA Multi-Instance GPU (MIG).
    """

    NVIDIA_MIG = "mig"


GPUPartitionTechnology = Union[GPUvPartitionTechnology, GPUhPartitionTechnology]


@dataclass
class GPUPartitionType:
    """
    A GPU partition type.

    A partition type has a specific amount of FB memory dedicated and is provided by the
    GPU using a specific technology.
    """

    name: str
    id: int
    tech: GPUPartitionTechnology
    memory: int

    def __str__(self) -> str:
        return f"{self.id}: ({self.tech}) {self.name} ({self.memory}MiB)"


@dataclass
class _GPUPartition:
    uuid: UUID
    type: GPUPartitionType


class GPUPartition(_GPUPartition, ABC):
    """
    A generic GPU partition.

    There can be two (sub)types of partitions:
    - GPUvPartition: GPU partitions to be used in VMs
    - GPUhPartition: GPU partitions which can be used by the host itself
    """

    @abstractmethod
    def destroy(self) -> None:
        ...


@dataclass
class GPUvPartitionType(GPUPartitionType):
    """
    GPUPartitionType specific for VM partitions.

    The `tech` field has a specific GPUvPartitionTechnology subtype.
    """

    tech: GPUvPartitionTechnology


class GPUvPartition(GPUPartition):
    """
    Repesents a generic VM (mdev) GPU partition.

    MDEVs are used since it seems to be the standard interface used by vendors to
    partition GPUs.
    """

    type: GPUvPartitionType


@dataclass
class GPUhPartitionType(GPUPartitionType):
    """
    GPUPartitionType specific for VM partitions.

    The `tech` field has a specific GPUhPartitionTechnology subtype.
    """

    tech: GPUhPartitionTechnology


class GPUhPartition(GPUPartition):
    """Represents a generic GPU host partition."""

    type: GPUhPartitionType
