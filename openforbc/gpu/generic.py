# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
"""Generic GPU partition management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, Union, overload


if TYPE_CHECKING:
    from typing import Sequence
    from uuid import UUID
    from openforbc.pci import PCIID


class GPUPartitionUse(Enum):
    VM_PARTITION = 1
    HOST_PARTITION = 2


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

    @overload
    @abstractmethod
    def get_partition_types(
        self, use: Literal[GPUPartitionUse.VM_PARTITION], creatable: bool = False
    ) -> Sequence[GPUvPartitionType]:
        ...

    @overload
    @abstractmethod
    def get_partition_types(
        self, use: Literal[GPUPartitionUse.HOST_PARTITION], creatable: bool = False
    ) -> Sequence[GPUhPartitionType]:
        ...

    @abstractmethod
    def get_partition_types(
        self, use: GPUPartitionUse, creatable: bool = False
    ) -> Sequence[GPUPartitionType]:
        """
        Get all the partition types supported by this GPU.

        The `creatable` parameter allows specifying whether to get only types which are
        currently available to be created.
        """
        ...

    @overload
    @abstractmethod
    def get_partitions(
        self, use: Literal[GPUPartitionUse.VM_PARTITION]
    ) -> Sequence[GPUvPartition]:
        ...

    @overload
    @abstractmethod
    def get_partitions(
        self, use: Literal[GPUPartitionUse.HOST_PARTITION]
    ) -> Sequence[GPUhPartition]:
        ...

    @abstractmethod
    def get_partitions(self, use: GPUPartitionUse) -> Sequence[GPUPartition]:
        """Get all created partitons on this GPU."""
        ...

    @overload
    @abstractmethod
    def create_partition(
        self, use: Literal[GPUPartitionUse.VM_PARTITION], type: GPUvPartitionType
    ) -> GPUvPartition:
        ...

    @overload
    @abstractmethod
    def create_partition(
        self, use: Literal[GPUPartitionUse.HOST_PARTITION], type: GPUhPartitionType
    ) -> GPUhPartition:
        ...

    @abstractmethod
    def create_partition(
        self, use: GPUPartitionUse, type: GPUPartitionType
    ) -> GPUPartition:
        """Create a partition on this GPU with the specified type."""
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
