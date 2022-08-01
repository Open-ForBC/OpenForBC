# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
"""Generic GPU partition management."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING


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
    """Generic GPU."""

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
    def get_supported_types(self) -> Sequence[GPUPartitionType]:
        """Get all the partition types supported by this GPU."""
        ...

    @abstractmethod
    def get_creatable_types(self) -> Sequence[GPUPartitionType]:
        """Get all the partition types which can be actually created at this time."""
        ...

    @abstractmethod
    def get_partitions(self) -> Sequence[GPUPartition]:
        """Get all created partitons on this GPU."""
        ...

    @abstractmethod
    def create_partition(self, type: GPUPartitionType) -> GPUPartition:
        """Create a partition on this GPU with the specified type."""
        ...


class GPUPartitionTechnology(str, Enum):
    """
    A GPU partitioning technology.

    Different GPU vendors have their technologies.

    NVIDIA:
    - NVIDIA vGPU (timeslot scheduled partition)
    - NVIDIA vGPU with MIG (partition with dedicated physical resources)
    """

    NVIDIA_VGPU_MIG = "vgpu+mig"
    NVIDIA_VGPU_TIMESHARED = "vgpu"


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


@dataclass
class _GPUPartition:
    uuid: UUID
    type: GPUPartitionType


class GPUPartition(_GPUPartition, ABC):
    """
    Repesent a generic (mdev) GPU partition.

    MDEVs are used since it seems to be the standard interface used by vendors to
    partition GPUs.
    """

    @abstractmethod
    def destroy(self) -> None:
        ...
