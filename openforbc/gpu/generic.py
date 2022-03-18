from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
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
    @staticmethod
    def get_gpus() -> Sequence[GPU]:
        from openforbc.gpu.nvidia.gpu import NvidiaGPU

        return NvidiaGPU.get_gpus()

    @classmethod
    @abstractmethod
    def from_uuid(cls, uuid: UUID) -> GPU:
        from openforbc.gpu.nvidia.gpu import NvidiaGPU

        return NvidiaGPU.from_uuid(uuid)

    @abstractmethod
    def get_supported_types(self) -> Sequence[GPUPartitionType]:
        ...

    @abstractmethod
    def get_creatable_types(self) -> Sequence[GPUPartitionType]:
        ...

    @abstractmethod
    def get_partitions(self) -> Sequence[GPUPartition]:
        ...

    @abstractmethod
    def create_partition(self, type: GPUPartitionType) -> GPUPartition:
        ...


class GPUPartitionTechnology(IntEnum):
    NVIDIA_VGPU_MIG = 1
    NVIDIA_VGPU_TIMESHARED = 2


@dataclass
class GPUPartitionType:
    name: str
    id: int
    tech: GPUPartitionTechnology
    memory: int


@dataclass
class _GPUPartition:
    uuid: UUID
    type: GPUPartitionType


class GPUPartition(_GPUPartition, ABC):
    @abstractmethod
    def destroy(self) -> None:
        ...
