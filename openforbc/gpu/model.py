# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass

from openforbc.gpu.generic import (
    GPU,
    GPUPartition,
    GPUPartitionType,
    GPUhPartitionType,
    GPUvPartitionType,
)
from openforbc.pci import PCIID


@dataclass
class GPUModel:
    name: str
    uuid: str
    pciid: PCIID

    @classmethod
    def from_raw(cls, gpu: GPU) -> GPUModel:
        return cls(gpu.name, str(gpu.uuid), gpu.pciid)


@dataclass
class GPUPartitionModel:
    uuid: str
    type: GPUPartitionType

    @classmethod
    def from_raw(cls, partition: GPUPartition) -> GPUPartitionModel:
        return cls(str(partition.uuid), partition.type)

    def __str__(self) -> str:
        return f"{self.uuid}: type=({self.type})"


@dataclass
class GPUvPartitionModel(GPUPartitionModel):
    type: GPUvPartitionType


@dataclass
class GPUhPartitionModel(GPUPartitionModel):
    type: GPUhPartitionType
