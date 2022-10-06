# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass

from openforbc.gpu.nvidia.gpu import NvidiaGPU
from openforbc.gpu.nvidia.mig import (
    ComputeInstance,
    ComputeInstanceProfile,
    GPUInstance,
    GPUInstanceProfile,
)
from openforbc.pci import PCIID


@dataclass
class NvidiaGPUModel:
    name: str
    uuid: str
    pciid: PCIID

    @classmethod
    def from_raw(cls, gpu: NvidiaGPU) -> NvidiaGPUModel:
        return cls(gpu.name, str(gpu.uuid), gpu.pciid)

    def __str__(self) -> str:
        return f"{self.name} ({self.uuid})"


@dataclass
class GPUInstanceModel:
    id: int
    profile: GPUInstanceProfile
    parent: NvidiaGPUModel

    @classmethod
    def from_raw(cls, instance: GPUInstance) -> GPUInstanceModel:
        return cls(
            instance.id, instance.profile, NvidiaGPUModel.from_raw(instance.parent)
        )

    def __str__(self) -> str:
        return f"{self.id}: gip=({self.profile}) @ {self.parent}"


@dataclass
class ComputeInstanceModel:
    id: int
    profile: ComputeInstanceProfile
    parent: GPUInstanceModel

    @classmethod
    def from_raw(cls, instance: ComputeInstance) -> ComputeInstanceModel:
        return cls(
            instance.id, instance.profile, GPUInstanceModel.from_raw(instance.parent)
        )

    def __str__(self) -> str:
        return f"{self.id}: type=({self.profile.name}) tech={self.profile.tech}"
