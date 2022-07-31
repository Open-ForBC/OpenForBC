# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
"""NVML interface helper."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from pynvml import c_nvmlComputeInstance_t, c_nvmlGpuInstance_t, nvmlInit, nvmlShutdown

if TYPE_CHECKING:
    from typing import Generator

    from ctypes import c_uint, pointer
    from pynvml import struct_c_nvmlDevice_t

    NVMLDevice = pointer[struct_c_nvmlDevice_t]
    NVMLVGPUInstance = c_uint

NVMLGpuInstance = c_nvmlGpuInstance_t
NVMLComputeInstance = c_nvmlComputeInstance_t


@contextmanager
def nvml_context() -> Generator[None, None, None]:
    """Manage NVML initialization and shutdown in a python context."""
    nvmlInit()
    yield
    nvmlShutdown()
