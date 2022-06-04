from contextlib import contextmanager
from typing import TYPE_CHECKING
from pynvml import c_nvmlComputeInstance_t, c_nvmlGpuInstance_t, nvmlInit, nvmlShutdown

if TYPE_CHECKING:
    from ctypes import pointer
    from pynvml import struct_c_nvmlDevice_t

    NVMLDevice = pointer[struct_c_nvmlDevice_t]

NVMLGpuInstance = c_nvmlGpuInstance_t
NVMLComputeInstance = c_nvmlComputeInstance_t


@contextmanager
def nvml():
    nvmlInit()
    yield
    nvmlShutdown()
