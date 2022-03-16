from contextlib import contextmanager
from typing import TYPE_CHECKING
from pynvml import nvmlInit, nvmlShutdown

if TYPE_CHECKING:
    from ctypes import pointer
    from pynvml import struct_c_nvmlDevice_t

    NVMLDevice = pointer[struct_c_nvmlDevice_t]


@contextmanager
def nvml():
    nvmlInit()
    yield
    nvmlShutdown()
