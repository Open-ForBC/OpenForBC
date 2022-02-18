from contextlib import contextmanager
from pynvml import nvmlInit, nvmlShutdown


@contextmanager
def nvml():
    nvmlInit()
    yield
    nvmlShutdown()
