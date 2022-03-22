from pathlib import Path


class SysFsHandle:
    def __init__(self, device_path: Path) -> None:
        self._path = device_path.resolve()

    def __repr__(self) -> str:
        return str(self._path.name)
