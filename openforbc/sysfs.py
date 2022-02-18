from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional
    from openforbc.gpu import GPU


from pathlib import Path
from uuid import UUID, uuid4


class MdevTypeException(Exception):
    pass


class SysFsHandle:
    def __init__(self, device_path: Path) -> None:
        self._path = device_path.resolve()

    def __repr__(self) -> str:
        return str(self._path.name)


class GPUSysFsHandle(SysFsHandle):
    def __init__(self, device_path: Path) -> None:
        super().__init__(device_path)

    @classmethod
    def from_gpu(cls, gpu: GPU) -> GPUSysFsHandle:
        return cls(Path("/sys", "bus", "pci", "devices", gpu.get_pci_id()))

    def create_mdev(self, type: str) -> MdevSysFsHandle:
        uuid = uuid4()
        try:
            (self._path / "mdev_supported_types" / type / "create").write_text(
                f"{uuid}\n"
            )
        except FileNotFoundError:
            raise MdevTypeException(
                f"mdev type {type} not supported on {self._path}"
            ) from None

        return MdevSysFsHandle.from_uuid(uuid)

    def get_mdev_devices(self) -> list[MdevSysFsHandle]:
        if not self.get_mdev_supported():
            return []

        return [
            MdevSysFsHandle(path)
            for path in self._path.glob("mdev_supported_types/*/devices/*")
        ]

    def get_mdev_supported(self) -> bool:
        return (self._path / "mdev_supported_types").exists()

    def get_mdev_type_available_instances(self, mdev_type: str) -> int:
        return int(
            (self._path / "mdev_supported_types" / mdev_type / "available_instances")
            .read_text()
            .strip()
        )

    def get_mdev_type_available(self, mdev_type: str) -> bool:
        return self.get_mdev_type_available_instances(mdev_type) > 0

    def get_pci_bus_id(self) -> str:
        return self._path.resolve().name

    def get_sriov_active(self) -> bool:
        return (self._path / "sriov_numvfs").exists()

    def get_sriov_available_vf(self) -> Optional[GPUSysFsHandle]:
        if not self.get_sriov_active():
            return None

        for vf in self.get_sriov_vfs():
            if vf.get_sriov_vf_is_available():
                return vf

        print("no vf")
        return None

    def get_sriov_is_vf(self) -> bool:
        return (self._path / "physfn").is_dir()

    def get_sriov_num_vfs(self) -> int:
        if not self.get_sriov_active():
            return 0

        return int((self._path / "sriov_numvfs").read_text().strip())

    def get_sriov_physfn(self) -> GPUSysFsHandle:
        return GPUSysFsHandle(self._path / "physfn")

    def get_sriov_vf_is_available(self) -> bool:
        return (
            self.get_sriov_is_vf()
            and self.get_mdev_supported()
            and len(list(self._path.glob("mdev_supported_types/*/devices/*"))) == 0
        )

    def get_sriov_vfs(self) -> list[GPUSysFsHandle]:
        if not self.get_sriov_active():
            return []

        vfs: list[GPUSysFsHandle] = []
        for pos in range(self.get_sriov_num_vfs()):
            vfs.append(GPUSysFsHandle((self._path / f"virtfn{pos}")))

        return vfs


class MdevDevicePathException(Exception):
    pass


class MdevSysFsHandle(SysFsHandle):
    def __init__(self, device_path: Path) -> None:
        if not (device_path / "mdev_type").is_dir():
            raise MdevDevicePathException(
                f"{device_path} is not a valid mdev device path"
            )

        super().__init__(device_path)

    @classmethod
    def from_uuid(cls, uuid: UUID) -> MdevSysFsHandle:
        return cls(Path(f"/sys/bus/mdev/devices/{uuid}"))

    def get_mdev_type(self) -> str:
        return (self._path / "mdev_type").resolve().name

    def get_parent_gpu(self) -> GPUSysFsHandle:
        return GPUSysFsHandle(self._path.resolve().parent)

    def get_uuid(self) -> UUID:
        return UUID(self._path.resolve().name)

    def remove(self) -> None:
        (self._path / "remove").write_text("1\n")
