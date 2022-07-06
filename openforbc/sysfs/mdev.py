# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from openforbc.sysfs import SysFsHandle

if TYPE_CHECKING:
    from openforbc.sysfs.gpu import GPUSysFsHandle


class MdevTypeException(Exception):
    pass


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
        from openforbc.sysfs.gpu import GPUSysFsHandle

        return GPUSysFsHandle(self._path.resolve().parent)

    def get_uuid(self) -> UUID:
        return UUID(self._path.resolve().name)

    def remove(self) -> None:
        (self._path / "remove").write_text("1\n")
