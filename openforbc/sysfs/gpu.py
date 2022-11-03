# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from openforbc.sysfs import SysFsHandle
from openforbc.sysfs.mdev import MdevTypeException

if TYPE_CHECKING:
    from typing import Optional
    from openforbc.gpu.nvidia.gpu import NvidiaGPU
    from openforbc.sysfs.mdev import MdevSysFsHandle


class GPUSysFsHandle(SysFsHandle):
    """
    A handle for a GPU's sysfs device.

    Can be used to talk with the driver and interact with SRIOV virtual functions or mdev
    childrens.
    """

    def __init__(self, device_path: Path) -> None:
        """Create a GPUSysFsHandle using the actual sysfs path."""
        super().__init__(device_path)

    @classmethod
    def from_gpu(cls, gpu: NvidiaGPU) -> GPUSysFsHandle:
        """Create GPUSysFsHandle instance from a NvidiaGPU instance."""
        return cls(Path("/sys", "bus", "pci", "devices", gpu.get_pci_bus_id()))

    def create_mdev(self, type: str) -> MdevSysFsHandle:
        """Create a mdev device with specified type."""
        from openforbc.sysfs.mdev import MdevSysFsHandle

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
        """Get all mdev children devices for this GPU."""
        from openforbc.sysfs.mdev import MdevSysFsHandle

        if not self.get_mdev_supported():
            return []

        return [
            MdevSysFsHandle(path)
            for path in self._path.glob("mdev_supported_types/*/devices/*")
        ]

    def get_mdev_supported(self) -> bool:
        """
        Return True if mdev is supported on this device.

        NOTE: mdev may be supported by virtual functions if this device supports SRIOV.
        """
        return (self._path / "mdev_supported_types").exists()

    def get_mdev_type_available_instances(self, mdev_type: str) -> int:
        """Get number of available instances of the specified mdev type."""
        return int(
            (self._path / "mdev_supported_types" / mdev_type / "available_instances")
            .read_text()
            .strip()
        )

    def get_mdev_type_available(self, mdev_type: str) -> bool:
        """Return True is the specified mdev type can be created."""
        return self.get_mdev_type_available_instances(mdev_type) > 0

    def get_pci_bus_id(self) -> str:
        """Get the pci bus ID."""
        return self._path.resolve().name

    def get_sriov_active(self) -> bool:
        """Return True if SRIOV is active on this device."""
        numvfs = self._path / "sriov_numvfs"
        return numvfs.exists() and numvfs.read_text().strip() != "0"

    def get_sriov_available_vf(self) -> Optional[GPUSysFsHandle]:
        """Get a sysfs handle to the first available virtual function of this device."""
        if not self.get_sriov_active():
            return None

        for vf in self.get_sriov_vfs():
            if vf.get_sriov_vf_is_available():
                return vf

        return None

    def get_sriov_is_vf(self) -> bool:
        """Return True if this device is a SRIOV virtual function."""
        return (self._path / "physfn").is_dir()

    def get_sriov_num_vfs(self) -> int:
        """Get number of virtual functions available on this device."""
        if not self.get_sriov_active():
            return 0

        return int((self._path / "sriov_numvfs").read_text().strip())

    def get_sriov_physfn(self) -> GPUSysFsHandle:
        """Get a sysfs handle for the physical function of this virtual function."""
        return GPUSysFsHandle(self._path / "physfn")

    def get_sriov_vf_is_available(self) -> bool:
        """Return True if this is a virtual function and has no child mdevs."""
        return (
            self.get_sriov_is_vf()
            and self.get_mdev_supported()
            and len(list(self._path.glob("mdev_supported_types/*/devices/*"))) == 0
        )

    def get_sriov_vfs(self) -> list[GPUSysFsHandle]:
        """Get all the child virtual functions of this device."""
        if not self.get_sriov_active():
            return []

        vfs: list[GPUSysFsHandle] = []
        for pos in range(self.get_sriov_num_vfs()):
            vfs.append(GPUSysFsHandle((self._path / f"virtfn{pos}")))

        return vfs
