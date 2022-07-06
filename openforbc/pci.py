# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT

from __future__ import annotations

VENDORS = {0x10DE: "nvidia"}
VENDORS_REV = {v: k for k, v in VENDORS.items()}
DEVICES = {0x10DE: {0x20F1: "a100"}}
DEVICES_REV = {k: {v: k for k, v in v.items()} for k, v in DEVICES.items()}


class PCIIDFormatException(Exception):
    pass


class PCIID:
    vendor: int
    device: int

    def __init__(self, vendor: int, device: int) -> None:
        self.vendor = vendor
        self.device = device

    @classmethod
    def from_int(cls, pci_id: int) -> PCIID:
        return cls(pci_id & 0xFFFF, pci_id >> 16)

    @classmethod
    def from_repr(cls, repr: str) -> PCIID:
        if ":" not in repr:
            raise PCIIDFormatException

        vendor_s, device_s = repr.split(":")
        vendor = VENDORS_REV[vendor_s] if vendor_s in VENDORS_REV else int(vendor_s, 16)
        device = (
            DEVICES_REV[vendor][device_s]
            if vendor in DEVICES_REV and device_s in DEVICES_REV[vendor]
            else int(device_s, 16)
        )
        return cls(vendor, device)

    def __repr__(self) -> str:
        vendor = VENDORS[self.vendor] if self.vendor in VENDORS else f"{self.vendor:x}"
        device = (
            DEVICES[self.vendor][self.device]
            if self.vendor in DEVICES and self.device in DEVICES[self.vendor]
            else f"{self.device:x}"
        )

        return f"{vendor}:{device}"
