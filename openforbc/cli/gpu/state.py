# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
from typing import TypedDict
from uuid import UUID, uuid4


CLIGPUState = TypedDict("CLIGPUState", {"gpu_uuid": UUID})
state: CLIGPUState = {"gpu_uuid": uuid4()}


def get_gpu_uuid() -> UUID:
    """
    Get the selected GPU UUID.

    This is a helper function which commands can use to retrieve the uuid of the
    selected (through either --gpu-id or --gpu-uuid options) gpu.
    """
    return state["gpu_uuid"]
