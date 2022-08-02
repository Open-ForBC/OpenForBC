# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
from typing import TypedDict
from uuid import UUID, uuid4


CLIGPUState = TypedDict("CLIGPUState", {"gpu_uuid": UUID})
state: CLIGPUState = {"gpu_uuid": uuid4()}


def get_gpu_uuid() -> UUID:
    return state["gpu_uuid"]
