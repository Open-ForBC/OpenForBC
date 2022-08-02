# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
from __future__ import annotations

from logging import getLogger
from uuid import UUID
from typing import TYPE_CHECKING, Optional  # noqa: TC002

from typer import Context, echo, Exit, Option, Typer  # noqa: TC002

from openforbc.cli.state import state as global_state
from openforbc.cli.gpu.mig import mig
from openforbc.cli.gpu.partition import partition
from openforbc.cli.gpu.state import get_gpu_uuid, state
from openforbc.gpu.generic import GPUPartitionTechnology  # noqa: TC001

if TYPE_CHECKING:
    from typing import Dict


logger = getLogger(__name__)

app = Typer(help="Operate on GPUs")
app.add_typer(mig, name="mig")
app.add_typer(partition, name="partition")


@app.callback(invoke_without_command=True)
def callback(
    ctx: Context,
    gpu_id: str = Option(None, "--gpu-id", "-i"),
    gpu_uuid: Optional[UUID] = Option(None, "--gpu-uuid", "-u"),
) -> None:
    if ctx.invoked_subcommand is None:
        return ctx.invoke(list_gpus)
    if ctx.invoked_subcommand == "list":
        return

    if gpu_uuid is None:
        if gpu_id is None:
            echo("ERROR: neither gpu_id nor gpu_uuid specified!", err=True)
            raise Exit(1)

        pciid, pos_s = gpu_id.split("-")
        pos = int(pos_s)
        pciids_count: Dict[str, int] = {}

        gpus = global_state["api_client"].get_gpus()
        assert isinstance(gpus, list)
        gpus.sort(key=lambda x: x.uuid)
        for gpu in gpus:
            pciids_count[str(gpu.pciid)] = (
                pciids_count[str(gpu.pciid)] + 1
                if str(gpu.pciid) in pciids_count
                else 0
            )
            if str(gpu.pciid) == pciid and pciids_count[pciid] == pos:
                logger.debug("selected gpu %s", gpu)
                state["gpu_uuid"] = UUID(gpu.uuid)
                return

        echo("ERROR: specified gpu_id not found!", err=True)
        raise Exit(1)
    else:
        state["gpu_uuid"] = gpu_uuid


@app.command("list")
def list_gpus() -> None:
    """List GPUs."""
    gpus = global_state["api_client"].get_gpus()
    assert isinstance(gpus, list)
    gpus.sort(key=lambda x: x.uuid)
    pciids_count: Dict[str, int] = {}
    for gpu in gpus:
        pciids_count[str(gpu.pciid)] = (
            pciids_count[str(gpu.pciid)] + 1 if str(gpu.pciid) in pciids_count else 0
        )
        echo(f"[{gpu.pciid}-{pciids_count[str(gpu.pciid)]}] " f"{gpu.uuid}: {gpu.name}")


@app.command("types")
def list_supported_types(
    creatable: bool = Option(False, "--creatable", "-c"),
    id_only: bool = Option(False, "--id-only", "-q"),
    technology: Optional[GPUPartitionTechnology] = Option(None, "--tech", "-t"),
) -> None:
    """List supported partition types."""

    gpu_uuid = get_gpu_uuid()

    client = global_state["api_client"]
    types = client.get_supported_types(gpu_uuid, creatable)

    for type in (
        filter(lambda type: type.tech == technology, types) if technology else types
    ):
        echo(type.id if id_only else type)
