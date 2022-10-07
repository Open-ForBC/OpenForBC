# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
from __future__ import annotations

from logging import getLogger
from uuid import UUID
from typing import TYPE_CHECKING, Optional  # noqa: TC002

from typer import Context, echo, Exit, Option, Typer  # noqa: TC002

from openforbc.cli.state import state as global_state
from openforbc.cli.gpu.mig import mig
from openforbc.cli.gpu.hpartition import hpartition
from openforbc.cli.gpu.vpartition import vpartition
from openforbc.cli.gpu.state import state

if TYPE_CHECKING:
    from typing import Dict


logger = getLogger(__name__)

app = Typer(help="Operate on GPUs")
app.add_typer(mig, name="mig")
app.add_typer(hpartition, name="host-partition")
app.add_typer(hpartition, name="hpart", help="Shortcut for *host-partition")
app.add_typer(vpartition, name="vm-partition")
app.add_typer(vpartition, name="vpart", help="Shortcut for *vm-partition*")


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
