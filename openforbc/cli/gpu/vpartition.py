# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Optional  # noqa: TC002
from uuid import UUID

from typer import Context, Exit, echo, Option, Typer  # noqa: TC002

from openforbc.cli.gpu.state import get_gpu_uuid
from openforbc.cli.state import state as global_state
from openforbc.gpu.generic import GPUvPartitionTechnology  # noqa: TC001


vpartition = Typer(help="Manage GPU partitions for VMs")


@vpartition.callback(invoke_without_command=True)
def part_callback(ctx: Context) -> None:
    if ctx.invoked_subcommand is None:
        ctx.invoke(list_partitions, uuid_only=False)


@vpartition.command("types")
def list_supported_types(
    creatable: bool = Option(False, "--creatable", "-c"),
    id_only: bool = Option(False, "--id-only", "-q"),
    technology: Optional[GPUvPartitionTechnology] = Option(None, "--tech", "-t"),
) -> None:
    """List supported VM partition types."""
    gpu_uuid = get_gpu_uuid()

    client = global_state["api_client"]
    types = client.get_supported_vpart_types(gpu_uuid, creatable)

    for type in (
        filter(lambda type: type.tech == technology, types) if technology else types
    ):
        echo(type.id if id_only else type)


@vpartition.command("list")
def list_partitions(uuid_only: bool = Option(False, "--uuid-only", "-q")) -> None:
    """List GPU VM partitions."""
    client = global_state["api_client"]
    gpu_uuid = get_gpu_uuid()

    partitions = client.get_vpartitions(gpu_uuid)

    for partition in partitions:
        echo(partition.uuid if uuid_only else partition)


@vpartition.command("create")
def create_partition(
    type_id: int, uuid_only: bool = Option(False, "--uuid-only", "-q")
) -> None:
    """Create a GPU VM partition."""
    partition = global_state["api_client"].create_vpartition(get_gpu_uuid(), type_id)
    echo(partition.uuid if uuid_only else partition)


@vpartition.command("dumpxml")
def get_partition_definition(partition_uuid: UUID):
    """Get libvirt XML definition for selected VM partition."""
    partitions = global_state["api_client"].get_vpartitions(get_gpu_uuid())

    if not [True for x in partitions if UUID(x.uuid) == partition_uuid]:
        echo(f'ERROR: no such partition "{partition_uuid}".', err=True)
        raise Exit(1)

    echo(
        f"""<hostdev mode='subsystem' type='mdev' managed='no' model='vfio-pci'
              display='on'>
              <source>
                <address uuid='{partition_uuid}'/>
              </source>
            </hostdev>""",
    )


@vpartition.command("destroy")
def destroy_partition(partition_uuid: UUID) -> None:
    """Destroy the specified VM partition."""
    return global_state["api_client"].destroy_vpartition(get_gpu_uuid(), partition_uuid)
