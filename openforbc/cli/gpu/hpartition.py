# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Optional  # noqa: TC002
from uuid import UUID  # noqa: TC002

from typer import Context, echo, Option, Typer  # noqa: TC002

from openforbc.cli.gpu.state import get_gpu_uuid
from openforbc.cli.state import state as global_state
from openforbc.gpu.generic import GPUhPartitionTechnology  # noqa: TC001


hpartition = Typer(help="Manage GPU host GPU partitions")


@hpartition.callback(invoke_without_command=True)
def part_callback(ctx: Context) -> None:
    if ctx.invoked_subcommand is None:
        ctx.invoke(list_partitions, uuid_only=False)


@hpartition.command("types")
def list_supported_types(
    creatable: bool = Option(False, "--creatable", "-c"),
    id_only: bool = Option(False, "--id-only", "-q"),
    technology: Optional[GPUhPartitionTechnology] = Option(None, "--tech", "-t"),
) -> None:
    """List supported VM partition types."""

    gpu_uuid = get_gpu_uuid()

    client = global_state["api_client"]
    types = client.get_supported_hpart_types(gpu_uuid, creatable)

    for type in (
        filter(lambda type: type.tech == technology, types) if technology else types
    ):
        echo(type.id if id_only else type)


@hpartition.command("list")
def list_partitions(uuid_only: bool = Option(False, "--uuid-only", "-q")) -> None:
    """List GPU host partitions."""
    client = global_state["api_client"]
    gpu_uuid = get_gpu_uuid()

    partitions = client.get_hpartitions(gpu_uuid)

    for partition in partitions:
        echo(partition.uuid if uuid_only else partition)


@hpartition.command("create")
def create_partition(
    type_id: int, uuid_only: bool = Option(False, "--uuid-only", "-q")
) -> None:
    """Create a GPU host partition."""
    partition = global_state["api_client"].create_hpartition(get_gpu_uuid(), type_id)
    echo(partition.uuid if uuid_only else partition)


@hpartition.command("destroy")
def destroy_partition(partition_uuid: UUID) -> None:
    """Destroy the selected host partition."""
    return global_state["api_client"].destroy_hpartition(get_gpu_uuid(), partition_uuid)
