# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT

from typer import echo, Typer
from uuid import UUID

from openforbc.cli.state import state

app = Typer()


@app.command("list")
def list_partitions(gpu_uuid: UUID) -> None:
    """List GPU partitions."""
    partitions = state["api_client"].get_partitions(gpu_uuid)
    assert isinstance(partitions, list)
    for partition in partitions:
        assert "uuid" in partition
        assert "type_id" in partition
        echo(f'{partition["uuid"]}: {partition["type_id"]}')


@app.command("create")
def create_partition(gpu_uuid: UUID, type_id: int) -> None:
    """Create a GPU partition."""
    result = state["api_client"].create_partition(gpu_uuid, type_id)
    assert "ok" in result
    assert result["ok"]
