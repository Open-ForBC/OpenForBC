from __future__ import annotations

from uuid import UUID

from typer import Context, Exit, echo, Option, Typer

from openforbc.cli.gpu.state import get_gpu_uuid
from openforbc.cli.state import state as global_state


partition = Typer(help="List and create GPU partitions")


@partition.callback(invoke_without_command=True)
def part_callback(ctx: Context) -> None:
    if ctx.invoked_subcommand is None:
        ctx.invoke(list_partitions, uuid_only=False)


@partition.command("list")
def list_partitions(uuid_only: bool = Option(False, "--uuid-only", "-q")) -> None:
    """List GPU partitions."""
    client = global_state["api_client"]
    gpu_uuid = get_gpu_uuid()

    partitions = client.get_partitions(gpu_uuid)

    for partition in partitions:
        echo(partition.uuid if uuid_only else partition)


@partition.command("create")
def create_partition(
    type_id: int, uuid_only: bool = Option(False, "--uuid-only", "-q")
) -> None:
    """Create a GPU partition."""
    partition = global_state["api_client"].create_partition(get_gpu_uuid(), type_id)
    echo(partition.uuid if uuid_only else partition)


@partition.command("get")
def get_partition_definition(partition_uuid: UUID):
    """Get libvirt XML definition for selected partition."""
    partitions = global_state["api_client"].get_partitions(get_gpu_uuid())

    if not [True for x in partitions if UUID(x.uuid) == partition_uuid]:
        echo(f'ERROR: no such partition "{partition_uuid}".', err=True)
        raise Exit(1)

    echo("NOTE: please ensure that PCI domain:bus:slot.function is not already used.")
    echo(
        f"""
<hostdev mode='subsystem' type='mdev' managed='no' model='vfio-pci' display='on'>
  <source>
    <address uuid='{partition_uuid}'/>
  </source>
  <address type='pci' domain='0x0000' bus='0x00' slot='0x10' function='0x0'/>
</hostdev>
        """,
        nl=False,
    )


@partition.command("destroy")
def destroy_partition(partition_uuid: UUID):
    """Destroy the selected partition."""
    return global_state["api_client"].destroy_partition(get_gpu_uuid(), partition_uuid)
