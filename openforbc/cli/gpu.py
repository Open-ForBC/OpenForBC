from typing import TYPE_CHECKING, Optional, TypedDict
from uuid import UUID, uuid4
from typer import Context, echo, Exit, Option, Typer
from requests import Session

from openforbc.cli.state import state as global_state

if TYPE_CHECKING:
    from typing import Dict

CLIGPUState = TypedDict("CLIGPUState", {"gpu_uuid": UUID})

app = Typer(help="Operate on GPUs")

state: CLIGPUState = {"gpu_uuid": uuid4()}


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

        with Session() as s:
            r = s.send(global_state["api_client"].get_gpus())
            gpus = r.json()
            assert isinstance(gpus, list)
            gpus.sort(key=lambda x: x["uuid"])
            for gpu in gpus:
                pciids_count[gpu["pciid"]] = (
                    pciids_count[gpu["pciid"]] + 1
                    if gpu["pciid"] in pciids_count
                    else 0
                )
                if gpu["pciid"] == pciid and pciids_count[pciid] == pos:
                    state["gpu_uuid"] = UUID(gpu["uuid"])
                    return

            echo("ERROR: specified gpu_id not found!", err=True)
            raise Exit(1)
    else:
        state["gpu_uuid"] = gpu_uuid


@app.command("list")
def list_gpus() -> None:
    """List GPUs."""
    with Session() as s:
        r = s.send(global_state["api_client"].get_gpus())
        gpus = r.json()
        assert isinstance(gpus, list)
        gpus.sort(key=lambda x: x["pciid"])
        pciids_count: Dict[str, int] = {}
        for gpu in gpus:
            pciids_count[gpu["pciid"]] = (
                pciids_count[gpu["pciid"]] + 1 if gpu["pciid"] in pciids_count else 0
            )
            assert "name" in gpu
            assert "uuid" in gpu
            echo(
                f'[{gpu["pciid"]}-{pciids_count[gpu["pciid"]]}] '
                f'{gpu["uuid"]}: {gpu["name"]}'
            )


@app.command("types")
def list_supported_types(
    creatable: bool = Option(False, "--creatable", "-c"),
    id_only: bool = Option(False, "--id-only", "-q"),
) -> None:
    """List supported partition types."""
    gpu_uuid = get_gpu_uuid(state)

    with Session() as s:
        client = global_state["api_client"]
        r = s.send(
            client.get_creatable_types(gpu_uuid)
            if creatable
            else client.get_supported_types(gpu_uuid)
        )
        types = r.json()
        assert isinstance(types, list)
        for type in types:
            assert "name" in type
            assert "id" in type
            assert "memory" in type
            echo(
                type["id"]
                if id_only
                else f'{type["id"]}: {type["name"]} ({type["memory"] / 2**30}GiB)'
            )


part = Typer(help="List and create GPU partitions")
app.add_typer(part, name="partition")


@part.callback(invoke_without_command=True)
def part_callback(ctx: Context) -> None:
    if ctx.invoked_subcommand is None:
        ctx.invoke(list_partitions)


@part.command("list")
def list_partitions(uuid_only: bool = Option(False, "--uuid-only", "-q")) -> None:
    """List GPU partitions."""
    client = global_state["api_client"]
    gpu_uuid = get_gpu_uuid(state)

    with Session() as s:
        r = s.send(client.get_partitions(gpu_uuid))
        partitions = r.json()
        assert isinstance(partitions, list)

        r = s.send(client.get_supported_types(gpu_uuid))
        types = r.json()
        assert isinstance(types, list)

        for partition in partitions:
            assert "uuid" in partition
            assert "type_id" in partition
            type = next(x for x in types if x["id"] == partition["type_id"])
            echo(
                partition["uuid"]
                if uuid_only
                else f'{partition["uuid"]} - '
                f'{type["id"]}: {type["name"]} ({type["memory"] / 2**30}GiB)'
            )


@part.command("create")
def create_partition(type_id: int) -> None:
    """Create a GPU partition."""
    with Session() as s:
        r = s.send(
            global_state["api_client"].create_partition(get_gpu_uuid(state), type_id)
        )
        rj = r.json()
        assert "ok" in rj
        assert "uuid" in rj

    echo(UUID(rj["uuid"]))


@part.command("get")
def get_partition_definition(partition_uuid: UUID):
    """Get libvirt XML definition for selected partition."""
    with Session() as s:
        r = s.send(global_state["api_client"].get_partitions(get_gpu_uuid(state)))
        partitions = r.json()
        assert isinstance(partitions, list)

    if not [True for x in partitions if UUID(x["uuid"]) == partition_uuid]:
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
</hostdev>"""
    )


@part.command("destroy")
def destroy_partition(partition_uuid: UUID):
    """Destroy the selected partition."""
    with Session() as s:
        r = s.send(
            global_state["api_client"].destroy_partition(
                get_gpu_uuid(state), partition_uuid
            )
        )
        json = r.json()
        assert "ok" in json
        assert json["ok"]


def get_gpu_uuid(state: CLIGPUState) -> UUID:
    if state["gpu_uuid"] is None:
        with Session() as s:
            r = s.send(global_state["api_client"].get_gpus())
            gpus = r.json()
            assert isinstance(gpus, list)
            gpus.sort(key=lambda x: x["uuid"])
            return gpus[state["gpu_pos"]]["uuid"]

    return state["gpu_uuid"]
