from uuid import UUID
from typer import echo, Option, Typer
from requests import Session

from openforbc.cli.state import state

app = Typer()


@app.command("list")
def list_gpus() -> None:
    """List GPUs."""
    with Session() as s:
        r = s.send(state["api_client"].get_gpus())
        gpus = r.json()
        assert isinstance(gpus, list)
        for gpu in gpus:
            assert "name" in gpu
            assert "uuid" in gpu
            echo(f'{gpu["uuid"]}: {gpu["name"]}')


@app.command("types")
def list_supported_types(
    gpu_uuid: UUID, creatable: bool = Option(False, "--creatable", "-c")
) -> None:
    """List supported partition types."""
    with Session() as s:
        client = state["api_client"]
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
            echo(f'{type["id"]}: {type["name"]} ({type["memory"] / 2**30}GiB)')
