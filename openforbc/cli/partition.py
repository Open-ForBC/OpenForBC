from uuid import UUID
from typer import echo, Typer
from requests import Session

from openforbc.cli.state import state

app = Typer()


@app.command("list")
def list_partitions(gpu_uuid: UUID) -> None:
    """List GPU partitions."""
    with Session() as s:
        r = s.send(state["api_client"].get_partitions(gpu_uuid))
        partitions = r.json()
        assert isinstance(partitions, list)
        for partition in partitions:
            assert "uuid" in partition
            assert "type_id" in partition
            echo(f'{partition["uuid"]}: {partition["type_id"]}')


@app.command("create")
def create_partition(gpu_uuid: UUID, type_id: int) -> None:
    """Create a GPU partition."""
    with Session() as s:
        r = s.send(state["api_client"].create_partition(gpu_uuid, type_id))
        rj = r.json()
        assert "ok" in rj
