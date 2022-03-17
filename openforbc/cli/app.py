from typer import Typer

from openforbc.cli.gpu import app as gpu_app
from openforbc.cli.partition import app as part_app


app = Typer()


app.add_typer(gpu_app, name="gpu")
app.add_typer(part_app, name="partition")


def run() -> None:
    app()
