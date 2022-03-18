from typer import Typer

from openforbc.cli.gpu import app as gpu_app


app = Typer()


app.add_typer(gpu_app, name="gpu")


def run() -> None:
    app()
