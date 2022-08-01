# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT

from logging import basicConfig
from os import environ
from typer import Typer

from openforbc.cli.gpu.app import app as gpu_app


basicConfig(
    level=environ.get("OPENFORBC_LOGLEVEL") or environ.get("LOGLEVEL"),
    format="%(name)s: %(levelname)s: %(message)s",
)

app = Typer()


app.add_typer(gpu_app, name="gpu")


def run() -> None:
    app()
