from os import environ as env
from logging import basicConfig

LOG_LEVEL = env.get("OPENFORBCD_LOGLEVEL") or env.get("LOGLEVEL")
basicConfig(level=LOG_LEVEL, format="%(name)s: %(levelname)s: %(message)s")


def run():
    from uvicorn import run as uv_run

    from openforbc.api.server import app

    uv_run(app, port=5000)
