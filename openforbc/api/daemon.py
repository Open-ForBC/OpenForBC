"""The OpenForBC daemon (API) server."""
from os import environ as env
from logging import basicConfig

LOG_LEVEL = env.get("OPENFORBCD_LOGLEVEL") or env.get("LOGLEVEL")
basicConfig(level=LOG_LEVEL, format="%(name)s: %(levelname)s: %(message)s")


def run() -> None:
    """Run the OpenForBC daemon."""
    from uvicorn import run as uv_run
    from uvicorn.config import LOGGING_CONFIG

    from openforbc.api.server import app

    log_config = LOGGING_CONFIG.copy()
    for name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        log_config["loggers"][name]["handlers"] = []
        log_config["loggers"][name]["propagate"] = True
        log_config["loggers"][name]["level"] = LOG_LEVEL or "INFO"

    uv_run(app, port=5000, log_config=log_config)
