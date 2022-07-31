# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT

from __future__ import annotations
from logging import getLogger
from typing import TYPE_CHECKING

from requests import JSONDecodeError, Request
from typer import Exit

from openforbc.api import DEFAULT_BASE_URL

if TYPE_CHECKING:
    from typing import Any
    from requests import PreparedRequest
    from uuid import UUID


logger = getLogger(__name__)


class APIException(Exception):
    """Base class for OpenForBC API exceptions."""

    pass


class BadRequest(APIException):
    """Raised when some arguments are not accepted by the server."""

    pass


class ExceptionDecodeFailed(APIException):
    """Raised when server exception json could not be decoded."""

    pass


class DaemonException(APIException):
    """Raised when an exception occurred on the OpenForBC daemon."""

    pass


class ResponseDecodeFailed(APIException):
    """Raised when a daemon response could not be decoded."""

    pass


class APIClient:
    def __init__(self, base_url: str, cli: bool) -> None:
        self.base_url = base_url
        self.cli = cli

    @classmethod
    def default(cls, cli: bool = True) -> APIClient:
        return cls(DEFAULT_BASE_URL, cli)

    def get_gpus(self) -> Any:
        return self.send_request(Request("GET", url=f"{self.base_url}/gpu").prepare())

    def get_supported_types(self, gpu_uuid: UUID, creatable: bool = False) -> Any:
        return self.send_request(
            Request(
                "GET",
                f"{self.base_url}/gpu/{gpu_uuid}/types"
                + ("?creatable=1" if creatable else ""),
            ).prepare()
        )

    def get_partitions(self, gpu_uuid: UUID) -> Any:
        return self.send_request(
            Request("GET", f"{self.base_url}/gpu/{gpu_uuid}/partition").prepare()
        )

    def create_partition(self, gpu_uuid: UUID, type_id: int) -> Any:
        return self.send_request(
            Request(
                "PUT",
                f"{self.base_url}/gpu/{gpu_uuid}/partition",
                data={"type_id": type_id},
            ).prepare()
        )

    def destroy_partition(self, gpu_uuid: UUID, partition_uuid: UUID) -> Any:
        return self.send_request(
            Request(
                "DELETE", f"{self.base_url}/gpu/{gpu_uuid}/partition/{partition_uuid}"
            ).prepare()
        )

    def send_request(self, request: PreparedRequest) -> Any:
        from requests import Session
        from requests.status_codes import codes

        with Session() as s:
            r = s.send(request)
            logger.debug('received: "%s"', r.text)
            if r.status_code == codes.bad_request:
                message = r.json()["detail"]
                logger.error("bad request: %s", message)
                raise Exit(1) if self.cli else BadRequest(message)
            elif r.status_code != codes.ok:
                try:
                    exc = r.json()["exc"]
                    logger.error(
                        "unhandled exception on daemon: %s. "
                        "check daemon's log (`journalctl -u openforbcd -e`) for further "
                        "information",
                        exc,
                    )
                    raise Exit(1) if self.cli else DaemonException(
                        f"Operation failed on daemon due to: {exc}"
                    )
                except (JSONDecodeError, KeyError):
                    logger.warning("couldn't decode daemon exception")
                    logger.error(
                        'daemon replied with unexpected status: %s, with body: "%s"',
                        r.status_code,
                        r.text,
                    )
                    raise Exit(1) if self.cli else ExceptionDecodeFailed(
                        "Could not decode exception", r.content
                    ) from None
            try:
                json = r.json()
            except JSONDecodeError:
                logger.error('couldn\'t decode response body as json, body: "%s"')
                raise Exit(1) if self.cli else ResponseDecodeFailed(
                    "Could not decode server response", r.content
                ) from None

            return json
