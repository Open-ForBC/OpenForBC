# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from requests import PreparedRequest
    from uuid import UUID

from requests import JSONDecodeError, Request

from openforbc.api import DEFAULT_BASE_URL


class APIException(Exception):
    """Base class for OpenForBC API exceptions."""

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
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    @classmethod
    def default(cls) -> APIClient:
        return cls(DEFAULT_BASE_URL)

    def get_gpus(self) -> Any:
        return self.send_request(Request("GET", url=f"{self.base_url}/gpu").prepare())

    def get_supported_types(self, gpu_uuid: UUID) -> Any:
        return self.send_request(
            Request("GET", f"{self.base_url}/gpu/{gpu_uuid}/types").prepare()
        )

    def get_creatable_types(self, gpu_uuid: UUID) -> Any:
        return self.send_request(
            Request("GET", f"{self.base_url}/gpu/{gpu_uuid}/types/creatable").prepare()
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
            if r.status_code != codes.ok:
                try:
                    exc_name = r.json()["exception"]
                    raise DaemonException(
                        f"Operation failed on daemon due to: {exc_name}"
                    )
                except (JSONDecodeError, KeyError):
                    raise ExceptionDecodeFailed(
                        "Could not decode exception", r.content
                    ) from None
            try:
                json = r.json()
            except JSONDecodeError:
                raise ResponseDecodeFailed(
                    "Could not decode server response", r.content
                ) from None

            return json
