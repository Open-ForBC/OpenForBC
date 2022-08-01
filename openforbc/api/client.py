# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT

from __future__ import annotations
from logging import getLogger
from typing import List, TYPE_CHECKING

from pydantic import parse_raw_as
from requests import JSONDecodeError, Session
from typer import Exit

from openforbc.api.url import DEFAULT_BASE_URL, GPU_ENDPOINT_PATH
from openforbc.gpu.generic import GPUPartitionType
from openforbc.gpu.model import GPUModel, GPUPartitionModel
from openforbc.gpu.nvidia.mig import GPUInstanceProfile, MIGModeStatus
from openforbc.gpu.nvidia.model import GPUInstanceModel

if TYPE_CHECKING:
    from typing import Any
    from uuid import UUID

    from requests import Response


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
        self.session = Session()
        self.cli = cli

    @classmethod
    def default(cls, cli: bool = True) -> APIClient:
        return cls(DEFAULT_BASE_URL, cli)

    def get_gpus(self) -> list[GPUModel]:
        return parse_raw_as(
            List[GPUModel], self.send_request("GET", f"{self.base_url}/gpu").text
        )

    def get_supported_types(
        self, gpu_uuid: UUID, creatable: bool = False
    ) -> list[GPUPartitionType]:
        return parse_raw_as(
            List[GPUPartitionType],
            self.send_gpu_request(
                gpu_uuid,
                "GET",
                "/types",
                {"creatable": int(creatable)},
            ).text,
        )

    def get_partitions(self, gpu_uuid: UUID) -> list[GPUPartitionModel]:
        return parse_raw_as(
            List[GPUPartitionModel],
            self.send_gpu_request(gpu_uuid, "GET", "/partition").text,
        )

    def create_partition(self, gpu_uuid: UUID, type_id: int) -> Any:
        return parse_raw_as(
            GPUPartitionModel,
            self.send_gpu_request(
                gpu_uuid, "POST", "/partition", data={"type_id": type_id}
            ).text,
        )

    def destroy_partition(self, gpu_uuid: UUID, partition_uuid: UUID) -> None:
        json = self.send_gpu_request(
            gpu_uuid, "DELETE", f"/partition/{partition_uuid}"
        ).json()
        assert "ok" in json
        assert json["ok"]

    def get_mig_mode(self, gpu_uuid: UUID) -> MIGModeStatus:
        return parse_raw_as(
            MIGModeStatus, self.send_gpu_request(gpu_uuid, "GET", "/mig/mode").text
        )

    def set_mig_mode(self, gpu_uuid: UUID, mode: MIGModeStatus) -> MIGModeStatus:
        return parse_raw_as(
            MIGModeStatus,
            self.send_gpu_request(gpu_uuid, "POST", "/mig/mode", {"mode": mode}).text,
        )

    def get_mig_profiles(self, gpu_uuid: UUID) -> list[GPUInstanceProfile]:
        return parse_raw_as(
            List[GPUInstanceProfile],
            self.send_gpu_request(gpu_uuid, "GET", "/mig/profile").text,
        )

    def get_mig_instances(self, gpu_uuid: UUID) -> list[GPUInstanceModel]:
        return parse_raw_as(
            List[GPUInstanceModel],
            self.send_gpu_request(gpu_uuid, "GET", "/mig/gi").text,
        )

    def create_gpu_instance(self, gpu_uuid: UUID, profile_id: int) -> GPUInstanceModel:
        return parse_raw_as(
            GPUInstanceModel,
            self.send_gpu_request(
                gpu_uuid, "POST", "/mig/gi", {"profile_id": profile_id}
            ).text,
        )

    def destroy_gpu_instance(self, gpu_uuid: UUID, instance_id: int) -> None:
        json = self.send_gpu_request(
            gpu_uuid, "DELETE", f"/mig/gi/{instance_id}"
        ).json()
        assert "ok" in json
        assert json["ok"]

    def send_request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] = {},
        data: dict[str, Any] = {},
    ) -> Response:
        from requests.status_codes import codes

        logger.debug("sending: %s %s params=%s data=%s", method, url, params, data)
        r = self.session.request(
            method, url if "://" in url else (self.base_url + url), params, data
        )
        logger.debug('received: "%s"', r.text)

        if r.status_code in [codes.bad_request, codes.unprocessable]:
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
        return r

    def send_gpu_request(
        self,
        gpu_uuid: UUID,
        method: str,
        url: str,
        params: dict[str, Any] = {},
        data: dict[str, Any] = {},
    ) -> Response:
        params["gpu_uuid"] = str(gpu_uuid)
        return self.send_request(
            method, f"{GPU_ENDPOINT_PATH}/{gpu_uuid}{url}", params, data
        )
