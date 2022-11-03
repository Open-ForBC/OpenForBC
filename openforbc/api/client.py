# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT
from __future__ import annotations

from logging import getLogger
from typing import List, TYPE_CHECKING, overload

from pydantic import parse_raw_as
from requests import JSONDecodeError, Session
from typer import Exit

from openforbc.api.url import DEFAULT_BASE_URL, GPU_ENDPOINT_PATH
from openforbc.gpu.generic import GPUPartitionType
from openforbc.gpu.model import GPUModel, GPUPartitionModel
from openforbc.gpu.nvidia.mig import (
    ComputeInstanceProfile,
    GPUInstanceProfile,
    MIGModeStatus,
)
from openforbc.gpu.nvidia.model import ComputeInstanceModel, GPUInstanceModel

if TYPE_CHECKING:
    from typing import Any, Literal, Optional, Type, TypeVar, Union
    from uuid import UUID

    from requests import Response

    T = TypeVar("T")


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
        return self.send_gpu_request(
            gpu_uuid,
            "GET",
            "/types",
            {"creatable": int(creatable)},
            type=List[GPUPartitionType],
        )

    def get_partitions(self, gpu_uuid: UUID) -> list[GPUPartitionModel]:
        return self.send_gpu_request(
            gpu_uuid, "GET", "/partition", type=List[GPUPartitionModel]
        )

    def create_partition(self, gpu_uuid: UUID, type_id: int) -> Any:
        return self.send_gpu_request(
            gpu_uuid, "POST", "/partition", {"type_id": type_id}, type=GPUPartitionModel
        )

    def destroy_partition(self, gpu_uuid: UUID, partition_uuid: UUID) -> None:
        json = self.send_gpu_request(
            gpu_uuid, "DELETE", f"/partition/{partition_uuid}", json=True
        )
        assert "ok" in json
        assert json["ok"]

    def get_mig_mode(self, gpu_uuid: UUID) -> MIGModeStatus:
        return self.send_gpu_request(gpu_uuid, "GET", "/mig/mode", type=MIGModeStatus)

    def set_mig_mode(self, gpu_uuid: UUID, mode: MIGModeStatus) -> MIGModeStatus:
        return self.send_gpu_request(
            gpu_uuid, "POST", "/mig/mode", {"mode": mode}, type=MIGModeStatus
        )

    def get_compute_instances(self, gpu_uuid: UUID) -> list[ComputeInstanceModel]:
        return self.send_gpu_request(
            gpu_uuid, "GET", "/mig/ci", type=List[ComputeInstanceModel]
        )

    def get_gi_profiles(self, gpu_uuid: UUID) -> list[GPUInstanceProfile]:
        return self.send_gpu_request(
            gpu_uuid, "GET", "/mig/gi/profile", type=List[GPUInstanceProfile]
        )

    def get_gpu_instance_profile_capacity(self, gpu_uuid: UUID, gip_id: int) -> int:
        return self.send_gpu_request(
            gpu_uuid, "GET", f"/mig/gi/profile/{gip_id}/capacity", type=int
        )

    def get_gpu_instances(self, gpu_uuid: UUID) -> list[GPUInstanceModel]:
        return self.send_gpu_request(
            gpu_uuid, "GET", "/mig/gi", type=List[GPUInstanceModel]
        )

    def create_gpu_instance(
        self, gpu_uuid: UUID, gip_id: int, default_ci: bool = True
    ) -> GPUInstanceModel:
        return self.send_gpu_request(
            gpu_uuid,
            "POST",
            "/mig/gi",
            {"gip_id": gip_id, "default_ci": default_ci},
            type=GPUInstanceModel,
        )

    def destroy_gpu_instance(self, gpu_uuid: UUID, instance_id: int) -> None:
        json = self.send_gpu_request(
            gpu_uuid, "DELETE", f"/mig/gi/{instance_id}", json=True
        )
        assert "ok" in json
        assert json["ok"]

    def get_compute_instance_profiles(
        self, gpu_uuid: UUID, gi_id: int
    ) -> list[ComputeInstanceProfile]:
        return self.send_gpu_request(
            gpu_uuid,
            "GET",
            f"/mig/gi/{gi_id}/ci/profile",
            type=List[ComputeInstanceProfile],
        )

    def get_compute_instance_profile_capacity(
        self, gpu_uuid: UUID, gi_id: int, cip_id: int
    ) -> int:
        return self.send_gpu_request(
            gpu_uuid, "GET", f"/mig/gi/{gi_id}/ci/profile/{cip_id}/capacity", type=int
        )

    def create_compute_instance(
        self, gpu_uuid: UUID, gi_id: int, cip_id: int
    ) -> ComputeInstanceModel:
        return self.send_gpu_request(
            gpu_uuid,
            "POST",
            f"/mig/gi/{gi_id}/ci",
            {"cip_id": cip_id},
            type=ComputeInstanceModel,
        )

    def get_gi_compute_instances(
        self, gpu_uuid: UUID, gi_id: int
    ) -> list[ComputeInstanceModel]:
        return self.send_gpu_request(
            gpu_uuid, "GET", f"/mig/gi/{gi_id}/ci", type=List[ComputeInstanceModel]
        )

    def destroy_compute_instance(self, gpu_uuid: UUID, gi_id: int, ci_id: int) -> None:
        json = self.send_gpu_request(
            gpu_uuid, "DELETE", f"/mig/gi/{gi_id}/ci/{ci_id}", json=True
        )
        assert "ok" in json
        assert json["ok"]

    @overload
    def send_request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] = {},
        data: dict[str, Any] = {},
        json: Literal[False] = False,
        type: Literal[None] = None,
    ) -> Response:
        ...

    @overload
    def send_request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] = {},
        data: dict[str, Any] = {},
        json: Literal[True] = True,
        type: Literal[None] = None,
    ) -> Any:
        ...

    @overload
    def send_request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] = {},
        data: dict[str, Any] = {},
        json: Literal[False] = False,
        type: Optional[Type[T]] = None,
    ) -> T:
        ...

    def send_request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] = {},
        data: dict[str, Any] = {},
        json: bool = False,
        type: Optional[Type[T]] = None,
    ) -> Union[Response, Any, T]:
        from requests.status_codes import codes

        if json:
            assert type is None
        if type is not None:
            assert not json

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
        if json:
            return r.json()
        if type is not None:
            return parse_raw_as(type, r.text)

        return r

    @overload
    def send_gpu_request(
        self,
        gpu_uuid: UUID,
        method: str,
        url: str,
        params: dict[str, Any] = {},
        data: dict[str, Any] = {},
        json: Literal[False] = False,
        type: Literal[None] = None,
    ) -> Response:
        ...

    @overload
    def send_gpu_request(
        self,
        gpu_uuid: UUID,
        method: str,
        url: str,
        params: dict[str, Any] = {},
        data: dict[str, Any] = {},
        json: Literal[True] = True,
        type: Literal[None] = None,
    ) -> Any:
        ...

    @overload
    def send_gpu_request(
        self,
        gpu_uuid: UUID,
        method: str,
        url: str,
        params: dict[str, Any] = {},
        data: dict[str, Any] = {},
        json: Literal[False] = False,
        type: Optional[Type[T]] = None,
    ) -> T:
        ...

    def send_gpu_request(
        self,
        gpu_uuid: UUID,
        method: str,
        url: str,
        params: dict[str, Any] = {},
        data: dict[str, Any] = {},
        json: bool = False,
        type: Optional[Type[T]] = None,
    ) -> Union[Response, Any, T]:
        params["gpu_uuid"] = str(gpu_uuid)

        if json:
            assert type is None
            return self.send_request(
                method, f"{GPU_ENDPOINT_PATH}/{gpu_uuid}{url}", params, data, json, type
            )
        if type is not None:
            assert not json
            return self.send_request(
                method, f"{GPU_ENDPOINT_PATH}/{gpu_uuid}{url}", params, data, json, type
            )

        return self.send_request(
            method, f"{GPU_ENDPOINT_PATH}/{gpu_uuid}{url}", params, data, json, type
        )
