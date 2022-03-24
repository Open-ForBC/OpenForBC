from __future__ import annotations
from typing import TYPE_CHECKING
from uuid import UUID
from requests import Request

from openforbc.api import DEFAULT_BASE_URL

if TYPE_CHECKING:
    from requests import PreparedRequest


class APIClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    @classmethod
    def default(cls) -> APIClient:
        return cls(DEFAULT_BASE_URL)

    def get_gpus(self) -> PreparedRequest:
        return Request("GET", url=f"{self.base_url}/gpu").prepare()

    def get_supported_types(self, gpu_uuid: UUID) -> PreparedRequest:
        return Request("GET", f"{self.base_url}/gpu/{gpu_uuid}/types").prepare()

    def get_creatable_types(self, gpu_uuid: UUID) -> PreparedRequest:
        return Request(
            "GET", f"{self.base_url}/gpu/{gpu_uuid}/types/creatable"
        ).prepare()

    def get_partitions(self, gpu_uuid: UUID) -> PreparedRequest:
        return Request("GET", f"{self.base_url}/gpu/{gpu_uuid}/partition").prepare()

    def create_partition(self, gpu_uuid: UUID, type_id: int) -> PreparedRequest:
        return Request(
            "PUT",
            f"{self.base_url}/gpu/{gpu_uuid}/partition",
            data={"type_id": type_id},
        ).prepare()

    def destroy_partition(
        self, gpu_uuid: UUID, partition_uuid: UUID
    ) -> PreparedRequest:
        return Request(
            "DELETE", f"{self.base_url}/gpu/{gpu_uuid}/partition/{partition_uuid}"
        ).prepare()
