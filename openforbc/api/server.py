from __future__ import annotations
from flask import abort, Flask, jsonify, request
from pynvml import NVMLError
from typing import TYPE_CHECKING
from uuid import UUID

from openforbc.gpu import GPU

if TYPE_CHECKING:
    from flask import Response
    from flask.typing import ResponseReturnValue

app = Flask(__name__)


@app.errorhandler(Exception)
def handle_exception(e: Exception) -> ResponseReturnValue:
    if isinstance(e, NVMLError):
        app.logger.error("NVML error", exc_info=True)
    else:
        app.logger.error("Generic exception", exc_info=True)

    return jsonify({"exception": repr(e)}), 500


@app.route("/gpu")
def list_gpus() -> Response:
    return jsonify(
        [
            {"name": gpu.name, "uuid": gpu.uuid, "pciid": gpu.pciid.__repr__()}
            for gpu in GPU.get_gpus()
        ]
    )


@app.route("/gpu/<uuid>/types")
def list_gpu_supported_types(uuid: UUID) -> Response:
    gpu = GPU.from_uuid(uuid)
    return jsonify(gpu.get_supported_types())


@app.route("/gpu/<uuid>/types/creatable")
def list_gpu_creatable_types(uuid: UUID) -> Response:
    gpu = GPU.from_uuid(uuid)
    return jsonify(gpu.get_creatable_types())


@app.route("/gpu/<uuid>/partition")
def list_gpu_partitions(uuid: UUID) -> Response:
    gpu = GPU.from_uuid(uuid)

    return jsonify(
        [
            {"uuid": partition.uuid, "type_id": partition.type.id}
            for partition in gpu.get_partitions()
        ]
    )


@app.put("/gpu/<uuid>/partition")
def create_gpu_partition(uuid: UUID) -> Response:
    if "type_id" not in request.form:
        return abort(400)

    gpu = GPU.from_uuid(uuid)
    type_id: int = int(request.form["type_id"])

    part_type = next((x for x in gpu.get_supported_types() if x.id == type_id))
    if not part_type:
        return abort(400, "unsupported partition type")

    if not next((True for x in gpu.get_creatable_types() if x.id == type_id), False):
        return abort(400, "unavailable partition type")

    partition = gpu.create_partition(part_type)

    return jsonify({"ok": True, "uuid": partition.uuid})


@app.delete("/gpu/<uuid:uuid>/partition/<uuid:p_uuid>")
def delete_gpu_partition(uuid: UUID, p_uuid: UUID) -> Response:
    gpu = GPU.from_uuid(uuid)

    print(uuid)
    print(p_uuid)
    partition = next((x for x in gpu.get_partitions() if x.uuid == p_uuid), None)
    if partition is None:
        return abort(404, "partition with specified uuid does not exist")

    partition.destroy()

    return jsonify({"ok": True})


def run() -> None:
    app.run()


if __name__ == "__main__":
    app.run()
