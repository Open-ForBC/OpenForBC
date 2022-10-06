# Copyright (c) 2021-2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: MIT

from fastapi import APIRouter

from openforbc.api.endpoint.gpu.mig import router as mig_router
from openforbc.api.endpoint.gpu.vpartition import router as vpartition_router

router = APIRouter()


router.include_router(mig_router, prefix="/mig")
router.include_router(vpartition_router, prefix="/vpart")
