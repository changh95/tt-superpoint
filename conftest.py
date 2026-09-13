# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Repo-local pytest fixtures for the SuperPoint device tests (``models/tests/test_superpoint.py``).

Why this file exists: ``code/models`` is a *regular* package (it has an ``__init__.py``), so in
any interpreter that also has a tt-metal checkout on ``sys.path`` it shadows tt-metal's
*namespace* ``models`` package -- a regular package wins over a namespace package regardless of
``sys.path`` order. tt-metal's own ``conftest.py`` imports ``models.demos...`` and therefore
cannot be loaded next to this repo (``pytest -p conftest`` fails with
``ModuleNotFoundError: No module named 'models.demos'``). This conftest provides the three
things the tests used to take from it:

* ``--device-id N`` -- default ``$TT_DEVICE_ID``, then ``$DEVICE_ID``, then ``0``;
* ``device_params`` -- the dict given by
  ``@pytest.mark.parametrize("device_params", [...], indirect=True)`` (``{}`` when absent);
* ``device`` -- function-scoped ``ttnn.CreateDevice(device_id=..., **device_params)``, set as the
  default device for the test and closed on teardown. These are the same calls tt-metal's
  ``device`` fixture makes (``get_updated_device_params`` adds the platform-default
  ``DispatchCoreConfig``, mirrored here), minus its CI bookkeeping, so benchmark numbers stay
  comparable with the rows in ``results.tsv`` that were measured under the tt-metal rootdir.

Run from ``code/`` (``pytest.ini`` pins the rootdir here, so this file is always picked up)::

    <tree>/python_env/bin/python -m pytest -s -q --device-id=0 \
        models/tests/test_superpoint.py::test_superpoint_fused

``ttnn`` is imported lazily, only inside the ``device`` fixture: collecting the tests or running
the host-only ``models/tests/test_fused_host.py`` never touches the driver.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

import pytest

DEVICE_ID_ENV_KEYS = ("TT_DEVICE_ID", "DEVICE_ID")


def resolve_device_id(cli_value: Optional[object], env: Optional[Mapping[str, str]] = None) -> int:
    """``--device-id`` if given, else the first non-empty of ``$TT_DEVICE_ID`` / ``$DEVICE_ID``,
    else 0. Pure function (unit-tested on the host in ``test_fused_host.py``)."""
    if cli_value is not None:
        return int(cli_value)
    env = os.environ if env is None else env
    for key in DEVICE_ID_ENV_KEYS:
        value = env.get(key)
        if value is not None and str(value).strip() != "":
            return int(value)
    return 0


def pytest_addoption(parser):
    parser.addoption(
        "--device-id",
        type=int,
        default=None,
        help="Blackhole chip id to open (default: $TT_DEVICE_ID, then $DEVICE_ID, then 0).",
    )


@pytest.fixture(scope="function")
def device_params(request):
    """Copy of the parametrized dict (indirect parametrization), ``{}`` when not parametrized."""
    return dict(getattr(request, "param", {}))


@pytest.fixture(scope="function")
def device(request, device_params):
    """One ``ttnn.CreateDevice(device_id=--device-id, **device_params)`` per test, set as the
    default device and closed on teardown (mirrors tt-metal's ``device`` fixture)."""
    import ttnn  # lazy: only a test that asks for the device loads the driver

    device_id = resolve_device_id(request.config.getoption("--device-id"))
    params = dict(device_params)
    # tt-metal's fixture routes every params dict through get_updated_device_params(), which adds
    # a default-constructed DispatchCoreConfig (WORKER cores, platform-default axis); mirror it.
    params.setdefault("dispatch_core_config", ttnn.DispatchCoreConfig())
    previous_default = ttnn.GetDefaultDevice()
    dev = ttnn.CreateDevice(device_id=device_id, **params)
    ttnn.SetDefaultDevice(dev)
    try:
        yield dev
    finally:
        ttnn.SetDefaultDevice(previous_default)
        ttnn.close_device(dev)
