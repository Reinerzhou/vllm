# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process diagnostics independent of model initialization and device state."""

import faulthandler
import logging
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def coordinator_fault_handler(logger: logging.Logger) -> Iterator[None]:
    """Keep the coordinator's fault file open until the handler is disabled."""
    path = os.getenv("VLLM_DP_COORDINATOR_FAULT_LOG") or (
        f"/tmp/vllm_dp_coordinator_fault_{os.getpid()}.log"
    )
    with ExitStack() as cleanup:
        try:
            stream = cleanup.enter_context(open(path, "a", buffering=1))
            faulthandler.enable(file=stream, all_threads=True)
        except (OSError, RuntimeError):
            cleanup.close()
            logger.warning(
                "Cannot initialize coordinator fault log: %s", path, exc_info=True
            )
        else:
            cleanup.callback(faulthandler.disable)
            logger.info("DP Coordinator fault handler enabled: %s", path)
        yield


def process_exit_details(pid: int, exitcode: int | None) -> dict[str, Any]:
    """Read safe process status without waiting or inspecting argv/environment."""
    details: dict[str, Any] = {
        "pid": pid,
        "parent_pid": os.getpid(),
        "parent_ppid": os.getppid(),
        "exitcode": exitcode,
        "signal": -exitcode if exitcode is not None and exitcode < 0 else None,
    }
    for key, getter in (("pgid", os.getpgid), ("sid", os.getsid)):
        try:
            details[key] = getter(pid)
        except OSError:
            details[key] = None
    directory = Path("/proc") / str(pid)
    try:
        fields = dict(
            line.split(":", 1)
            for line in (directory / "status").read_text().splitlines()
            if ":" in line
        )
        for key in ("State", "Threads", "VmRSS"):
            if key in fields:
                details[key] = fields[key].strip()
        details["wait_channel"] = (directory / "wchan").read_text().strip()
    except OSError:
        pass
    return details
