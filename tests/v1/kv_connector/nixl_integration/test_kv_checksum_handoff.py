# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real P/D OpenAI servers with real model weights.

Both must enable KV checksum and failure policy=fail, with D prefix cache off.
Set both URLs for dedicated servers, or start the two-GPU fixture automatically.
This tests the real admission/failure path, not actual KV byte corruption.
"""

import json
import os
import uuid
from contextlib import ExitStack

import numpy as np
import pybase64 as base64
import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.distributed.kv_transfer.kv_checksum.scheduler import KVChecksumPayload
from vllm.utils.network_utils import get_open_port


@pytest.fixture(scope="module")
def checksum_servers():
    prefill = os.environ.get("CHECKSUM_PREFILL_URL")
    decode = os.environ.get("CHECKSUM_DECODE_URL")
    if prefill is not None or decode is not None:
        assert prefill and decode, "Set both URLs, or use the two-GPU local fixture"
        yield prefill.rstrip("/"), decode.rstrip("/")
        return

    model = os.environ.get("CHECKSUM_TEST_MODEL", "Qwen/Qwen3-0.6B")
    with ExitStack() as stack:
        urls = []
        for gpu, role in enumerate(("kv_producer", "kv_consumer")):
            config = {
                "kv_connector": "NixlConnector",
                "kv_role": role,
                "kv_port": get_open_port(),
                "enable_kv_checksum": True,
                "kv_load_failure_policy": "fail",
            }
            server = stack.enter_context(
                RemoteOpenAIServer(
                    model,
                    [
                        "--max-model-len",
                        "1024",
                        "--enforce-eager",
                        "--enable-extensible-kv-cache",
                        "--no-enable-prefix-caching",
                        "--kv-transfer-config",
                        json.dumps(config),
                    ],
                    env_dict={
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "VLLM_USE_V2_MODEL_RUNNER": "1",
                        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
                        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(get_open_port()),
                    },
                    max_wait_seconds=600,
                )
            )
            urls.append(server.url_root)
        yield tuple(urls)


@pytest.mark.parametrize(
    "endpoint", ["completions", "chat/completions", "responses", "messages"]
)
@pytest.mark.parametrize("fault", ["none", "mismatch", "missing", "malformed"])
def test_real_nixl_checksum_handoff_and_request_failure(
    checksum_servers, fault, endpoint
):
    prefill, decode = checksum_servers
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"}
    models = requests.get(f"{prefill}/v1/models", headers=headers, timeout=30)
    models.raise_for_status()
    model = models.json()["data"][0]["id"]
    prompt = (f"Checksum integration case {fault}. " * 32) + "Explain one cache block."

    def body(max_tokens):
        value = {"model": model, "temperature": 0, "stream": False}
        if endpoint == "responses":
            value.update(input=prompt, max_output_tokens=max_tokens)
        elif endpoint == "completions":
            value.update(prompt=prompt, max_tokens=max_tokens)
        else:
            value.update(
                messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens
            )
        return value

    def handoff(corrupt):
        request_headers = {
            **headers,
            "X-Observability-Request-Id": f"checksum-{uuid.uuid4().hex}",
        }
        response = requests.post(
            f"{prefill}/v1/{endpoint}",
            headers=request_headers,
            json={
                **body(1),
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "remote_engine_id": None,
                    "remote_block_ids": None,
                    "remote_host": None,
                    "remote_port": None,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        params = response.json()["kv_transfer_params"]
        assert params.get("remote_block_ids"), "Test must export real KV blocks"
        assert params.get("kv_checksums"), "P must actually produce a checksum"
        if corrupt == "missing":
            del params["kv_checksums"]
        elif corrupt == "malformed":
            params["kv_checksums"] = "not-a-checksum"
        elif corrupt == "mismatch":
            payload = KVChecksumPayload.from_bytes(
                base64.b64decode(params["kv_checksums"])
            )
            assert payload.groups, "Test needs a checksummed model group"
            for key, (fingerprint, size, start, values) in payload.groups.items():
                changed = values.copy()
                changed ^= np.uint64(1)
                payload.groups[key] = (fingerprint, size, start, changed)
            params["kv_checksums"] = base64.b64encode_as_string(payload.to_bytes())
        return requests.post(
            f"{decode}/v1/{endpoint}",
            headers=request_headers,
            json={
                **body(8),
                "kv_transfer_params": params,
            },
            timeout=300,
        )

    response = handoff(fault)
    if fault == "none":
        assert response.status_code == 200, response.text
        assert response.json().get("id")
    else:
        assert response.status_code == 500, response.text
        assert response.json().get("error"), response.text
        healthy = handoff("none")
        assert healthy.status_code == 200, healthy.text
        assert healthy.json().get("id")
