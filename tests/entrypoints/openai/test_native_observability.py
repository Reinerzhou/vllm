# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-server regression for telemetry parity and chunked prompt logprobs."""

import asyncio
import json

import httpx
import pytest

from ...utils import RemoteOpenAIServer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.asyncio
async def test_disconnected_stream_is_cancelled_preserving_observed_content(tmp_path):
    """Closing a real model stream before its terminal event is not a server error."""
    directory = tmp_path / "disconnect"
    args = [
        "--max-model-len",
        "512",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "512",
        "--enforce-eager",
        "--no-enable-prefix-caching",
    ]
    env = {
        "OBSERVABILITY_ENABLED": "1",
        "OBSERVABILITY_LOG_DIR": str(directory),
        "VLLM_USE_RUST_FRONTEND": "0",
    }
    request_id = "client-disconnect"
    with RemoteOpenAIServer(MODEL, args, env_dict=env) as server:
        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                server.url_for("v1/chat/completions"),
                headers={"X-Observability-Request-Id": request_id},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Count from one."}],
                    "max_tokens": 256,
                    "ignore_eos": True,
                    "temperature": 0,
                    "stream": True,
                },
            ) as response,
        ):
            response.raise_for_status()
            visible = False
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                assert line[5:].strip() != "[DONE]"
                data = json.loads(line[5:])
                if any(
                    c.get("delta", {}).get("content") for c in data.get("choices", [])
                ):
                    visible = True
                    break
            assert visible

        async def finished():
            while True:
                for path in directory.rglob("*.jsonl"):
                    for line in path.read_text().splitlines():
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if (
                            row.get("request_id") == request_id
                            and row["event"] == "request_finished"
                        ):
                            return row
                await asyncio.sleep(0.01)

        row = await asyncio.wait_for(finished(), timeout=10)
        assert row["outcome"] == "cancelled", row
        assert row["content_first_seconds"] is not None
        assert row["protocol_done"] is False
        assert row["reasoning_tokens_complete"] is False


@pytest.mark.asyncio
async def test_lifecycle_preserves_multi_choice_output_and_chunked_logprobs(tmp_path):
    """A zero-logit final chunk must return accumulated prompt logprobs."""
    outputs = []
    for enabled in ("0", "1"):
        directory = tmp_path / enabled
        args = [
            "--max-model-len",
            "128",
            "--max-num-seqs",
            "2",
            "--max-num-batched-tokens",
            "4",
            "--enable-chunked-prefill",
            "--no-enable-prefix-caching",
            "--enforce-eager",
        ]
        env = {
            "OBSERVABILITY_ENABLED": enabled,
            "OBSERVABILITY_LOG_DIR": str(directory),
            "OBSERVABILITY_ROLE": "vllm",
            "VLLM_LOGGING_LEVEL": "ERROR",
            "VLLM_USE_RUST_FRONTEND": "0",
            "VLLM_BATCH_INVARIANT": "1",
        }
        with RemoteOpenAIServer(MODEL, args, env_dict=env) as server:
            async with server.get_async_client() as client:
                stream = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": "Say hello."}],
                    temperature=0,
                    max_tokens=4,
                    n=2,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_headers={"X-Observability-Request-Id": "multi-choice"},
                )
                text = ["", ""]
                async for event in stream:
                    for choice in event.choices:
                        text[choice.index] += choice.delta.content or ""
                logprobs = []
                for prompt in ([42], [42] * 5):
                    response = await client.completions.create(
                        model=MODEL,
                        prompt=prompt,
                        max_tokens=1,
                        temperature=0,
                        extra_body={"prompt_logprobs": 1},
                    )
                    values = response.choices[0].prompt_logprobs
                    assert values is not None
                    assert len(values) == len(prompt)
                    assert values[0] is None
                    logprobs.append(values)
                response_text = ""
                response_stream = await client.responses.create(
                    model=MODEL,
                    input="Say hello.",
                    max_output_tokens=4,
                    temperature=0,
                    stream=True,
                    extra_headers={"X-Observability-Request-Id": "responses-request"},
                )
                async for response_event in response_stream:
                    if response_event.type == "response.output_text.delta":
                        response_text += response_event.delta
                async with httpx.AsyncClient() as raw:
                    messages_text = ""
                    async with raw.stream(
                        "POST",
                        server.url_for("v1/messages"),
                        json={
                            "model": MODEL,
                            "messages": [{"role": "user", "content": "Say hello."}],
                            "temperature": 0,
                            "max_tokens": 4,
                            "stream": True,
                        },
                        headers={"X-Observability-Request-Id": "messages-request"},
                    ) as messages:
                        messages.raise_for_status()
                        async for line in messages.aiter_lines():
                            if line.startswith("data:"):
                                data = json.loads(line[5:])
                                assert data["type"] != "error", data
                                delta = data.get("delta") or {}
                                if delta.get("type") == "text_delta":
                                    messages_text += delta.get("text", "")
                    batch = await raw.post(
                        server.url_for("v1/chat/completions/batch"),
                        json={
                            "model": MODEL,
                            "messages": [
                                [{"role": "user", "content": "Say hello."}],
                                [{"role": "user", "content": "Say goodbye."}],
                            ],
                            "temperature": 0,
                            "max_tokens": 4,
                        },
                        headers={"X-Observability-Request-Id": "batch-request"},
                    )
                    batch.raise_for_status()
                batch_text = [c["message"]["content"] for c in batch.json()["choices"]]
                assert len(batch_text) == 2
                outputs.append(
                    (text, logprobs, batch_text, response_text, messages_text)
                )
                if enabled == "1":

                    async def read_finished(directory, request_id):
                        while True:
                            rows = []
                            for path in directory.rglob("*.jsonl"):
                                for line in path.read_text().splitlines():
                                    try:
                                        rows.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        continue
                            matched = [
                                r for r in rows if r.get("request_id") == request_id
                            ]
                            finished = [
                                r for r in matched if r["event"] == "request_finished"
                            ]
                            if finished:
                                return matched, finished[0]
                            await asyncio.sleep(0.01)

                    rows, finished = await asyncio.wait_for(
                        read_finished(directory, "multi-choice"), timeout=10
                    )
                    assert finished["outcome"] == "success"
                    assert finished["content_first_seconds"] is not None
                    submitted = [r for r in rows if r["event"] == "engine_submit"]
                    assert len(submitted) == 2
                    for request_id in ("responses-request", "messages-request"):
                        _, finished = await asyncio.wait_for(
                            read_finished(directory, request_id), timeout=10
                        )
                        assert finished["outcome"] == "success"
                        assert finished["content_first_seconds"] is not None
                        if request_id == "messages-request":
                            assert finished["reasoning_tokens"] is None
                            assert finished["reasoning_tokens_complete"] is False
                    assert len({r["engine_request_id"] for r in submitted}) == 2
                    rows, finished = await asyncio.wait_for(
                        read_finished(directory, "batch-request"), timeout=10
                    )
                    assert finished["outcome"] == "success"
                    assert finished["reasoning_first_seconds"] is None
                    submitted = [r for r in rows if r["event"] == "engine_submit"]
                    assert len(submitted) == 2
        if enabled == "0":
            assert not list(directory.rglob("*.jsonl"))
    assert outputs[0] == outputs[1]
