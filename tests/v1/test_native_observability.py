# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle I/O contracts using the real writer, middleware and HTTP stack."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "vllm/observability.py"
SPEC = importlib.util.spec_from_file_location("observability", SOURCE)
assert SPEC is not None and SPEC.loader is not None
obs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(obs)
LOAD = f"""
import importlib.util, json
from pathlib import Path
spec = importlib.util.spec_from_file_location('observability', {str(SOURCE)!r})
obs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(obs)
"""


class SummaryTests(unittest.TestCase):
    def test_all_endpoint_summaries_match_shared_contract(self):
        cases = json.loads(
            Path(__file__).with_name("native_observability_cases.json").read_text()
        )
        for case in cases:
            with self.subTest(case=case["name"]):
                summary = obs._ReasoningSummary(0.0)
                first = None
                for item in case["events"]:
                    visible = summary.observe(
                        item["data"], item["at"], case["streaming"]
                    )
                    if visible and first is None:
                        first = item["at"]
                actual = summary.fields(case["success"])
                actual["first_output_seconds"] = first
                self.assertEqual(actual, case["expected"])

    def test_long_reasoning_and_request_wide_choices_keep_final_usage(self):
        for field in ("reasoning", "reasoning_content"):
            summary = obs._ReasoningSummary(100.0)
            summary.observe({"choices": [{"delta": {field: ""}}]}, 100.01)
            summary.observe({"choices": [{"delta": {field: "a"}}]}, 100.05)
            summary.observe(
                {
                    "choices": [{"delta": {field: "b"}}, {"delta": {"reasoning": "c"}}],
                    "usage": {"completion_tokens_details": {"reasoning_tokens": 999}},
                },
                139.0,
            )
            summary.observe({"choices": [{"delta": {"content": "answer"}}]}, 140.0)
            summary.observe(
                {
                    "choices": [],
                    "usage": {"completion_tokens_details": {"reasoning_tokens": 12}},
                },
                140.1,
            )
            fields = summary.fields(True)
            self.assertAlmostEqual(fields["reasoning_first_seconds"], 0.05)
            self.assertEqual(fields["reasoning_last_seconds"], 39.0)
            self.assertEqual(fields["content_first_seconds"], 40.0)
            self.assertAlmostEqual(fields["reasoning_to_content_seconds"], 39.95)
            self.assertEqual(fields["reasoning_tokens"], 12)
            self.assertTrue(fields["reasoning_tokens_complete"])
            failed = summary.fields(False)
            self.assertEqual(failed["reasoning_last_seconds"], 39.0)
            self.assertIsNone(failed["reasoning_tokens"])
            self.assertFalse(failed["reasoning_tokens_complete"])

    def test_unknown_zero_interleaving_and_nonstream_are_distinct(self):
        summary = obs._ReasoningSummary(0.0)
        self.assertIsNone(summary.fields(True)["reasoning_tokens"])
        summary.observe({"choices": [{"delta": {"content": "first"}}]}, 1.0)
        summary.observe({"choices": [{"delta": {"reasoning": "later"}}]}, 2.0)
        self.assertIsNone(summary.fields(True)["reasoning_to_content_seconds"])
        for value in (0, -1, True, None, "12"):
            summary.observe(
                {
                    "choices": [],
                    "usage": {"completion_tokens_details": {"reasoning_tokens": value}},
                },
                3.0,
            )
            fields = summary.fields(True)
            self.assertEqual(
                fields["reasoning_tokens"],
                0 if type(value) is int and value == 0 else None,
            )
        summary = obs._ReasoningSummary(0.0)
        summary.observe(
            {
                "choices": [{"message": {"reasoning": "hidden"}}],
                "usage": {"completion_tokens_details": {"reasoning_tokens": 8}},
            },
            40.0,
            streaming=False,
        )
        self.assertIsNone(summary.fields(True)["reasoning_first_seconds"])
        self.assertEqual(summary.fields(True)["reasoning_tokens"], 8)


class FileTests(unittest.TestCase):
    def test_concurrent_collisions_never_overwrite_files_or_links(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            stamp = datetime(2026, 11, 2, tzinfo=UTC)
            old = directory / "20261102T000000.000000Z.42.jsonl"
            old.write_text("protected")
            (directory / "20261102T000000.000000Z.42.1.jsonl").symlink_to(old)

            def create(number):
                with obs._open_log_file(directory, 42, stamp) as stream:
                    stream.write(str(number).encode())
                    return Path(stream.name)

            with ThreadPoolExecutor(max_workers=8) as pool:
                paths = list(pool.map(create, range(16)))
            self.assertEqual(len(set(paths)), 16)
            self.assertEqual(
                {p.read_text() for p in paths}, {str(n) for n in range(16)}
            )
            self.assertEqual(old.read_text(), "protected")

    def test_retention_only_deletes_owned_formats_and_compatible_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            legacy = directory / "vllm"
            legacy.mkdir()
            other = directory / "d0"
            other.mkdir()
            name = "20000101T000000.000000Z.42.jsonl"
            expired = [
                directory / name,
                legacy / "vllm.20000101T000000.000000Z.42.1234abcd.jsonl",
            ]
            keep = [other / name, directory / "manual.jsonl"]
            for path in expired + keep:
                path.write_text("original")
                os.utime(path, (1, 1))
            link = directory / "20000101T000000.000000Z.42.1.jsonl"
            link.symlink_to(keep[1])
            obs._cleanup_logs(directory, 0, legacy_default=True)
            self.assertTrue(all(p.exists() for p in expired))
            obs._cleanup_logs(directory, 86400, legacy_default=True)
            self.assertFalse(any(p.exists() for p in expired))
            self.assertTrue(all(p.exists() for p in keep))
            self.assertTrue(link.is_symlink())


class MiddlewareTests(unittest.TestCase):
    def test_real_asgi_disconnects_preserve_terminal_usage_and_application_errors(self):
        program = (
            LOAD
            + r"""
import asyncio
import os
from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse

reason = b'data: {"choices":[{"delta":{"reasoning":"r"}}]}\n\n'
terminal = (
    b'data: {"choices":[],"usage":'
    b'{"completion_tokens_details":{"reasoning_tokens":12}}}\n\n'
    b'data: [DONE]\n\n'
)
expected = {}


async def scenario(mode, version="2.4"):
    case = mode + "-" + version
    event = asyncio.Event()
    disconnected = {"type": "http.disconnect"}
    sent_terminal = False

    async def receive():
        await event.wait()
        return disconnected

    async def send(message):
        nonlocal sent_terminal
        if message["type"] != "http.response.body" or not message.get("body"):
            return
        body = message["body"]
        if mode == "pre" or mode == "protocol_error":
            if version == "2.4":
                raise OSError("controlled closed transport")
            asyncio.get_running_loop().call_soon(event.set)
        elif mode == "post":
            if sent_terminal and version == "2.4":
                raise OSError("controlled close after terminal")
            if b"[DONE]" in body:
                sent_terminal = True
                if version == "2.3":
                    asyncio.get_running_loop().call_soon(event.set)
        elif mode == "failed_terminal" and b"[DONE]" in body:
            raise OSError("terminal was not delivered")
        elif mode == "nonstream_failed":
            raise OSError("nonstream was not delivered")

    async def chunks():
        yield (
            b'data: {"error":{"type":"controlled"}}\n\n'
            if mode == "protocol_error"
            else reason
        )
        if mode in ("pre", "protocol_error"):
            await asyncio.Event().wait()
        yield terminal
        yield b": after terminal\n\n"
        await asyncio.Event().wait()

    async def application(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (
                        b"content-type",
                        b"application/json"
                        if mode.startswith("nonstream")
                        else b"text/event-stream",
                    )
                ],
            }
        )
        if mode.startswith("nonstream"):
            await send(
                {
                    "type": "http.response.body",
                    "body": (
                        b'{"usage":{"completion_tokens_details":'
                        b'{"reasoning_tokens":12}}}'
                    ),
                    "more_body": False,
                }
            )
            event.set()
            assert await receive() is disconnected
            return
        await send({"type": "http.response.body", "body": reason, "more_body": True})
        if mode == "cancel_post":
            await send(
                {"type": "http.response.body", "body": terminal, "more_body": True}
            )
        if mode == "application_error":
            event.set()
            assert await receive() is disconnected
            raise RuntimeError("genuine application error")
        raise asyncio.CancelledError()

    app = (
        application
        if mode
        in (
            "application_error",
            "cancel_pre",
            "cancel_post",
            "nonstream",
            "nonstream_failed",
        )
        else StreamingResponse(chunks(), media_type="text/event-stream")
    )
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": version},
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"x-observability-request-id", case.encode())],
    }
    try:
        await asyncio.wait_for(
            obs.RequestObservabilityMiddleware(app)(scope, receive, send), 5
        )
    except TimeoutError:
        raise
    except (ClientDisconnect, OSError, RuntimeError, asyncio.CancelledError):
        pass
    successful = mode in ("post", "cancel_post", "nonstream")
    expected[case] = (
        "success"
        if successful
        else "error"
        if mode in ("protocol_error", "application_error")
        else "cancelled",
        12 if successful else None,
    )


async def main():
    for version in ("2.3", "2.4"):
        for mode in ("pre", "post"):
            await scenario(mode, version)
    for mode in (
        "failed_terminal",
        "protocol_error",
        "application_error",
        "cancel_pre",
        "cancel_post",
        "nonstream",
        "nonstream_failed",
    ):
        await scenario(mode)


asyncio.run(main())
obs.runtime().close()
rows = [
    json.loads(line)
    for path in Path(os.environ["OBSERVABILITY_LOG_DIR"]).rglob("*.jsonl")
    for line in path.read_text().splitlines()
]
for case, (outcome, tokens) in expected.items():
    end = next(
        row
        for row in rows
        if row["event"] == "request_finished" and row["request_id"] == case
    )
    assert end["outcome"] == outcome, (case, end)
    assert end["reasoning_tokens"] == tokens, (case, end)
    assert end["reasoning_tokens_complete"] == (tokens is not None), (case, end)
    if not case.startswith(("nonstream", "protocol_error")):
        assert end["reasoning_first_seconds"] is not None, (case, end)
    if case.startswith(("failed_terminal", "nonstream_failed")):
        assert not end["response_complete"], (case, end)
"""
        )
        with tempfile.TemporaryDirectory() as temp:
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith("OBSERVABILITY_")
            }
            env.update(OBSERVABILITY_ENABLED="1", OBSERVABILITY_LOG_DIR=temp)
            subprocess.run(
                [sys.executable, "-c", program], env=env, check=True, timeout=30
            )

    def test_all_endpoints_preserve_fragmented_stream_and_nonstream_bytes(self):
        cases = Path(__file__).with_name("native_observability_cases.json")
        program = (
            LOAD
            + f"\nCASES = json.loads(Path({str(cases)!r}).read_text())\n"
            + """
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.testclient import TestClient
app = FastAPI()
app.add_middleware(obs.RequestObservabilityMiddleware)
cases = {c['name']: c for c in CASES}
def wire(case):
    if not case['streaming']:
        return json.dumps(case['events'][-1]['data']).encode()
    chunks = [b'data: ' + json.dumps(item['data']).encode() + b'\\n\\n'
              for item in case['events']]
    if case['route'] == '/v1/chat/completions':
        chunks.append(b'data: [DONE]\\n\\n')
    return b''.join(chunks)
@app.post('/{path:path}')
async def endpoint(request: Request):
    case = cases[request.headers['x-fixture-case']]
    data = wire(case)
    if case['streaming']:
        return StreamingResponse(iter(data[i:i+17] for i in range(0, len(data), 17)),
                                 media_type='text/event-stream')
    return Response(data, media_type='application/json')
with TestClient(app) as client:
    for case in CASES:
        response = client.post(case['route'], headers={
            'X-Fixture-Case': case['name'],
            'X-Observability-Request-Id': case['name']})
        assert response.content == wire(case), case['name']
obs.runtime().close()
rows = [json.loads(line) for p in Path(__import__('os').environ[
    'OBSERVABILITY_LOG_DIR']).rglob('*.jsonl') for line in p.read_text().splitlines()]
for case in CASES:
    end = next(r for r in rows if r['event'] == 'request_finished'
               and r['request_id'] == case['name'])
    assert end['outcome'] == ('success' if case['success'] else 'error'), case['name']
    for name, expected in case['expected'].items():
        if name == 'first_output_seconds':
            count = sum(r['event'] == 'first_output' and r['request_id'] == case['name']
                        for r in rows)
            assert count == (0 if expected is None else 1), case['name']
        elif name in ('reasoning_tokens', 'reasoning_tokens_complete'):
            assert end[name] == expected, (case['name'], name, end)
        else:
            assert (end[name] is None) == (expected is None), (case['name'], name, end)
"""
        )
        with tempfile.TemporaryDirectory() as temp:
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith("OBSERVABILITY_")
            }
            env.update(OBSERVABILITY_ENABLED="1", OBSERVABILITY_LOG_DIR=temp)
            result = subprocess.run(
                [sys.executable, "-c", program],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_nonstream_batch_has_lifecycle_without_inferred_stage_times(self):
        program = (
            LOAD
            + """
from fastapi import FastAPI
from fastapi.testclient import TestClient
app = FastAPI()
app.add_middleware(obs.RequestObservabilityMiddleware)
@app.post('/v1/chat/completions/batch')
async def batch():
    return {'choices': [{'message': {'content': 'one'}},
                        {'message': {'content': 'two'}}],
            'usage': {'completion_tokens_details': {'reasoning_tokens': 0}}}
with TestClient(app) as client:
    response = client.post('/v1/chat/completions/batch', headers={
        'X-Observability-Request-Id': 'batch-canonical'})
assert len(response.json()['choices']) == 2
obs.runtime().close()
rows = [json.loads(line) for p in Path(__import__('os').environ[
    'OBSERVABILITY_LOG_DIR']).rglob('*.jsonl') for line in p.read_text().splitlines()]
end = next(r for r in rows if r['event'] == 'request_finished')
assert end['request_id'] == 'batch-canonical'
assert end['outcome'] == 'success'
assert end['reasoning_tokens'] == 0 and end['reasoning_tokens_complete'] is True
assert end['content_first_seconds'] is None
assert not any(r['event'] == 'first_output' for r in rows)
"""
        )
        with tempfile.TemporaryDirectory() as temp:
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith("OBSERVABILITY_")
            }
            env.update(OBSERVABILITY_ENABLED="1", OBSERVABILITY_LOG_DIR=temp)
            result = subprocess.run(
                [sys.executable, "-c", program],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_http_stack_preserves_wire_bytes_fields_and_native_level_independence(self):
        program = (
            LOAD
            + """
import logging, os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
native = logging.getLogger("vllm")
level = os.environ["TEST_NATIVE_LEVEL"]
native.disabled = level == "OFF"
if not native.disabled:
    native.setLevel(level)
app = FastAPI()
app.add_middleware(obs.RequestObservabilityMiddleware)
parts = [b'data: {"choices":[{"delta":null}]}\\n\\n',
         b'data: {"choices":[{"delta":{"reasoning":"think"}}]}\\n\\n',
         b'data: {"choices":[{"delta":{"content":"answer"}}]}\\n\\n',
         b'data: {"choices":[],"usage":{'
         b'"completion_tokens_details":{"reasoning_tokens":12}}}\\n\\n',
         b'data: [DONE]\\n\\n']
@app.post('/v1/chat/completions')
async def completion():
    wire = b''.join(parts)
    fragments = [wire[offset:offset + 17] for offset in range(0, len(wire), 17)]
    return StreamingResponse(iter(fragments), media_type='text/event-stream')
with TestClient(app) as client:
    response = client.post('/v1/chat/completions', headers={
        'X-Observability-Request-Id': 'canonical', 'X-Request-Id': 'backend',
        'X-Client-Request-Id': 'client', 'X-Observability-Phase': 'decode'})
assert response.content == b''.join(parts)
assert response.headers['X-Observability-Request-Id'] == 'canonical'
obs.runtime().close()
directory = Path(os.environ["OBSERVABILITY_LOG_DIR"])
rows = [json.loads(line) for path in directory.rglob("*.jsonl")
        for line in path.read_text().splitlines()]
finished = next(r for r in rows if r['event'] == 'request_finished')
assert finished['request_id'] == 'canonical'
assert finished['phase'] == 'decode'
assert finished['reasoning_tokens'] == 12
assert finished['reasoning_tokens_complete'] is True
assert finished['reasoning_first_seconds'] <= finished['content_first_seconds']
assert finished['outcome'] == 'success'
assert sum(r['event'] == 'first_output' for r in rows) == 1
assert all(r.get('log_dropped', 0) == 0 for r in rows)
assert all(r.get('log_io_errors', 0) == 0 for r in rows)
print(json.dumps(sorted(finished)))
"""
        )
        contracts = []
        for role in (None, "vllm", "d0"):
            for level in ("INFO", "WARN", "ERROR", "OFF"):
                with tempfile.TemporaryDirectory() as temp:
                    env = {
                        k: v
                        for k, v in os.environ.items()
                        if not k.startswith("OBSERVABILITY_")
                    }
                    env.update(
                        OBSERVABILITY_ENABLED="1",
                        OBSERVABILITY_LOG_DIR=temp,
                        TEST_NATIVE_LEVEL=level,
                    )
                    if role is not None:
                        env["OBSERVABILITY_ROLE"] = role
                    result = subprocess.run(
                        [sys.executable, "-c", program],
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=20,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    contracts.append(json.loads(result.stdout))
                    paths = list(Path(temp).rglob("*.jsonl"))
                    self.assertTrue(paths)
                    self.assertEqual(
                        len(paths[0].relative_to(temp).parts), 4 if role == "d0" else 3
                    )
        self.assertTrue(all(c == contracts[0] for c in contracts))

    def test_disabled_default_and_invalid_paths(self):
        for enabled, directory, success in (
            (None, "relative", True),
            ("0", "", True),
            ("1", "relative", False),
            ("1", "", False),
        ):
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith("OBSERVABILITY_")
            }
            env["OBSERVABILITY_LOG_DIR"] = directory
            if enabled is not None:
                env["OBSERVABILITY_ENABLED"] = enabled
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    LOAD + "\nif obs.enabled(): obs.runtime().close()",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode == 0, success, result.stderr)


if __name__ == "__main__":
    unittest.main()
