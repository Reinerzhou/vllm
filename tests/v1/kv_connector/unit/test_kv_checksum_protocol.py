# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the actual wire payload and TP/replica aggregation contract."""

import json
import os
import subprocess
import sys
from pathlib import Path

import msgspec
import numpy as np
import pytest
import torch

from vllm.config import CacheConfig, KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_checksum.checksum import (
    KVChecksumBlocks,
    KVChecksumGroup,
    KVChecksumRecord,
)
from vllm.distributed.kv_transfer.kv_checksum.scheduler import (
    KVChecksumPayload,
    KVChecksumScheduler,
    _chunk_sums,
    _Pending,
    _sum_replicas,
    _Unverified,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

pytestmark = pytest.mark.cpu_test


def test_payload_preserves_groups_positions_and_uint64_bits():
    payload = KVChecksumPayload(
        8,
        33,
        {
            0: (123, 16, 0, np.array([0, 2**63, 2**64 - 1], dtype=np.uint64)),
            2: (456, 16, 1, np.array([7, 9], dtype=np.uint64)),
        },
    )
    restored = KVChecksumPayload.from_bytes(payload.to_bytes())
    assert (restored.tp_size, restored.num_tokens) == (8, 33)
    assert restored.groups.keys() == payload.groups.keys()
    for key, (fingerprint, size, start, values) in payload.groups.items():
        assert restored.groups[key][:3] == (fingerprint, size, start)
        np.testing.assert_array_equal(restored.groups[key][3], values)


@pytest.mark.parametrize(
    "wire",
    [
        b"not msgpack",
        msgspec.msgpack.encode((999, 8, 32, [])),
        msgspec.msgpack.encode((1, 8, 32, [(0, 7, 16, 0, b"bad")])),
        msgspec.msgpack.encode((1, 0, 32, [])),
        msgspec.msgpack.encode((1, 8, -1, [])),
        msgspec.msgpack.encode((1, 8, 32, [(0, 7, 0, 0, bytes(8))])),
        msgspec.msgpack.encode((1, 8, 32, [(0, 7, 16, -1, bytes(8))])),
        msgspec.msgpack.encode(
            (1, 8, 32, [(0, 7, 16, 0, bytes(8)), (0, 8, 16, 0, bytes(8))])
        ),
    ],
)
def test_invalid_wire_is_unverified_instead_of_accepted(wire):
    pending = _Pending(KVChecksumBlocks(32, {0: (0, [1, 2])}))
    pending.set_expected("bad", wire)
    assert pending.expected is None
    assert pending.missing_reason == "the producer's checksums are invalid"


def test_tp8_shards_sum_modulo_uint64_for_tp1_comparison():
    groups = [
        KVChecksumGroup(0, [(1, 0)], np.array([[2**64 - 1], [rank]], dtype=np.uint64))
        for rank in range(8)
    ]
    checks = _sum_replicas(groups, 2)
    assert len(checks) == 1
    np.testing.assert_array_equal(checks[0], np.array([2**64 - 8, 28], dtype=np.uint64))


def test_every_replicated_mla_copy_is_checked_not_only_the_sum():
    groups = [
        KVChecksumGroup(0, [(8, rank)], np.array([[123]], dtype=np.uint64))
        for rank in range(8)
    ]
    groups[5].values[0, 0] ^= np.uint64(1)
    checks = _sum_replicas(groups, 1)
    assert [int(values[0]) for values in checks] == [123] * 5 + [122, 123, 123]
    with pytest.raises(_Unverified, match="a replica was not checksummed"):
        _sum_replicas(groups[:-1], 1)


def test_missing_rank_and_stale_block_report_cannot_pass():
    pending = _Pending(KVChecksumBlocks(32, {0: (0, [3, 4])}))
    record = KVChecksumRecord(0, 0, {}, frozenset(), {0: 7})
    pending.reports[(0, 0)] = (
        {0: KVChecksumGroup(1, [(1, 0)], np.array([[1], [2]], dtype=np.uint64))},
        record,
    )
    with pytest.raises(_Unverified, match="not every worker"):
        pending.check_complete(8)
    with pytest.raises(_Unverified, match="other blocks than scheduled"):
        pending.group_checksums(0)


def test_different_block_sizes_compare_the_same_valid_token_chunks():
    first, sums = _chunk_sums(np.array([3, 5, 7], dtype=np.uint64), 0, 16, 32, 33)
    assert first == 0
    np.testing.assert_array_equal(sums, np.array([8, 7], dtype=np.uint64))


def _checker_and_pending(sent_values, sent_tokens):
    config = VllmConfig(
        cache_config=CacheConfig(block_size=16),
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector", kv_role="kv_both"
        ),
    )
    cache = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["model.layers.0"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32
                ),
            )
        ],
    )
    checker = KVChecksumScheduler(
        config, cache, ExampleConnector(config, KVConnectorRole.SCHEDULER, cache)
    )
    record = KVChecksumRecord(0, 0, {}, frozenset(), {0: 7})
    pending = _Pending(
        KVChecksumBlocks(31, {0: (0, [2, 3])}),
        expected=KVChecksumPayload(
            1, sent_tokens, {0: (7, 16, 0, np.array(sent_values, dtype=np.uint64))}
        ),
        reports={
            (0, 0): (
                {
                    0: KVChecksumGroup(
                        0, [(1, 0)], np.array([[11], [22]], dtype=np.uint64)
                    )
                },
                record,
            )
        },
    )
    return checker, pending


@pytest.mark.parametrize("sent_values,sent_tokens", [([11], 32), ([11, 22], 33)])
def test_incomplete_producer_coverage_does_not_verify_only_common_prefix(
    sent_values, sent_tokens
):
    checker, pending = _checker_and_pending(sent_values, sent_tokens)
    mismatches, unverified = checker._compare(pending)
    assert not mismatches
    assert unverified, (
        "Every loaded valid chunk must be covered, not just a common prefix"
    )


@pytest.mark.parametrize("values,verified", [([11, 22], True), ([11, 23], False)])
def test_complete_valid_range_accepts_equal_values_and_rejects_corruption(
    values, verified
):
    checker, pending = _checker_and_pending(values, 31)
    assert checker._verify("request", pending) is verified


def test_checksum_results_reach_jsonl_with_request_context(tmp_path):
    # A fresh process uses the real process-global writer and its startup config.
    script = f"""
import runpy
from vllm.observability import context_fields, runtime
helper = runpy.run_path({str(Path(__file__).resolve())!r})['_checker_and_pending']
for name, values, tokens, expected in [
    ('valid', [11, 22], 31, True),
    ('corrupt', [11, 23], 31, False),
    ('missing-tail', [11], 32, False),
]:
    checker, pending = helper(values, tokens)
    pending.log_context = context_fields(
        {{'x-native-request-id': name}}, 'engine-' + name)
    assert checker._verify('engine-' + name, pending) is expected
runtime().close()
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={
            **os.environ,
            "OBSERVABILITY_ENABLED": "1",
            "OBSERVABILITY_LOG_DIR": str(tmp_path),
        },
        timeout=180,
    )
    rows = [
        json.loads(line)
        for path in tmp_path.rglob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    results = {
        row["request_id"]: row
        for row in rows
        if row["event"].startswith("kv_checksum_")
    }
    assert results["valid"]["event"] == "kv_checksum_verified"
    assert results["corrupt"]["mismatched_block_count"] == 1
    assert results["corrupt"]["level"] == "ERROR"
    assert results["missing-tail"]["unverified_reasons"]
    assert all(
        row["engine_request_id"] == "engine-" + name for name, row in results.items()
    )
    assert not any("kv_checksums" in row or "values" in row for row in results.values())


@pytest.mark.parametrize(
    "blocks,tokens,valid",
    [
        (([2, 3],), 31, True),
        (([2, 4],), 31, False),
        (([2],), 31, False),
        (([2, 3],), 32, False),
        ((), 31, False),
    ],
)
def test_producer_publication_must_match_checksummed_allocation(blocks, tokens, valid):
    checker, pending = _checker_and_pending([11, 22], 31)
    if valid:
        checker._validate_publication(pending, blocks, tokens)
    else:
        with pytest.raises(_Unverified, match="published"):
            checker._validate_publication(pending, blocks, tokens)
