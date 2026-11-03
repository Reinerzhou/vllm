# Online fork validation scope

KV checksums default to enabled by deployment choice. To disable them, explicitly
set `enable_kv_checksum=false` in the outer KV transfer configuration on both P
and D, then restart both. Logging has a separate switch. NIXL protocol 18 requires
matching fork endpoints even when checksum is disabled.

## Checksum cost evidence

The earlier main-based build `51bcc95fcf5372eb5c8c80294f831f1053cdbe24` ran an
OFF → ON → OFF checksum comparison on four H200 nodes: GLM-5.2-FP8, two TP8
prefill instances and one TP1/DP16/EP16 decode instance across two nodes, MRV2,
MTP3, Torch 2.14.0+cu130, NIXL protocol 17. Logging stayed ON. AIPerf used the
public SemiAnalysis agentic traces, trajectory concurrency 32 and a 1000-second
profiling window per group. Decode prefix caching was OFF. The normalized launch
configuration SHA256 was
`dfcaadbfaaa68d4eaa6ec9255571a2dc59d8ce78b65fb468284c72e06b015472`.

| Checksum | Successful requests | Output tokens/s | TTFT P99 (s) |
| --- | --- | --- | --- |
| OFF first | 1457 | 1268.305 | 32.627 |
| ON | 1436 | 1248.153 | 29.237 |
| OFF repeat | 1469 | 1273.275 | 28.393 |

ON throughput was 1.781% below the mean of the OFF groups. OFF TTFT P99 drifted
by -12.978%, so these runs do not isolate a causal P99 checksum overhead. They
are historical measurements on protocol 17, not acceptance data for v0.33.0 or
protocol 18. The v0.33.0 logging comparisons kept checksum enabled in both groups
and cannot estimate its overhead. Current-release checksum ON/OFF P99 overhead
remains unquantified. No new performance or model evaluation is implied by the
review corrections.

## Review corrections

Correlation fields live in existing output-processor request states rather than
a separate process-global external-ID queue. This removes retention when stats
are disabled or submission/abort bypasses the metrics logger, and preserves the
correct context when requests sharing an external ID finish out of order.
A CPU regression uses real engine requests and output processors to exercise
normal completion, engine errors and abort, with statistics ON/OFF. Engine timing
summaries still require statistics; native stdout/stderr behavior is unchanged.

`OBSERVABILITY_LOG_DIR` is the supported directory setting. The unused
`VLLM_OBSERVABILITY_LOG_DIR` registration is removed. Filename and alert matching
use the standard library `re`.

The multi-choice lifecycle regression uses `temperature=1, seed=0, n=2` because
vLLM rejects `n>1` with greedy sampling (`temperature=0`). This changes the
request into a valid seeded multi-choice request; its logging ON/OFF comparison
and assertions are retained. It is this fork's lifecycle test, not an upstream
model-accuracy threshold adjustment.

MRV1 dummy forward clears the per-forward input range after graph padding and
DP coordination. Speculative depth repeats forwards and does not expand a
single forward beyond that range. No additional production MTP change is made
by the review. The existing eager GPU regression is retained; padded graph
execution has not been newly validated on GPUs.
