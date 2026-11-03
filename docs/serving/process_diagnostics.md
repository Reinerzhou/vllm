# Process failure diagnostics

✅ The data-parallel coordinator records Python exceptions and its PID on exit.
Fatal interpreter faults are written to `VLLM_DP_COORDINATOR_FAULT_LOG`, defaulting to
`/tmp/vllm_dp_coordinator_fault_<PID>.log`. The handler keeps the file open while the coordinator
runs. If the file cannot be opened, startup continues with a warning in the native logger.

✅ These are native process diagnostics. File capacity and cleanup are deployment-owned; they are
independent of request lifecycle JSONL and its retention settings. No model request or device
synchronization is added for diagnostics.

✅ API-server and coordinator exits include PID, parent PID, process group, session, exit code and
termination signal in the native logger. When still available, `/proc` state, thread count, RSS and
wait channel are included. Command lines and environments are never read for these diagnostics.
Missing or inaccessible process information is omitted; no extra shutdown wait is added.
