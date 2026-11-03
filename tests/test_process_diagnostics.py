# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise fault/exit diagnostics with real child processes and files."""

import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "vllm/process_diagnostics.py"
LOAD = f"""
import importlib.util, logging, os, resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
spec = importlib.util.spec_from_file_location('process_diagnostics', {str(SOURCE)!r})
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)
"""
SPEC = importlib.util.spec_from_file_location("process_diagnostics", SOURCE)
assert SPEC is not None and SPEC.loader is not None
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class ProcessDiagnosticsTests(unittest.TestCase):
    def run_child(self, code, path):
        env = dict(os.environ, VLLM_DP_COORDINATOR_FAULT_LOG=str(path))
        env.pop("PYTHONFAULTHANDLER", None)
        return subprocess.run(
            [sys.executable, "-c", LOAD + code],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_fault_file_survives_descriptor_reuse_until_process_crashes(self):
        with tempfile.TemporaryDirectory() as temp:
            fault = Path(temp) / "fault.log"
            decoy = Path(temp) / "unrelated.log"
            result = self.run_child(
                f"""
with diagnostics.coordinator_fault_handler(logging.getLogger('test')):
    with open({str(decoy)!r}, 'w') as unrelated:
        unrelated.write('keep me')
        unrelated.flush()
        os.abort()
""",
                fault,
            )
            self.assertEqual(result.returncode, -signal.SIGABRT)
            self.assertIn("Fatal Python error", fault.read_text())
            self.assertEqual(decoy.read_text(), "keep me")

    def test_normal_exit_disables_owned_handler_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as temp:
            fault = Path(temp) / "fault.log"
            result = self.run_child(
                """
with diagnostics.coordinator_fault_handler(logging.getLogger('test')):
    assert diagnostics.faulthandler.is_enabled()
assert not diagnostics.faulthandler.is_enabled()
print('completed')
""",
                fault,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "completed")
            self.assertTrue(fault.is_file())

    def test_unwritable_fault_file_falls_back_without_stopping_coordinator(self):
        result = self.run_child(
            """
with diagnostics.coordinator_fault_handler(logging.getLogger('test')):
    print('coordinator continues')
""",
            Path("/proc/vllm-cannot-create-fault.log"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "coordinator continues")
        self.assertIn("Cannot initialize coordinator fault log", result.stderr)

    def test_live_process_details_never_include_command_line_credentials(self):
        with subprocess.Popen(
            [
                sys.executable,
                "-c",
                "print('ready', flush=True); input()",
                "--api-key",
                "private-fixture-token",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                assert process.stdout is not None
                self.assertEqual(process.stdout.readline().strip(), "ready")
                details = diagnostics.process_exit_details(process.pid, None)
                self.assertEqual(details["pid"], process.pid)
                self.assertIn("Threads", details)
                self.assertNotIn("private-fixture-token", str(details))
                self.assertNotIn("--api-key", str(details))
            finally:
                process.communicate("done\n", timeout=10)

    def test_reaped_process_and_exit_signals_are_reported_without_proc_files(self):
        for code in (0, 7):
            with subprocess.Popen(
                [sys.executable, "-c", f"raise SystemExit({code})"]
            ) as process:
                process.wait(timeout=10)
                details = diagnostics.process_exit_details(
                    process.pid, process.returncode
                )
                self.assertEqual(details["exitcode"], code)
                self.assertIsNone(details["signal"])
        with subprocess.Popen(
            [sys.executable, "-c", "print('ready', flush=True); input()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            assert process.stdout is not None
            self.assertEqual(process.stdout.readline().strip(), "ready")
            process.terminate()
            process.wait(timeout=10)
            details = diagnostics.process_exit_details(process.pid, process.returncode)
            self.assertEqual(details["signal"], signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
