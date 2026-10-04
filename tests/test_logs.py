from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from launcher._remote_logs import read_batch
from launcher.logs import _decode_cursor, _encode_cursor, add_logs_args, run_logs


class LogNavigationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "job.out"

    def request(self, **options):
        return {
            "files": [
                {
                    "job_id": "123",
                    "job_name": "train",
                    "stream": "stdout",
                    "path": str(self.path),
                    "source": "tracking",
                }
            ],
            "options": {
                "lines": 100,
                "max_bytes": 65536,
                "scan_bytes": 65536,
                "search": None,
                "context": 20,
                "path_only": False,
                **options,
            },
        }

    def resume(self, request, response):
        for spec, position in zip(request["files"], response["positions"]):
            spec["position"] = position
        return read_batch(request)

    def test_tail_then_append_does_not_repeat_bytes(self):
        self.path.write_text("old\nlast\n", encoding="utf-8")
        request = self.request(lines=1)
        first = read_batch(request)
        self.assertEqual(first["files"][0]["content"], "last\n")
        with self.path.open("a") as handle:
            handle.write("new\n")
        second = self.resume(request, first)
        self.assertEqual(second["files"][0]["content"], "new\n")
        self.assertEqual(
            second["files"][0]["start_offset"], first["files"][0]["end_offset"]
        )
        self.assertFalse(second["files"][0]["reset"])

    def test_split_utf8_append_waits_for_complete_character(self):
        self.path.write_bytes(b"hello\n")
        request = self.request()
        first = read_batch(request)
        with self.path.open("ab") as handle:
            handle.write(b"\xf0\x9f")
        partial = self.resume(request, first)
        self.assertEqual(partial["files"][0]["content"], "")
        self.assertEqual(partial["files"][0]["end_offset"], 6)
        self.assertTrue(partial["files"][0]["incomplete_utf8"])
        self.assertFalse(partial["files"][0]["has_more"])
        with self.path.open("ab") as handle:
            handle.write(b"\x8c\x8d\n")
        complete = self.resume(request, partial)
        self.assertEqual(complete["files"][0]["content"], "\U0001f30d\n")
        self.assertEqual(complete["files"][0]["bytes_returned"], 5)

    def test_replacement_and_same_inode_rewrite_restart(self):
        self.path.write_text("original\n")
        request = self.request()
        first = read_batch(request)
        replacement = self.root / "replacement"
        replacement.write_text("replacement content\n")
        replacement.replace(self.path)
        second = self.resume(request, first)
        self.assertTrue(second["files"][0]["reset"])
        self.assertEqual(second["files"][0]["content"], "replacement content\n")
        self.path.write_text("short\n")
        third = self.resume(request, second)
        self.assertTrue(third["files"][0]["reset"])
        self.assertEqual(third["files"][0]["content"], "short\n")

    def test_same_inode_regrown_rewrite_preserving_header_resets(self):
        self.path.write_bytes(b"h" * 64 + b"old\n" * 100)
        request = self.request()
        first = read_batch(request)
        self.path.write_bytes(b"h" * 64 + b"new\n" * 200)
        second = self.resume(request, first)
        self.assertTrue(second["files"][0]["reset"])
        self.assertEqual(second["files"][0]["start_offset"], 0)
        self.assertIn("new\n", second["files"][0]["content"])

    def test_malformed_bytes_at_zero_and_saved_offset_are_replaced(self):
        self.path.write_bytes(b"\x80start\n")
        request = self.request()
        first = read_batch(request)
        self.assertEqual(first["files"][0]["content"], "\ufffdstart\n")
        with self.path.open("ab") as handle:
            handle.write(b"\x80\xffend\n")
        second = self.resume(request, first)
        self.assertEqual(second["files"][0]["content"], "\ufffd\ufffdend\n")
        self.assertEqual(second["files"][0]["start_offset"], len(b"\x80start\n"))

    def test_search_progress_across_many_completed_files(self):
        request = self.request(search="MATCH", context=0, max_bytes=4096)
        request["files"] = []
        for index in range(32):
            path = self.root / str(index)
            path.write_bytes(b"x" * 4091 + (b"MATCH" if index == 31 else b"xxxxx"))
            request["files"].append(
                {"path": str(path), "stream": "stdout", "source": "tracking"}
            )
        response = read_batch(request)
        returned = []
        for _ in range(32):
            returned.append(response["files"][-1]["content"])
            if all(file["search_complete"] for file in response["files"]):
                break
            old_positions = response["positions"]
            response = self.resume(request, response)
            self.assertNotEqual(response["positions"], old_positions)
        self.assertTrue(all(file["search_complete"] for file in response["files"]))
        self.assertEqual("".join(returned).count("MATCH"), 1)

    def test_search_crosses_scan_boundary_with_literal_metacharacters(self):
        self.path.write_bytes(b"x" * 65534 + b"[hit].*\nnext\n")
        request = self.request(search="[hit].*", context=0)
        first = read_batch(request)
        self.assertEqual(first["files"][0]["content"], "")
        self.assertFalse(first["files"][0]["search_complete"])
        second = self.resume(request, first)
        self.assertIn("[hit].*", second["files"][0]["content"])
        self.assertLessEqual(second["files"][0]["bytes_returned"], 65536)

    def test_search_context_boundary_does_not_strand_utf8(self):
        self.path.write_bytes(b"MATCH" + b"x" * 65530 + "\U0001f30d\n".encode())
        request = self.request(search="MATCH", context=0)
        first = read_batch(request)
        self.assertEqual(first["files"][0]["end_offset"], 65535)
        second = self.resume(request, first)
        self.assertEqual(second["files"][0]["content"], "\U0001f30d")
        third = self.resume(request, second)
        self.assertTrue(third["files"][0]["search_complete"])
        self.assertEqual(third["files"][0]["content"], "")

    def test_search_context_can_be_drained_with_small_content_budget(self):
        self.path.write_text("before\nhit\nafter\nlast\n")
        request = self.request(search="hit", context=1, max_bytes=4)
        response = read_batch(request)
        chunks = [response["files"][0]["content"]]
        for _ in range(10):
            if not response["files"][0]["has_more"]:
                break
            response = self.resume(request, response)
            chunks.append(response["files"][0]["content"])
            self.assertLessEqual(response["files"][0]["bytes_returned"], 4)
        self.assertEqual("".join(chunks), "before\nhit\nafter\n")
        self.assertTrue(response["files"][0]["search_complete"])

    def test_aggregate_byte_bound_across_files(self):
        self.path.write_text("a" * 100)
        other = self.root / "job.err"
        other.write_text("old content\nlast\n")
        request = self.request(max_bytes=8)
        request["files"].append(
            {**request["files"][0], "path": str(other), "stream": "stderr"}
        )
        response = read_batch(request)
        self.assertEqual(sum(file["bytes_returned"] for file in response["files"]), 8)
        self.assertTrue(response["files"][1]["has_more"])
        next_response = self.resume(request, response)
        self.assertEqual(next_response["files"][1]["content"], "nt\nlast\n")

    def test_missing_empty_and_unreadable_are_distinct(self):
        request = self.request()
        missing = read_batch(request)["files"][0]
        self.assertEqual((missing["status"], missing["exists"]), ("missing", False))
        self.path.touch()
        empty = read_batch(request)["files"][0]
        self.assertEqual((empty["status"], empty["exists"]), ("empty", True))
        self.path.unlink()
        self.path.mkdir()
        unreadable = read_batch(request)["files"][0]
        self.assertEqual(
            (unreadable["status"], unreadable["exists"]), ("unreadable", True)
        )
        self.assertEqual(unreadable["source"], "tracking")

    def test_application_path_symlink_escape_is_not_read(self):
        inside = self.root / "work"
        inside.mkdir()
        self.path.write_text("outside secret")
        link = inside / "app.log"
        link.symlink_to(self.path)
        request = self.request()
        request["files"][0].update(path=str(link), root=str(inside))
        result = read_batch(request)["files"][0]
        self.assertEqual(result["status"], "unreadable")
        self.assertEqual(result["content"], "")

    def test_path_only_stats_without_content(self):
        self.path.write_text("content\n")
        self.path.chmod(0)
        self.addCleanup(self.path.chmod, 0o600)
        with patch("launcher._remote_logs.os.open", side_effect=PermissionError):
            result = read_batch(self.request(path_only=True))["files"][0]
        self.assertEqual(result["size"], 8)
        self.assertEqual(result["content"], "")
        self.assertEqual(result["status"], "ok")


class LogsCommandTests(unittest.TestCase):
    def args(self, *arguments):
        parser = argparse.ArgumentParser()
        add_logs_args(parser)
        return parser.parse_args(arguments)

    def test_tracked_missing_stream_recovers_and_cursor_keeps_saved_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "saved.out"
            err = root / "resolved.err"
            out.write_text("saved stdout\n")
            err.write_text("resolved stderr\n")
            scripts = {
                "ssh": (
                    '#!/bin/sh\nfor last; do :; done\n'
                    'if [ "$last" = "-s" ]; then exec /bin/bash; fi\n'
                    'exec /bin/sh -c "$last"\n'
                ),
                "scontrol": "#!/bin/sh\nprintf '%s\\n' " + shlex.quote(
                    f"JobId=123 JobName=train JobState=RUNNING StdOut={root / 'wrong.out'}"
                ) + "\n",
                "sacct": "#!/bin/sh\nprintf '%s\\n' " + shlex.quote(
                    f"123|train|RUNNING||{err}"
                ) + "\n",
            }
            for name, source in scripts.items():
                script = root / name
                script.write_text(source)
                script.chmod(0o700)
            tracking = root / "jobs.json"
            tracking.write_text(json.dumps({
                "cluster_login": "user@loopback",
                "ssh_config_file": str(root / "ssh-config"),
                "ssh_options": ["-o", "BatchMode=yes"],
                "jobs": [{"job_id": "123", "job_name": "train", "stdout": str(out)}],
            }))
            with patch.dict(os.environ, {"PATH": str(root) + os.pathsep + os.environ["PATH"]}):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(run_logs(self.args("--run", str(tracking), "--json")), 0)
                first = json.loads(output.getvalue())
                self.assertEqual(first["files"][0]["content"], "saved stdout\n")
                self.assertEqual(first["files"][0]["source"], "tracking")
                self.assertEqual(first["files"][1]["content"], "resolved stderr\n")
                self.assertEqual(first["files"][1]["source"], "sacct")
                tracking.unlink()
                (root / "scontrol").unlink()
                (root / "sacct").unlink()
                with err.open("a") as handle:
                    handle.write("later\n")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(run_logs(self.args("--cursor", first["cursor"], "--json")), 0)
                second = json.loads(output.getvalue())
                self.assertEqual(second["files"][1]["content"], "later\n")
                self.assertEqual(second["files"][1]["source"], "sacct")

    @staticmethod
    def remote(command, *, input, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, json.dumps(read_batch(json.loads(input))), ""
        )

    def test_cursor_resumes_without_tracking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "stdout"
            log.write_text("first\n")
            tracking = root / "jobs.json"
            tracking.write_text(
                json.dumps(
                    {
                        "cluster_login": "user@cluster",
                        "ssh_config_file": "/custom/ssh",
                        "ssh_options": ["-o", "BatchMode=yes"],
                        "jobs": [
                            {"job_id": "123", "job_name": "train", "stdout": str(log)}
                        ],
                    }
                )
            )
            output = io.StringIO()
            with (
                patch("launcher.logs.subprocess.run", side_effect=self.remote),
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(
                    run_logs(
                        self.args(
                            "--run", str(tracking), "--stream", "stdout", "--json"
                        )
                    ),
                    0,
                )
            first = json.loads(output.getvalue())
            self.assertEqual(first["files"][0]["content"], "first\n")
            tracking.unlink()
            with log.open("a") as handle:
                handle.write("second\n")
            output = io.StringIO()
            with (
                patch("launcher.logs.subprocess.run", side_effect=self.remote),
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(
                    run_logs(self.args("--cursor", first["cursor"], "--json")), 0
                )
            second = json.loads(output.getvalue())
            self.assertEqual(second["files"][0]["content"], "second\n")
            state = _decode_cursor(second["cursor"])
            state["files"][0]["position"]["offset"] = -1
            with self.assertRaises(ValueError):
                _decode_cursor(_encode_cursor(state))

    def test_json_follow_rejected_before_remote_access(self):
        output = io.StringIO()
        with (
            patch("launcher.logs.subprocess.run") as remote,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(run_logs(self.args("--follow", "--json")), 1)
        self.assertFalse(json.loads(output.getvalue())["ok"])
        remote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
