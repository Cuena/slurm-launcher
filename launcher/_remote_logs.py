"""Standalone, standard-library-only reader shipped to the login node over SSH."""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import stat
import sys


def _decode_window(data, start, budget, align_tail=False):
    # Only an arbitrary initial tail may start inside a valid UTF-8 sequence.
    skip = 0
    if align_tail:
        while skip < min(3, len(data)) and data[skip] & 0xC0 == 0x80:
            skip += 1
    data = data[skip:]
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    text = decoder.decode(data[:budget], final=False)
    consumed = min(len(data), budget) - len(decoder.getstate()[0])
    while len(text.encode("utf-8")) > budget:
        consumed -= 1
        decoder.reset()
        text = decoder.decode(data[:consumed], final=False)
        consumed -= len(decoder.getstate()[0])
    return text, start + skip, start + skip + consumed


def _save_boundary(handle, position, offset):
    start = max(0, offset - 64)
    handle.seek(start)
    data = handle.read(offset - start)
    position.update(
        boundary_start=start, boundary_length=len(data),
        boundary=hashlib.sha256(data).hexdigest(),
    )

def _complete_utf8_boundary(handle, finish, size):
    # A bounded search window must not strand a valid character at its edge.
    if finish >= size:
        return finish
    handle.seek(max(0, finish - 3))
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    decoder.decode(handle.read(min(3, finish)), final=False)
    if not decoder.getstate()[0]:
        return finish
    for end in range(finish + 1, min(size, finish + 3) + 1):
        decoder.decode(handle.read(1), final=False)
        if not decoder.getstate()[0]:
            return end
    return finish



def read_file(spec, options, budget, scan_budget):
    previous = spec.get("position") or {}
    result = {
        key: spec.get(key) for key in (
            "job_id", "job_name", "stream", "path", "source", "resolution_errors"
        )
    }
    result.update(
        status="unresolved",
        exists=None,
        size=None,
        identity=None,
        start_offset=0,
        end_offset=0,
        content="",
        bytes_returned=0,
        has_more=False,
        reset=False,
        search_complete=None,
        truncated=False,
        incomplete_utf8=False,
        modified_at_ns=None,
        context_limited=False,
        error=None,
    )
    position = dict(previous)
    path = spec.get("path")
    if not path:
        result["error"] = "No log path was resolved."
        return result, position, 0
    scanned = 0
    try:
        root = spec.get("root")
        if root and os.path.commonpath(
            (os.path.realpath(root), os.path.realpath(path))
        ) != os.path.realpath(root):
            raise PermissionError("Application log escapes the tracked workdir")
        if options.get("path_only"):
            info = os.stat(path)
            result.update(
                exists=True, size=info.st_size,
                identity=f"{info.st_dev}:{info.st_ino}",
                modified_at_ns=info.st_mtime_ns,
                status="empty" if not info.st_size else "ok",
            )
            if not stat.S_ISREG(info.st_mode):
                raise OSError("Log path is not a regular file")
            return result, position, 0
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            result.update(
                exists=True,
                size=info.st_size,
                identity=f"{info.st_dev}:{info.st_ino}",
                modified_at_ns=info.st_mtime_ns,
            )
            if not stat.S_ISREG(info.st_mode):
                raise OSError("Log path is not a regular file")
            old_offset = previous.get("offset", 0)
            anchor_length = previous.get("anchor_length", min(info.st_size, 64))
            anchor = hashlib.sha256(handle.read(anchor_length)).hexdigest()
            boundary_changed = False
            if previous.get("boundary") is not None:
                handle.seek(previous["boundary_start"])
                boundary_changed = hashlib.sha256(
                    handle.read(previous["boundary_length"])
                ).hexdigest() != previous["boundary"]
            reset = bool(previous) and (
                previous.get("identity") != result["identity"]
                or old_offset > info.st_size
                or (previous.get("anchor") is not None and previous["anchor"] != anchor)
                or boundary_changed
            )
            if reset:
                previous = {}
                anchor_length = min(info.st_size, 64)
                handle.seek(0)
                anchor = hashlib.sha256(handle.read(anchor_length)).hexdigest()
            result["reset"] = reset
            position = dict(previous)
            position.update(
                identity=result["identity"], anchor=anchor, anchor_length=anchor_length
            )
            search = options.get("search")
            offset = previous.get("offset", 0)
            emitted = previous.get("emitted", 0)
            result.update(
                status="empty" if not info.st_size else "ok",
                start_offset=offset,
                end_offset=offset,
            )
            if search is not None and offset >= info.st_size:
                result.update(search_complete=True)
                return result, position, 0
            if budget < 4 or (search is not None and scan_budget < 4):
                result.update(
                    has_more=offset < info.st_size,
                    search_complete=False if search is not None else None,
                )
                return result, previous, scanned
            pending_end = previous.get("pending_end", 0)
            if search is not None and pending_end > offset:
                start = offset
                finish = _complete_utf8_boundary(handle, pending_end, info.st_size)
                handle.seek(start)
                data = handle.read(min(budget + 4, finish - start))
            elif search is not None:
                needle = search.encode("utf-8")
                if scan_budget < len(needle) * 2:
                    result.update(has_more=offset < info.st_size, search_complete=False)
                    return result, previous, scanned
                # Bounded lookbehind supplies context without retaining content in cursors.
                start = max(
                    emitted, offset - min(options["max_bytes"], scan_budget // 2)
                )
                handle.seek(start)
                window = handle.read(min(scan_budget, info.st_size - start))
                scanned = len(window)
                found = window.find(needle, offset - start)
                if found < 0:
                    end = start + len(window)
                    # Rescan only the literal overlap, not previously returned content.
                    next_offset = (
                        end
                        if end >= info.st_size
                        else max(offset, end - len(needle) + 1)
                    )
                    position.update(offset=next_offset, emitted=emitted)
                    _save_boundary(handle, position, next_offset)
                    result.update(
                        start_offset=offset,
                        end_offset=next_offset,
                        has_more=end < info.st_size,
                        search_complete=end >= info.st_size,
                    )
                    return result, position, scanned
                line_start = window.rfind(b"\n", 0, found) + 1
                for _ in range(options["context"]):
                    if line_start == 0:
                        break
                    line_start = window.rfind(b"\n", 0, line_start - 1) + 1
                line_end = window.find(b"\n", found + len(needle))
                remaining = options["context"]
                while line_end >= 0 and remaining:
                    next_end = window.find(b"\n", line_end + 1)
                    if next_end < 0:
                        line_end = -1
                        break
                    line_end = next_end
                    remaining -= 1
                finish = start + (len(window) if line_end < 0 else line_end + 1)
                finish = _complete_utf8_boundary(handle, finish, info.st_size)
                result["context_limited"] = (line_start == 0 and start > emitted) or (
                    line_end < 0 and start + len(window) < info.st_size
                )
                start += line_start
                handle.seek(start)
                data = handle.read(min(budget + 4, finish - start))
            else:
                if previous:
                    start = offset
                elif reset:
                    start = 0
                else:
                    start = max(0, info.st_size - budget)
                handle.seek(start)
                data = handle.read(budget + 4)
                if not previous and not reset:
                    # Initial navigation is a tail; continuation always moves forward.
                    lines = data.splitlines(keepends=True)
                    if len(lines) > options["lines"]:
                        skipped = sum(map(len, lines[: -options["lines"]]))
                        start += skipped
                        data = data[skipped:]
                finish = info.st_size
            text, start, end = _decode_window(
                data, start, budget,
                align_tail=search is None and not previous and not reset and start > 0,
            )
            incomplete_utf8 = (
                end < info.st_size and start + len(data) >= info.st_size and not text
            )
            result.update(
                start_offset=start,
                end_offset=end,
                content=text,
                bytes_returned=len(text.encode("utf-8")),
                has_more=end < info.st_size and not incomplete_utf8,
                incomplete_utf8=incomplete_utf8,
                truncated=not previous and not reset and search is None and start > 0,
            )
            position.update(offset=end, emitted=end)
            _save_boundary(handle, position, end)
            if search is not None:
                position["pending_end"] = finish if end < finish else 0
                result["search_complete"] = end >= info.st_size
            return result, position, scanned
    except FileNotFoundError as exc:
        result.update(status="missing", exists=False, error=str(exc))
    except OSError as exc:
        result.update(status="unreadable", error=str(exc))
        if result["exists"] is None:
            try:
                os.stat(path)
                result["exists"] = True
            except FileNotFoundError:
                result["exists"] = False
            except OSError:
                pass
    return result, position, scanned


def read_batch(request):
    options = request["options"]
    budget = options["max_bytes"]
    scan_budget = options["scan_bytes"]
    files, positions = [], []
    for spec in request["files"]:
        result, position, scanned = read_file(spec, options, budget, scan_budget)
        files.append(result)
        positions.append(position)
        budget -= result["bytes_returned"]
        scan_budget -= scanned
    return {"files": files, "positions": positions}


if __name__ == "__main__":
    print(json.dumps(read_batch(json.load(sys.stdin)), ensure_ascii=True))
