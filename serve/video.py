"""Bounded finite-video preparation for Responses; no model or image-path changes.

The worker owns downloads and codec processes. The caller owns its directory and
process group, so cancellation also removes an in-flight probe/decode.
"""
from __future__ import annotations

import base64
import bisect
import contextlib
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import uuid


class VideoError(ValueError):
    pass


class VideoContextError(VideoError):
    pass


class VideoStageError(VideoError):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


def validate_artifact(path, layout, settings):
    """Check every SVE1 record and timestamp before reuse; never trust a cached token count."""
    if not layout or len(layout) > (settings["max_frames"] + 1) // 2:
        raise VideoError("invalid cached video layout")
    total, previous = 0, -1.0
    with Path(path).open("rb") as source:
        for seconds, cells in layout:
            if not math.isfinite(seconds) or not previous < seconds <= settings["max_duration_s"] \
                    or not isinstance(cells, int) or not 0 < cells <= 1024:
                raise VideoError("invalid cached video timestamp or cells")
            header = source.read(20)
            if len(header) != 20:
                raise VideoError("incomplete cached video header")
            magic, rows, nx, ny, width = struct.unpack("<5I", header)
            if magic != 0x31455653 or rows != cells or nx * ny != rows or width != 2560:
                raise VideoError("invalid cached video embeddings")
            source.seek(rows * width * 4, 1)
            if source.tell() > Path(path).stat().st_size:
                raise VideoError("incomplete cached video embeddings")
            total += rows
            previous = seconds
        if source.read(1) or total > settings["max_tokens"]:
            raise VideoError("cached video exceeds current limits")


DEFAULTS = {"fps": 2.0, "max_duration_s": 120.0, "max_frames": 256,
            "max_source_bytes": 64 << 20, "max_source_frames": 32768,
            "max_source_pixels": 16 << 20, "pair_tokens": 256,
            "max_tokens": 32768, "deadline_s": 600.0,
            "ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}
CEILINGS = {"fps": 24, "max_duration_s": 600, "max_frames": 1024,
            "max_source_bytes": 256 << 20, "max_source_frames": 65536,
            "max_source_pixels": 16 << 20, "pair_tokens": 1024,
            "max_tokens": 65536, "deadline_s": 1800}


def policy(config):
    if config is None or config is False:
        return None
    if not isinstance(config, dict) or not isinstance(config.get("enabled", False), bool):
        raise VideoError("vision.video must be an object with enabled: true or false")
    if not config.get("enabled", False):
        return None
    if set(config) - set(DEFAULTS) - {"enabled", "diagnostics"}:
        raise VideoError("unknown vision.video setting")
    if not isinstance(config.get("diagnostics", False), bool):
        raise VideoError("vision.video.diagnostics must be true or false")
    result = {**DEFAULTS, **{k: v for k, v in config.items() if k != "enabled"}}
    for key, ceiling in CEILINGS.items():
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or not 0 < value <= ceiling or (key not in ("fps", "deadline_s", "max_duration_s")
                                               and not isinstance(value, int)):
            raise VideoError(f"vision.video.{key} must be positive and at most {ceiling}")
    for key in ("ffmpeg", "ffprobe"):
        if not isinstance(result[key], str) or not result[key] or any(c in result[key] for c in "\x00\r\n"):
            raise VideoError(f"vision.video.{key} must name an executable")
    return result


class Budget:
    def __init__(self, settings, cancel):
        self.settings, self.cancel = dict(settings), cancel
        self.deadline = time.monotonic() + settings["deadline_s"]
        self.trace = Trace(settings.get("diagnostics", False))
        self.current_stage = "preparation"

    def check(self):
        if self.cancel.is_set():
            raise VideoStageError(self.current_stage, "video preparation cancelled")
        if time.monotonic() >= self.deadline:
            raise VideoStageError(self.current_stage, "video preparation deadline exceeded")

    @contextlib.contextmanager
    def stage(self, name):
        previous, self.current_stage = self.current_stage, name
        try:
            with self.trace.stage(name, self.cancel, self.deadline):
                try:
                    yield
                except VideoError as error:
                    if isinstance(error, (VideoStageError, VideoContextError)):
                        raise
                    raise VideoStageError(name, str(error)) from None
        finally:
            self.current_stage = previous


class Trace:
    """Opt-in, bounded stage events: generated IDs only, never request payloads."""
    def __init__(self, enabled):
        self.enabled = enabled
        self.id = uuid.uuid4().hex

    def event(self, stage, **fields):
        if self.enabled:
            details = " ".join(f"{key}={value}" for key, value in fields.items())
            print(f"[strata] video trace={self.id} stage={stage} {details}", flush=True)

    @contextlib.contextmanager
    def stage(self, name, cancel=None, deadline=None, timeout_errors=()):
        started = time.monotonic()
        self.event(name, event="start")
        outcome = "ok"
        try:
            yield
        except BaseException as error:
            code = getattr(error, "code", "") or ""
            cancelled = (cancel is not None and cancel.is_set()) or code.endswith("_cancelled")
            timed_out = isinstance(error, timeout_errors) or code.endswith("_timeout") \
                or (deadline is not None and time.monotonic() >= deadline)
            outcome = "cancelled" if cancelled else "timeout" if timed_out else "failed"
            raise
        finally:
            if outcome == "ok" and cancel is not None and cancel.is_set():
                outcome = "cancelled"
            self.event(name, event="end", outcome=outcome, elapsed_s=f"{time.monotonic()-started:.3f}")


@contextlib.contextmanager
def acquire(lock, budget):
    while True:
        budget.check()
        if lock.acquire(timeout=0.1):
            break
    try:
        budget.check()
        yield
    finally:
        lock.release()


def stop(proc):
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif proc.poll() is None:
        proc.kill()  # The worker owns its own kill-on-close Job, including codec descendants.
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        raise VideoError("video worker did not stop") from None


def prepare(source, directory, budget, *, local_file=None):
    """One video per request: bounded source, packet and artifact disk ownership."""
    settings = budget.settings
    if local_file is None and not source.startswith(("http://", "https://", "data:")):
        raise VideoError("video_url must be an http(s) or base64 data: URL; local paths are not accepted")
    if len(source) > (settings["max_source_bytes"] * 4 // 3 + 4096):
        raise VideoError("video source exceeds its byte budget")
    for key in ("ffmpeg", "ffprobe"):
        if not shutil.which(settings[key]):
            raise VideoError(f"video requires {key}; configure vision.video.{key}")
    directory = Path(directory)
    request = directory / "source.json"
    spec = {"source": source, "settings": settings}
    if local_file is not None:
        spec["local_file"] = str(local_file)
    with request.open("w", encoding="utf-8") as out:
        for part in json.JSONEncoder().iterencode(spec):
            for at in range(0, len(part), 65536):
                budget.check()
                out.write(part[at:at + 65536])
    budget.check()
    log = directory / "worker.log"
    # The worker never prints URLs or credentials. It creates at most one codec at a time.
    with log.open("wb") as stderr:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(request)],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr,
                                start_new_session=os.name == "posix")
        observed = []

        def progress():
            # Each marker is written once. Replacing a marker while this reader
            # has it open can fail with a sharing violation on Windows.
            expected = (("source", "download"), ("decode",), ("sampling",), ("ready",))
            while len(observed) < len(expected):
                try:
                    with (directory / f"stage-{len(observed)}.json").open() as marker:
                        event = json.loads(marker.read(1024))
                except (OSError, ValueError):
                    return  # A newly created marker may not have been fully written yet.
                if not isinstance(event, list) or len(event) != 2:
                    return
                name, started = event
                if name not in expected[len(observed)] or isinstance(started, bool) \
                        or not isinstance(started, (int, float)) or not math.isfinite(started) \
                        or (observed and started < observed[-1][1]):
                    return
                if observed:
                    previous, before = observed[-1]
                    budget.trace.event(previous, event="end", outcome="ok", elapsed_s=f"{started-before:.3f}")
                observed.append((name, started))
                if name != "ready":
                    budget.current_stage = name
                    budget.trace.event(name, event="start")

        try:
            while proc.poll() is None:
                progress()
                budget.check()
                if log.stat().st_size > 65536:
                    raise VideoError("video worker diagnostic limit exceeded")
                time.sleep(0.05)
            budget.check()
            progress()
            if proc.returncode:
                message = (directory / "error.json")
                detail = json.loads(message.read_text()) if message.is_file() else "video worker failed"
                raise VideoError(detail)
        except VideoError as e:
            progress()
            stage = observed[-1][0] if observed and observed[-1][0] != "ready" else "preparation"
            if observed and observed[-1][0] != "ready":
                budget.trace.event(stage, event="end", outcome="cancelled" if budget.cancel.is_set() else
                                   "timeout" if time.monotonic() >= budget.deadline else "failed",
                                   elapsed_s=f"{time.monotonic()-observed[-1][1]:.3f}")
            raise VideoStageError(stage, str(e)) from None
        finally:
            stop(proc)
            request.unlink(missing_ok=True)
            budget.current_stage = "preparation"
    return directory / "frames.svf"


def _command(args, output, limit):
    """Bound both pipes while draining; never let codec diagnostics fill disk/RAM."""
    proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    errors = []

    def drain(stream, sink, maximum):
        total = 0
        try:
            while chunk := stream.read1(65536):
                total += len(chunk)
                if total > maximum:
                    errors.append("video codec output exceeds its budget")
                    proc.kill()
                    break
                if sink is not None:
                    sink.write(chunk)
                else:                                   # codecs run with -v error; an error is not a clean EOF
                    errors.append("video codec reported a decoding error")
                    proc.kill()
                    break
        except (OSError, ValueError):
            errors.append("video codec output could not be read")
            proc.kill()
        finally:
            stream.close()

    with Path(output).open("wb") as sink:
        readers = [threading.Thread(target=drain, args=(proc.stdout, sink, limit)),
                   threading.Thread(target=drain, args=(proc.stderr, None, 65536))]
        for reader in readers:
            reader.start()
        code = proc.wait()
        for reader in readers:
            reader.join()
    if code or errors:
        raise VideoError(errors[0] if errors else "video codec rejected the input")


def _worker(request):
    directory = request.parent
    spec = json.loads(request.read_text(encoding="utf-8"))
    stages = []

    def record_stage(name):
        marker = directory / f"stage-{len(stages)}.json"
        event = (name, time.monotonic())
        marker.write_text(json.dumps(event))
        stages.append(event)

    p, source = spec["settings"], spec["source"]
    record_stage("source" if spec.get("local_file") is not None or source.startswith("data:") else "download")
    source_file = directory / "source.bin"
    if spec.get("local_file") is not None:
        # Only the authenticated store supplies this field; it is never read from an API part.
        local = Path(spec["local_file"])
        if not 0 < local.stat().st_size <= p["max_source_bytes"]:
            raise VideoError("video source exceeds its byte budget")
        shutil.copyfile(local, source_file)
    elif source.startswith("data:"):
        head, comma, data = source.partition(",")
        if not comma or not head.startswith("data:video/") or not head.endswith(";base64"):
            raise VideoError("video data URL requires a video MIME type and base64 encoding")
        try:
            raw = base64.b64decode(data, validate=True)
        except ValueError:
            raise VideoError("invalid video base64") from None
        if len(raw) > p["max_source_bytes"]:
            raise VideoError("video source exceeds its byte budget")
        source_file.write_bytes(raw)
        del raw, data
    else:
        try:
            with urllib.request.urlopen(urllib.request.Request(source, headers={"User-Agent": "Strata"}),
                                        timeout=10) as response, source_file.open("wb") as out:
                if not response.geturl().startswith(("http://", "https://")):
                    raise VideoError("video URL redirected to an unsupported scheme")
                length = response.headers.get("Content-Length")
                if length is not None:
                    if not length.isdigit():
                        raise VideoError("video URL returned an invalid Content-Length")
                    length = int(length)
                    if length > p["max_source_bytes"]:
                        raise VideoError("video source exceeds its byte budget")
                size = 0
                while chunk := response.read(65536):
                    size += len(chunk)
                    if size > p["max_source_bytes"]:
                        raise VideoError("video source exceeds its byte budget")
                    out.write(chunk)
                if length is not None and size != length:
                    raise VideoError("video URL returned an incomplete body")
        except OSError:
            raise VideoError("video URL could not be downloaded") from None
    del source, spec
    request.unlink(missing_ok=True)
    common = ["-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", "mov,matroska,avi,mpegts"]
    probe = directory / "probe.json"
    record_stage("decode")
    _command([p["ffprobe"], *common, "-threads", "1", "-select_streams", "v:0", "-show_frames", "-show_streams",
              "-show_entries", "stream=width,height,duration,sample_aspect_ratio:stream_side_data=rotation:frame=best_effort_timestamp_time,width,height,duration_time,pkt_duration_time",
              "-of", "json", str(source_file)], probe, 16 << 20)
    try:
        meta = json.loads(probe.read_text())
        stream, = meta["streams"]
        width, height = stream["width"], stream["height"]
        if not (0 < width <= 8192 and 0 < height <= 8192 and width * height <= p["max_source_pixels"]):
            raise VideoError("video source dimensions exceed the pixel budget")
        if any(float(s.get("rotation", 0)) % 360 for s in stream.get("side_data_list", [])):
            raise VideoError("rotated videos must be normalized before upload")
        if stream.get("sample_aspect_ratio", "1:1") not in ("1:1", "0:1", "N/A"):
            raise VideoError("video with non-square pixels must be normalized before upload")
        frames = meta["frames"]
        if not 1 <= len(frames) <= p["max_source_frames"]:
            raise VideoError("video source frame budget exceeded")
        times = [Decimal(f["best_effort_timestamp_time"]) for f in frames]
        if any(not t.is_finite() for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise VideoError("video timestamps must be finite and strictly increasing")
        if any(f["width"] != width or f["height"] != height for f in frames):
            raise VideoError("video resolution changes are not supported")
        times = [t - times[0] for t in times]
        tail = float(frames[-1].get("duration_time") or frames[-1].get("pkt_duration_time") or 0)
        declared = float(stream.get("duration") or 0)
        if not math.isfinite(tail) or not math.isfinite(declared) or tail < 0 or declared < 0:
            raise VideoError("video duration metadata is invalid")
        if tail == 0 and declared == 0:
            raise VideoError("video has no finite duration metadata")
        if max(float(times[-1]) + tail, declared) > p["max_duration_s"]:
            raise VideoError("video exceeds the duration budget")
    except (KeyError, TypeError, ValueError, OverflowError, InvalidOperation) as e:
        if isinstance(e, VideoError):
            raise
        raise VideoError("video metadata is missing or invalid") from None
    # Nearest source PTS at a fixed grid; earlier frame wins ties. Always retain
    # the last frame; reject excess cost instead of silently reducing cadence.
    record_stage("sampling")
    chosen = set()
    fps = Decimal(str(p["fps"]))
    for k in range(math.floor(times[-1] * fps) + 1):
        target = Decimal(k) / fps
        at = bisect.bisect_left(times, target)
        candidates = [i for i in (at - 1, at) if 0 <= i < len(times)]
        chosen.add(min(candidates, key=lambda i: (abs(times[i] - target), i)))
    indices = sorted(chosen | {len(times) - 1})
    if len(indices) > p["max_frames"]:
        raise VideoError("video exceeds selected frame budget; reduce duration or server FPS")
    scale = min(1.0, math.sqrt(p["pair_tokens"] * 1024 / (width * height)))
    w, h = max(32, int(width * scale) // 32 * 32), max(32, int(height * scale) // 32 * 32)
    rows = w // 32 * (h // 32)
    size = w * h * 3
    if rows > p["pair_tokens"] or rows * ((len(indices) + 1) // 2) > p["max_tokens"] \
            or 16 + len(indices) * (8 + size) > 256 << 20:
        raise VideoError("video exceeds visual token or RGB budget")
    raw = directory / "frames.rgb"
    select = "+".join(f"eq(n\\,{i})" for i in indices)
    _command([p["ffmpeg"], *common, "-xerror", "-noautorotate", "-threads", "1", "-i", str(source_file),
              "-map", "0:v:0", "-an", "-sn", "-dn", "-filter_threads", "1", "-vf", f"select='{select}',scale={w}:{h}",
              "-fps_mode", "passthrough", "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
             raw, size * len(indices))
    if raw.stat().st_size != size * len(indices):
        raise VideoError("video decoder returned an incomplete frame sequence")
    with raw.open("rb") as pixels, (directory / "frames.svf").open("wb") as packet:
        packet.write(struct.pack("<4sIII", b"SVF2", w, h, len(indices)))
        for index in indices:
            packet.write(struct.pack("<d", float(times[index])))
            packet.write(pixels.read(size))
    raw.unlink()
    source_file.unlink()
    probe.unlink()
    record_stage("ready")


if __name__ == "__main__":
    request = Path(sys.argv[1])
    try:
        if os.name == "posix":
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
            resource.setrlimit(resource.RLIMIT_FSIZE, (300 << 20, 300 << 20))
        elif os.name == "nt":
            # A separate job per worker: killing one request must not kill other
            # model processes, and killing the worker must kill its codec too.
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from serve import winjob
            import ctypes
            job = winjob._make_job()
            limits = winjob._ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x100
            limits.ProcessMemoryLimit = 2 << 30
            if not job or not winjob._k32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)) \
                    or not winjob._k32.AssignProcessToJobObject(job, winjob._k32.GetCurrentProcess()):
                raise VideoError("video worker could not establish Windows process containment")
        _worker(request)
    except Exception as e:
        # Codec/library exceptions may include a URL. Only our controlled errors
        # cross the public boundary.
        detail = str(e) if isinstance(e, VideoError) else "video preparation failed"
        (request.parent / "error.json").write_text(json.dumps(detail))
        sys.exit(1)
