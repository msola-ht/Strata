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
import re
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
                    or not isinstance(cells, int) or not 0 < cells <= settings["pair_tokens"]:
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
# Optional keys are deliberately not merged into DEFAULTS: a config that does
# not name them keeps the same settings dictionary, and therefore the same
# video cache key, as before this change.
OPTIONAL = {"keyframes", "fit", "scene_threshold", "scene_share", "scene_min_gap_s"}


def policy(config):
    if config is None or config is False:
        return None
    if not isinstance(config, dict) or not isinstance(config.get("enabled", False), bool):
        raise VideoError("vision.video must be an object with enabled: true or false")
    if not config.get("enabled", False):
        return None
    if set(config) - set(DEFAULTS) - OPTIONAL - {"enabled", "diagnostics"}:
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
    if "keyframes" in result and result["keyframes"] not in ("uniform", "hybrid"):
        raise VideoError("vision.video.keyframes must be \"uniform\" or \"hybrid\"")
    if "fit" in result and result["fit"] not in ("fixed", "budget"):
        raise VideoError("vision.video.fit must be \"fixed\" or \"budget\"")
    if "scene_threshold" in result:
        value = result["scene_threshold"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or not 0 < value <= 100:
            raise VideoError("vision.video.scene_threshold must be greater than 0 and at most 100")
    if "scene_share" in result:
        value = result["scene_share"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or not 0 < value < 1:
            raise VideoError("vision.video.scene_share must be greater than 0 and less than 1")
    if "scene_min_gap_s" in result:
        value = result["scene_min_gap_s"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or not 0 < value <= result["max_duration_s"]:
            raise VideoError("vision.video.scene_min_gap_s must be positive and at most max_duration_s")
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
        from serve import lifeline
        proc = lifeline.popen([sys.executable, str(Path(__file__).resolve()), str(request)], "video",
                              stdin=subprocess.PIPE if os.name == "nt" else subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=stderr,
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
            if os.name == "nt":
                # Attach to the server's Job before the worker creates its nested per-request Job. Otherwise a
                # fast worker can establish its own hierarchy first and make the server's assignment fail.
                import ctypes
                from ctypes import wintypes
                from serve import winjob
                contained = wintypes.BOOL()
                check = winjob._k32.IsProcessInJob
                check.argtypes = (wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL))
                if not winjob._job or not check(int(proc._handle), winjob._job, ctypes.byref(contained)) \
                        or not contained.value:
                    raise VideoError("video worker could not establish Windows parent containment")
                budget.check()
                proc.stdin.write(b"\x01")
                proc.stdin.close()
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
            if proc.stdin is not None:
                with contextlib.suppress(OSError, ValueError):
                    proc.stdin.close()
            stop(proc)
            request.unlink(missing_ok=True)
            budget.current_stage = "preparation"
    return directory / "frames.svf"


def _command(args, output, limit):
    """Bound both pipes while draining; never let codec diagnostics fill disk/RAM."""
    from serve import lifeline
    # PDEATHSIG is not inherited through fork: every codec needs its own lifeline to the worker. On macOS its
    # watcher owns a separate pipe to the worker; on Windows it also inherits the worker's per-request Job.
    # Keep the worker's POSIX process group so request cancellation still ends the entire decode immediately.
    proc = lifeline.popen(args, "video-codec", stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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


def _nearest(times, target):
    target = Decimal(str(target))
    at = bisect.bisect_left(times, target)
    candidates = [i for i in (at - 1, at) if 0 <= i < len(times)]
    return min(candidates, key=lambda i: (abs(times[i] - target), i))


def _uniform_indices(times, fps):
    """The nearest source frame at every 1/fps step; earlier frame wins ties."""
    fps = Decimal(str(fps))
    chosen = set()
    for k in range(math.floor(times[-1] * fps) + 1):
        chosen.add(_nearest(times, Decimal(k) / fps))
    return sorted(chosen | {len(times) - 1})


def _pairs_from_indices(indices):
    """Pair selected frames in order; the last pair may contain one frame."""
    return [(indices[i], indices[i + 1] if i + 1 < len(indices) else indices[i])
            for i in range(0, len(indices), 2)]


def _flatten_pairs(pairs):
    """Put disjoint pairs back into the one chronological frame list the packet stores."""
    out = []
    for a, b in sorted(pairs):
        if out and a <= out[-1]:
            continue                                # defensive: selection already filters overlaps
        out.append(a)
        if b != a:
            out.append(b)
    return out


def _select_expression(indices):
    """A balanced OR tree; a flat a+b+... chain hits FFmpeg's expression depth limit near 100 frames."""
    terms = [f"eq(n\\,{index})" for index in indices]
    while len(terms) > 1:
        terms = [f"({terms[i]}+{terms[i + 1]})" if i + 1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    return terms[0]


def _thin_pairs(pairs, limit):
    """Keep the first and last pair and spread the rest evenly."""
    if len(pairs) <= limit:
        return list(pairs)
    if limit <= 1:
        return [pairs[-1]]
    picks = [round(i * (len(pairs) - 1) / (limit - 1)) for i in range(limit)]
    return [pairs[i] for i in dict.fromkeys(picks)]


def _cut_pair(times, cut, fps):
    """One pair centred on a cut, at the same time span as a uniform pair."""
    fps = Decimal(str(fps))
    half = Decimal("0.5") / fps
    cut = Decimal(str(cut))
    a = _nearest(times, max(Decimal(0), cut - half))
    b = _nearest(times, min(times[-1], cut + half))
    if b <= a:
        at = bisect.bisect_left(times, cut)
        a, b = max(0, at - 1), min(len(times) - 1, at)
    if b <= a:
        if a + 1 < len(times):
            b = a + 1
        elif a > 0:
            a, b = a - 1, a
        else:
            return None
    return a, b


def _i_pair(times, index):
    if index + 1 < len(times):
        return index, index + 1
    if index > 0:
        return index - 1, index
    return None


def _overlaps(pair, selected):
    a, b = pair
    return any(a <= d and c <= b for c, d in selected)


def _select_pairs(times, settings, rows, cuts=(), key_indices=()):
    """Choose chronological, disjoint frame pairs within the frame and token budgets."""
    fps = float(settings["fps"])
    # Two frames per pair: an odd max_frames leaves the last slot unused rather
    # than letting a 2-frame pair exceed the configured frame budget.
    max_pairs = min(max(1, settings["max_frames"] // 2), settings["max_tokens"] // rows)
    if max_pairs < 1:
        raise VideoError("video exceeds its visual token budget; reduce pair_tokens or max_tokens")

    effective = fps
    if settings["max_frames"] < 2:
        return [(len(times) - 1, len(times) - 1)], effective
    uniform = _pairs_from_indices(_uniform_indices(times, effective))
    if settings.get("fit", "fixed") == "budget" and len(uniform) > max_pairs:
        duration = max(float(times[-1]), 1e-9)
        # Each pair holds two frames, so the grid needs about duration * fps / 2
        # pairs. Solve that for the FPS that fills the pair budget.
        effective = max(1.0 / 60.0, min(fps, 2.0 * max_pairs / duration))
        uniform = _pairs_from_indices(_uniform_indices(times, effective))
        uniform = _thin_pairs(uniform, max_pairs)

    if settings.get("keyframes", "uniform") == "uniform":
        if len(uniform) > max_pairs:
            raise VideoError("video exceeds selected frame budget; reduce duration or server FPS")
        return uniform, effective

    if max_pairs == 1:
        selected = [(len(times) - 1, len(times) - 1)]
    else:
        first = (0, 1) if len(times) > 1 else (0, 0)
        if len(times) <= 2:
            last = first
        elif len(times) == 3:
            last = (2, 2)                 # a one-frame pair; the encoder duplicates it
        else:
            last = (len(times) - 2, len(times) - 1)
        selected = [first] if first == last else [first, last]

    scene_limit = max(0, int(max_pairs * float(settings.get("scene_share", 0.4))))
    gap = float(settings.get("scene_min_gap_s", 0.5))
    scene_pairs = []

    for cut, _score in sorted(cuts, key=lambda c: (-c[1], c[0])):
        if len(scene_pairs) >= scene_limit or len(selected) >= max_pairs:
            break
        pair = _cut_pair(times, cut, effective)
        if pair is None or _overlaps(pair, selected):
            continue
        if any(abs(times[pair[0]] - times[s[0]]) < gap for s in scene_pairs):
            continue
        scene_pairs.append(pair)
        selected.append(pair)

    for index in sorted(key_indices):
        if len(scene_pairs) >= scene_limit or len(selected) >= max_pairs:
            break
        pair = _i_pair(times, index)
        if pair is None or _overlaps(pair, selected):
            continue
        if any(abs(times[pair[0]] - times[s[0]]) < gap for s in scene_pairs):
            continue
        scene_pairs.append(pair)
        selected.append(pair)

    for pair in uniform:
        if len(selected) >= max_pairs:
            break
        if _overlaps(pair, selected):
            continue
        if any(abs(times[pair[0]] - times[s[0]]) < gap for s in scene_pairs):
            continue
        selected.append(pair)

    return sorted(selected), effective


def _scene_cuts(source_file, settings, common):
    """FFmpeg's scene score at 320 px; an unsupported filter falls back to uniform."""
    output = source_file.parent / "scenes.txt"
    threshold = float(settings.get("scene_threshold", 10.0))
    filt = f"scale=320:-2,scdet=threshold={threshold:g}:sc_pass=1,metadata=print:file=-"
    try:
        _command([settings["ffmpeg"], *common, "-threads", "1", "-i", str(source_file),
                  "-map", "0:v:0", "-an", "-sn", "-dn", "-filter_threads", "1",
                  "-vf", filt, "-f", "null", "-"], output, 1 << 20)
    except (VideoError, OSError):
        output.unlink(missing_ok=True)
        print("[strata] video scene=fallback", file=sys.stderr, flush=True)
        return []
    try:
        text = output.read_text(encoding="utf-8", errors="replace")
    except OSError:
        print("[strata] video scene=fallback", file=sys.stderr, flush=True)
        return []
    finally:
        output.unlink(missing_ok=True)
    cuts = {}
    current = None
    for line in text.splitlines():
        if "pts_time:" in line:
            match = re.search(r"pts_time:([0-9eE+.-]+)", line)
            try:
                current = float(match.group(1)) if match else None
            except ValueError:
                current = None
            continue
        match = re.search(r"lavfi\.scd\.score=([0-9eE+.-]+)", line)
        if match and current is not None:
            try:
                score = float(match.group(1))
            except ValueError:
                score = 0.0
            if math.isfinite(current) and math.isfinite(score) and current >= 0:
                key = round(current, 6)
                cuts[key] = max(score, cuts.get(key, 0.0))
            current = None
    print(f"[strata] video scene=cuts count={len(cuts)}", file=sys.stderr, flush=True)
    return sorted(cuts.items())


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
              "-show_entries", "stream=width,height,duration,sample_aspect_ratio:stream_side_data=rotation:frame=key_frame,best_effort_timestamp_time,width,height,duration_time,pkt_duration_time",
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
        origin = times[0]
        times = [t - origin for t in times]
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
    # The vision encoder turns each pair into one temporal patch. Select pairs
    # before decoding so a scene cut cannot shift every later pair boundary.
    record_stage("sampling")
    scale = min(1.0, math.sqrt(p["pair_tokens"] * 1024 / (width * height)))
    w, h = max(32, int(width * scale) // 32 * 32), max(32, int(height * scale) // 32 * 32)
    rows = w // 32 * (h // 32)
    size = w * h * 3
    cuts = []
    if p.get("keyframes", "uniform") == "hybrid":
        cuts = [(float(stamp) - float(origin), score) for stamp, score in _scene_cuts(source_file, p, common)]
        cuts = [(stamp, score) for stamp, score in cuts if 0 <= stamp <= float(times[-1])]
    key_indices = [i for i, frame in enumerate(frames) if frame.get("key_frame") == 1]
    pairs, effective = _select_pairs(times, p, rows, cuts, key_indices)
    indices = _flatten_pairs(pairs)
    if len(indices) > p["max_frames"]:
        raise VideoError("video exceeds selected frame budget; reduce duration or server FPS")
    if rows > p["pair_tokens"] or rows * len(pairs) > p["max_tokens"] \
            or 16 + len(indices) * (8 + size) > 256 << 20:
        raise VideoError("video exceeds visual token or RGB budget")
    print(f"[strata] video selected_pairs={len(pairs)} frames={len(indices)} "
          f"effective_fps={effective:.6g}", file=sys.stderr, flush=True)
    raw = directory / "frames.rgb"
    select = _select_expression(indices)
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
    # This file is also executed by path, where Python otherwise searches serve/ rather than the package root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    request = Path(sys.argv[1])
    try:
        if os.name == "nt" and sys.stdin.buffer.read(1) != b"\x01":
            raise VideoError("video worker parent ended before startup")
        if os.name == "posix":
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
            resource.setrlimit(resource.RLIMIT_FSIZE, (300 << 20, 300 << 20))
        elif os.name == "nt":
            # A separate job per worker: killing one request must not kill other
            # model processes, and killing the worker must kill its codec too.
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
