"""Process-private, owner-scoped video uploads and bounded preparation cache.

No old directory is scanned or reused. Cache producers must use copy_artifact
to publish their bounded embedding artifact; decoder scratch belongs to the
existing video worker's separate budget. Only opaque IDs cross the HTTP API.
"""
from __future__ import annotations

from collections import OrderedDict
import contextlib
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import secrets
import tempfile
import threading
import time

from .video import Budget, VideoError


DEFAULTS = {
    "max_bytes": 2 << 30, "max_files": 32, "ttl_s": 3600,
    "directory": None, "cache_max_bytes": 1 << 30,
    "cache_max_entries": 16, "cache_ttl_s": 3600,
    "cache_max_entry_bytes": 671098880, "cache_max_workers": 1,
}


class StoreError(ValueError):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.status = code, status


def policy(config):
    """Validate store limits; the server separately controls feature enablement."""
    if not isinstance(config, dict):
        raise StoreError("invalid_request", "video_files must be an object")
    if set(config) - set(DEFAULTS) - {"enabled"}:
        raise StoreError("invalid_request", "unknown video_files setting")
    if "enabled" in config and not isinstance(config["enabled"], bool):
        raise StoreError("invalid_request", "video_files.enabled must be a boolean")
    result = {**DEFAULTS, **{k: v for k, v in config.items() if k != "enabled"}}
    ceilings = {"max_bytes": 8 << 30, "max_files": 512,
                "cache_max_bytes": 8 << 30, "cache_max_entries": 512,
                "cache_max_entry_bytes": 1 << 30, "cache_max_workers": 1}
    for key, ceiling in ceilings.items():
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= ceiling:
            raise StoreError("invalid_request", f"video_files.{key} must be positive and at most {ceiling}")
    for key in ("ttl_s", "cache_ttl_s"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or not 0 < value <= 30 * 86400:
            raise StoreError("invalid_request", f"video_files.{key} must be positive and at most 2592000")
    directory = result["directory"]
    if directory is not None and (not isinstance(directory, str) or not directory
                                  or any(c in directory for c in "\x00\r\n")):
        raise StoreError("invalid_request", "video_files.directory must name a temporary parent directory")
    if result["cache_max_entry_bytes"] > min(result["cache_max_bytes"], result["max_bytes"]):
        raise StoreError("invalid_request", "video cache entry limit exceeds the cache or total disk limit")
    return result


@dataclass(frozen=True)
class FileLease:
    path: Path
    sha256: str
    size: int


@dataclass
class _File:
    owner: str
    path: Path
    created: float
    expires: float
    deadline: float
    charge: int = 0
    size: int = 0
    sha256: str = ""
    state: str = "uploading"
    leases: int = 1  # The upload itself owns a lease until its writer closes.


@dataclass
class _Cache:
    key: str
    path: Path
    layout: str
    charge: int
    sha256: str
    deadline: float
    used: float
    sampling: str = ""
    leases: int = 0
    retired: bool = False


@dataclass
class _Flight:
    key: str
    path: Path
    cancel: threading.Event
    waiters: int = 0
    done: bool = False
    result: _Cache | None = None
    error: Exception | None = None
    resources: contextlib.ExitStack | None = None
    budget: Budget | None = None


class VideoStore:
    def __init__(self, config, max_source_bytes):
        self.settings = policy(config)
        if isinstance(max_source_bytes, bool) or not isinstance(max_source_bytes, int) or max_source_bytes <= 0:
            raise StoreError("invalid_request", "video source limit must be a positive integer")
        self.max_source_bytes = max_source_bytes
        try:
            self._temporary = tempfile.TemporaryDirectory(prefix="strata-video-", dir=self.settings["directory"])
        except OSError:
            raise StoreError("video_store_unavailable", "video temporary storage could not be created", 503) from None
        self._root = Path(self._temporary.name)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._workers = threading.BoundedSemaphore(self.settings["cache_max_workers"])
        self._files = {}
        self._tombstones = OrderedDict()
        self._cache = {}
        self._cache_all = {}
        self._flights = {}
        self._source_bytes = self._cache_bytes = 0
        self._closed = False
        self._cleanup_stop = threading.Event()
        try:
            threading.Thread(target=self._sweep, daemon=True).start()
        except Exception:
            self._temporary.cleanup()
            raise StoreError("video_store_unavailable", "video storage cleanup could not start", 503) from None

    def _open(self):
        if self._closed:
            raise StoreError("video_store_closed", "video storage is closed", 503)

    @staticmethod
    def _unlink(path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Do not release the disk reservation if deletion failed.
            return False
        return True

    def _retire_file(self, file_id, entry, state):
        entry.state = state
        self._tombstones[file_id] = (entry.owner, state)
        self._tombstones.move_to_end(file_id)
        while len(self._tombstones) > self.settings["max_files"] * 4:
            self._tombstones.popitem(last=False)
        if not entry.leases and self._unlink(entry.path):
            self._source_bytes -= entry.charge
            self._files.pop(file_id, None)

    def _retire_cache(self, entry):
        entry.retired = True
        if self._cache.get(entry.key) is entry:
            self._cache.pop(entry.key)
        if not entry.leases and self._unlink(entry.path):
            self._cache_bytes -= entry.charge
            self._cache_all.pop(entry.path, None)

    def _clean(self):
        now = time.monotonic()
        for file_id, entry in list(self._files.items()):
            if entry.state in ("deleted", "expired"):
                self._retire_file(file_id, entry, entry.state)
            elif now >= entry.deadline:
                self._retire_file(file_id, entry, "expired")
        for entry in list(self._cache_all.values()):
            if entry.retired or now >= entry.deadline:
                self._retire_cache(entry)

    def cleanup(self):
        with self._lock:
            self._clean()
            self._finish_close()

    def _sweep(self):
        interval = min(60, self.settings["ttl_s"], self.settings["cache_ttl_s"])
        while not self._cleanup_stop.wait(interval):
            self.cleanup()

    def _capacity(self, extra, *, cache=False, entry=False):
        """Reserve room by evicting only cache entries without active leases."""
        def fits():
            total_ok = self._source_bytes + self._cache_bytes + extra <= self.settings["max_bytes"]
            cache_ok = not cache or self._cache_bytes + extra <= self.settings["cache_max_bytes"]
            count = len(self._cache_all) + sum(not f.done for f in self._flights.values())
            return total_ok and cache_ok and (not entry or count < self.settings["cache_max_entries"])
        for candidate in sorted(self._cache_all.values(), key=lambda c: c.used):
            if fits():
                break
            if not candidate.leases:
                self._retire_cache(candidate)
        if not fits():
            if cache:
                raise StoreError("video_cache_full", "video preparation cache capacity is exhausted", 503)
            raise StoreError("video_store_full", "video storage capacity is exhausted", 507)

    def _lookup(self, owner, file_id):
        if not isinstance(file_id, str):
            raise StoreError("video_file_not_found", "video file was not found", 404)
        entry = self._files.get(file_id)
        state = entry.state if entry and entry.owner == owner else None
        if state is None:
            tombstone = self._tombstones.get(file_id)
            state = tombstone[1] if tombstone and tombstone[0] == owner else None
        if state in ("deleted", "expired"):
            raise StoreError(f"video_file_{state}", f"video file is {state}", 410)
        if state is None:
            raise StoreError("video_file_not_found", "video file was not found", 404)
        if state != "ready":
            raise StoreError("video_file_not_ready", "video file is not ready", 409)
        return entry

    @staticmethod
    def _metadata(file_id, entry):
        return {"id": file_id, "object": "video_file", "bytes": entry.size,
                "created_at": int(entry.created), "expires_at": int(entry.expires), "status": "ready"}

    def create(self, owner, stream, length=None):
        if not isinstance(owner, str) or not owner:
            raise StoreError("invalid_request", "video file owner is required")
        if length is not None and (isinstance(length, bool) or not isinstance(length, int) or length < 0):
            raise StoreError("invalid_request", "invalid video Content-Length")
        if length is not None and length > self.max_source_bytes:
            raise StoreError("video_file_too_large", "video source exceeds its byte budget", 413)
        with self._lock:
            self._open()
            self._clean()
            if len(self._files) >= self.settings["max_files"]:
                raise StoreError("video_store_full", "video file count limit is exhausted", 507)
            self._capacity(length or 0)
            file_id = "video_" + secrets.token_hex(24)
            now = time.time()
            entry = _File(owner, self._root / file_id, now, now + self.settings["ttl_s"],
                          time.monotonic() + self.settings["ttl_s"], charge=length or 0)
            self._files[file_id] = entry
            self._source_bytes += entry.charge
        digest = hashlib.sha256()
        try:
            with entry.path.open("xb") as out:
                while True:
                    # A sized HTTP body must not read into the next request.
                    if length is not None and entry.size == length:
                        break
                    chunk = stream.read(min(65536, (length - entry.size) if length is not None
                                             else self.max_source_bytes - entry.size + 1))
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise StoreError("invalid_request", "video upload must contain bytes")
                    size = entry.size + len(chunk)
                    if size > self.max_source_bytes:
                        raise StoreError("video_file_too_large", "video source exceeds its byte budget", 413)
                    if length is not None and size > length:
                        raise StoreError("invalid_request", "video upload exceeds Content-Length")
                    with self._lock:
                        self._open()
                        self._clean()
                        if entry.state != "uploading":
                            raise StoreError("video_file_expired", "video upload expired", 410)
                        if length is None:
                            self._capacity(len(chunk))
                            entry.charge += len(chunk)
                            self._source_bytes += len(chunk)
                    out.write(chunk)
                    digest.update(chunk)
                    entry.size = size
            if not entry.size or (length is not None and entry.size != length):
                raise StoreError("invalid_request", "video upload is empty or incomplete")
            with self._lock:
                self._open()
                self._clean()
                if entry.state != "uploading":
                    raise StoreError("video_file_expired", "video upload expired", 410)
                entry.sha256, entry.state = digest.hexdigest(), "ready"
                return self._metadata(file_id, entry)
        except TimeoutError:
            raise
        except OSError:
            raise StoreError("video_store_unavailable", "video upload could not be stored", 503) from None
        finally:
            with self._lock:
                entry.leases -= 1
                if entry.state == "uploading":
                    self._retire_file(file_id, entry, "deleted")
                elif entry.state in ("deleted", "expired"):
                    self._retire_file(file_id, entry, entry.state)
                self._finish_close()

    def status(self, owner, file_id):
        with self._lock:
            self._open()
            self._clean()
            return self._metadata(file_id, self._lookup(owner, file_id))

    get = status

    def delete(self, owner, file_id):
        with self._lock:
            self._open()
            self._clean()
            entry = self._lookup(owner, file_id)
            self._retire_file(file_id, entry, "deleted")
            return {"id": file_id, "object": "video_file", "deleted": True}

    @contextlib.contextmanager
    def lease(self, owner, file_id):
        with self._lock:
            self._open()
            self._clean()
            entry = self._lookup(owner, file_id)
            entry.leases += 1
            result = FileLease(entry.path, entry.sha256, entry.size)
        try:
            yield result
        finally:
            with self._lock:
                entry.leases -= 1
                if entry.state in ("deleted", "expired"):
                    self._retire_file(file_id, entry, entry.state)
                self._finish_close()

    @contextlib.contextmanager
    def retain(self, source):
        """Transfer an already authorized live pin to a producer, even after logical deletion."""
        with self._lock:
            self._open()
            item = next(((key, entry) for key, entry in self._files.items()
                         if entry.path == source.path and entry.sha256 == source.sha256 and entry.leases), None)
            if item is None:
                raise StoreError("video_file_not_found", "video source lease no longer exists", 404)
            file_id, entry = item
            entry.leases += 1
        try:
            yield source
        finally:
            with self._lock:
                entry.leases -= 1
                if entry.state in ("deleted", "expired"):
                    self._retire_file(file_id, entry, entry.state)
                self._finish_close()

    @staticmethod
    def cache_key(sha256, settings, model_identity):
        if not isinstance(sha256, str) or len(sha256) != 64 \
                or any(c not in "0123456789abcdef" for c in sha256):
            raise StoreError("invalid_request", "invalid video content digest")
        if not isinstance(model_identity, str) or not model_identity:
            raise StoreError("invalid_request", "video encoder identity is required")
        effective = {k: v for k, v in settings.items() if k != "diagnostics"}
        payload = json.dumps(["video-cache-v1", sha256, effective, model_identity],
                             sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(payload).hexdigest()

    def copy_artifact(self, source, target, cancel):
        """Only a flight's producer can publish, with a bound before every write."""
        target = Path(target)
        with self._lock:
            flight = next((f for f in self._flights.values() if f.path == target and f.cancel is cancel), None)
            if flight is None or flight.done:
                raise StoreError("invalid_request", "invalid video cache publication target")
        size = 0
        try:
            with Path(source).open("rb") as src, target.open("xb") as out:
                while chunk := src.read(65536):
                    flight.budget.check()
                    size += len(chunk)
                    if size > self.settings["cache_max_entry_bytes"]:
                        raise StoreError("video_cache_entry_too_large", "video embeddings exceed cache entry budget", 413)
                    out.write(chunk)
            if not size:
                raise StoreError("invalid_request", "video embedding artifact is empty")
        except OSError:
            raise StoreError("video_store_unavailable", "video embeddings could not be cached", 503) from None

    def _artifact_digest(self, path, check):
        digest, size = hashlib.sha256(), 0
        try:
            with path.open("rb") as source:
                while chunk := source.read(65536):
                    check()
                    size += len(chunk)
                    if size > self.settings["cache_max_entry_bytes"]:
                        raise StoreError("video_cache_corrupt", "video cache artifact is invalid", 503)
                    digest.update(chunk)
        except OSError:
            raise StoreError("video_cache_corrupt", "video cache artifact is unavailable", 503) from None
        if not size:
            raise StoreError("video_cache_corrupt", "video cache artifact is empty", 503)
        return size, digest.hexdigest()

    def _compute(self, flight, compute):
        acquired = False
        result = error = None
        try:
            with flight.budget.stage("producer_queue"):
                while not acquired:
                    flight.budget.check()
                    acquired = self._workers.acquire(timeout=0.1)
            flight.budget.check()
            computed = compute(flight.budget, flight.path)
            if isinstance(computed, tuple) and len(computed) == 2:
                layout, sampling = computed
            else:
                layout, sampling = computed, None
            encoded = json.dumps(layout, separators=(",", ":"), allow_nan=False)
            if len(encoded.encode()) > 65536:
                raise StoreError("invalid_request", "video layout exceeds its size budget")
            encoded_sampling = json.dumps(sampling, separators=(",", ":"), allow_nan=False) \
                if sampling is not None else ""
            if len(encoded_sampling.encode()) > 65536:
                raise StoreError("invalid_request", "video sampling metadata exceeds its size budget")
            with flight.budget.stage("checksum"):
                size, digest = self._artifact_digest(flight.path, flight.budget.check)
            flight.budget.check()
            result = _Cache(flight.key, flight.path, encoded, size, digest,
                            time.monotonic() + self.settings["cache_ttl_s"], time.monotonic(),
                            sampling=encoded_sampling)
        except Exception as exc:
            if isinstance(exc, VideoError):
                suffix = "cancelled" if flight.cancel.is_set() else \
                    "timeout" if time.monotonic() >= flight.budget.deadline else "failed"
                code = f"video_{getattr(exc, 'stage', 'preparation')}_{suffix}"
                error = StoreError(code, str(exc), 499 if suffix == "cancelled" else 408 if suffix == "timeout" else 400)
            else:
                error = exc if isinstance(exc, StoreError) else StoreError(
                    "video_preparation_failed", "video preparation failed", 502)
        finally:
            if acquired:
                self._workers.release()
            if flight.resources is not None:
                try:
                    flight.resources.close()
                except Exception:
                    result = None
                    error = StoreError("video_store_unavailable", "video source release failed", 503)
            with self._lock:
                if result is not None and time.monotonic() >= flight.budget.deadline:
                    result = None
                    error = StoreError("video_preparation_timeout", "video preparation deadline exceeded", 408)
                if result is not None and not flight.cancel.is_set() and not self._closed and flight.waiters:
                    self._cache_bytes -= self.settings["cache_max_entry_bytes"] - result.charge
                    result.leases = flight.waiters  # Pin results for all already joined waiters.
                    self._cache[flight.key] = result
                    self._cache_all[result.path] = result
                    flight.result = result
                else:
                    if self._unlink(flight.path):
                        self._cache_bytes -= self.settings["cache_max_entry_bytes"]
                    else:
                        # Retain a bounded reservation until cleanup can remove the file.
                        orphan = _Cache(flight.key, flight.path, "null", self.settings["cache_max_entry_bytes"], "",
                                        0, 0, retired=True)
                        self._cache_all[orphan.path] = orphan
                    if error is None:
                        error = StoreError("video_preparation_cancelled", "video preparation cancelled", 499)
                flight.error, flight.done = error, True
                if self._flights.get(flight.key) is flight:
                    self._flights.pop(flight.key, None)
                self._condition.notify_all()
                self._finish_close()
            flight.budget.trace.event("publication", event="result", outcome="ok" if flight.result else "failed",
                                     code=error.code if error else "none")

    @contextlib.contextmanager
    def get_or_compute(self, key, budget, compute, *, producer_context=None):
        """Join one producer; budget cancels only this waiter, never its peers.

        compute(worker_budget, target_path) returns a JSON-compatible layout or
        ``(layout, sampling)``, using copy_artifact to publish embeddings.
        producer_context is entered before the thread starts and closed by that
        thread, pinning its source independently of the first waiter. Its Budget
        must also belong to the producer. The yielded artifact is pinned: copy
        it into the request, never move it.
        """
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise StoreError("invalid_request", "invalid video cache key")
        flight = entry = None
        transferred = False
        hit = "hit"
        try:
            while True:
                budget.check()
                with self._condition:
                    self._open()
                    self._clean()
                    entry = self._cache.get(key)
                    if entry is not None:
                        entry.leases += 1
                        entry.used = time.monotonic()
                        break
                    existing = self._flights.get(key)
                    if existing is not None and existing.cancel.is_set():
                        # Do not add new demand to an already cancelled worker.
                        self._condition.wait(timeout=0.1)
                        continue
                    if existing is None:
                        self._capacity(self.settings["cache_max_entry_bytes"], cache=True, entry=True)
                        flight = _Flight(key, self._root / ("cache_" + secrets.token_hex(24)), threading.Event())
                        flight.budget = Budget(budget.settings, flight.cancel)
                        self._flights[key] = flight
                        self._cache_bytes += self.settings["cache_max_entry_bytes"]
                        flight.waiters = 1
                        hit = "miss"
                        try:
                            flight.resources = contextlib.ExitStack()
                            if producer_context is not None:
                                flight.resources.enter_context(producer_context)
                            threading.Thread(target=self._compute, args=(flight, compute), daemon=True).start()
                        except Exception:
                            try:
                                flight.resources.close()
                            finally:
                                self._flights.pop(key)
                                self._cache_bytes -= self.settings["cache_max_entry_bytes"]
                                flight = None
                            raise StoreError("video_store_unavailable", "video preparation could not start", 503) from None
                    else:
                        flight = existing
                        flight.waiters += 1
                        hit = "shared"
                    break
            budget.trace.event("cache", result=hit, producer=flight.budget.trace.id if flight else "none")
            if flight is not None:
                while True:
                    budget.check()
                    with self._condition:
                        self._open()
                        if flight.done:
                            if flight.error is not None:
                                raise flight.error
                            entry = flight.result
                            transferred = True
                            entry.used = time.monotonic()
                            break
                        self._condition.wait(timeout=0.1)
            budget.check()
            try:
                size, digest = self._artifact_digest(entry.path, budget.check)
                if size != entry.charge or digest != entry.sha256:
                    raise StoreError("video_cache_corrupt", "video cache artifact checksum does not match", 503)
            except StoreError:
                with self._lock:
                    self._retire_cache(entry)
                raise
            try:
                sampling = json.loads(entry.sampling) if entry.sampling else None
                yield entry.path, json.loads(entry.layout), sampling, hit
            except StoreError as error:
                if error.code == "video_cache_corrupt":
                    with self._lock:
                        self._retire_cache(entry)
                raise
        finally:
            with self._condition:
                if flight is not None:
                    flight.waiters -= 1
                    if not transferred and flight.result is not None:
                        flight.result.leases -= 1
                        if flight.result.retired:
                            self._retire_cache(flight.result)
                    if not flight.waiters:
                        if flight.done:
                            if self._flights.get(key) is flight:
                                self._flights.pop(key, None)
                        else:
                            flight.cancel.set()
                    self._condition.notify_all()
                if entry is not None and (flight is None or transferred):
                    entry.leases -= 1
                    if entry.retired:
                        self._retire_cache(entry)
                self._finish_close()

    def _finish_close(self):
        if self._closed and not self._files and not self._cache_all and not self._flights:
            try:
                self._temporary.cleanup()
            except OSError:
                return  # Keep the sweeper alive to retry a temporarily locked directory.
            self._cleanup_stop.set()

    def close(self):
        """Reject new work and cancel producers; active leases finish cleanup."""
        with self._condition:
            self._closed = True
            for flight in self._flights.values():
                flight.cancel.set()
            for file_id, entry in list(self._files.items()):
                self._retire_file(file_id, entry, "deleted")
            for entry in list(self._cache_all.values()):
                self._retire_cache(entry)
            self._condition.notify_all()
            self._finish_close()

    def wait_for_producers(self):
        """After close, drain producer scratch before shutting down its encoder directory."""
        with self._condition:
            if not self._closed:
                raise RuntimeError("close video storage before draining producers")
            while self._flights:
                self._condition.wait(timeout=0.1)
