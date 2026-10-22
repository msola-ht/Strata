"""Run the Strata video pipeline acceptance checks on Windows.

This is a manual test driver, not part of the automated test suite. It copies
the selected model config for every run, so the person's real config is never
changed. The script builds the synthetic videos with FFmpeg, starts the server
from this checkout, uploads each video through /v1/files, and sends the
Responses request described in docs/VIDEO_PIPELINE.md.

Examples:

  python tools/windows/video_acceptance.py --config strata-iq2_xs.json
  python tools/windows/video_acceptance.py --config strata-iq2_xs.json --build
  python tools/windows/video_acceptance.py --config strata-iq2_xs.json --cpu
  python tools/windows/video_acceptance.py --config strata-iq2_xs.json --quick
  python tools/windows/video_acceptance.py --config strata-iq2_xs.json --prepare-only

The exit code is 0 when every selected check passes, 1 when a check fails, and
2 when the setup cannot start the checks.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "serve" / "server.py"
BUILD_VIDEO = ROOT / "BUILD-VIDEO.ps1"
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

VARIANTS = {
    "uniform": {"keyframes": "uniform", "fit": "fixed", "max_frames": 256},
    "hybrid": {"keyframes": "hybrid", "fit": "fixed", "max_frames": 256},
    "budget-fixed": {"keyframes": "uniform", "fit": "fixed", "max_frames": 32},
    "budget": {"keyframes": "uniform", "fit": "budget", "max_frames": 32},
}
ORDER = ("uniform", "hybrid", "budget-fixed", "budget")


class SetupError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def say(message: str = "") -> None:
    print(message, flush=True)


def fail(message: str) -> None:
    raise SetupError(message)


def executable(path: str | os.PathLike | None, label: str) -> Path:
    if not path:
        fail(f"{label} is not set")
    found = shutil.which(str(path))
    if found is None:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        fail(f"{label} was not found: {path}")
    return Path(found).resolve()


def check_platform() -> None:
    if os.name != "nt":
        fail("This driver starts a Windows Strata server. Run it on Windows.")
    if not SERVER.is_file():
        fail(f"serve/server.py is missing below {ROOT}")


def valid_config(path: Path) -> dict | None:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict) or not cfg.get("exe") or not isinstance(cfg.get("args"), list):
        return None
    vision = cfg.get("vision")
    if not isinstance(vision, dict) or not vision.get("mmproj") or not vision.get("model"):
        return None
    return cfg


def config_candidates() -> list[Path]:
    roots = [ROOT]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / "Strata" / "install")
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("strata-*.json")):
            if path.name.endswith(".shared-settings.json"):
                continue
            if valid_config(path) is not None and path not in found:
                found.append(path)
    return found


def choose_config(raw: str | None) -> Path:
    if raw:
        path = Path(raw).expanduser().resolve()
        if valid_config(path) is None:
            fail(f"not a Strata model config with exe, args and a vision section: {path}")
        return path
    candidates = config_candidates()
    if not candidates:
        fail("no model config found. Pass --config PATH (the folder's strata-<model>.json).")
    if len(candidates) == 1:
        say(f"Using model config: {candidates[0]}")
        return candidates[0]
    say("Several model configs were found:")
    for i, path in enumerate(candidates, 1):
        say(f"  {i}. {path}")
    if not sys.stdin.isatty():
        fail("more than one model config was found; choose one with --config PATH")
    while True:
        answer = input(f"Choose 1-{len(candidates)} [1]: ").strip() or "1"
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            return candidates[int(answer) - 1]
        say("Enter a number from the list.")


def strip_engine_args(args: list[str]) -> tuple[list[str], list[str]]:
    """Remove batching options from the temporary config only."""
    no_value = {"--batch-mtp"}
    with_value = {"--batch", "--slots", "--batch-groups"}
    out: list[str] = []
    removed: list[str] = []
    i = 0
    while i < len(args):
        item = str(args[i])
        if item in no_value:
            removed.append(item)
            i += 1
            continue
        if item in with_value:
            removed.append(item)
            if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                i += 2
            else:
                i += 1
            continue
        out.append(str(args[i]))
        i += 1
    return out, removed


def resolved_vision_path(cfg: dict, key: str) -> Path | None:
    value = (cfg.get("vision") or {}).get(key)
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    base = Path(cfg.get("cwd") or ".")
    if not base.is_absolute():
        base = ROOT / base
    return (base / path).resolve()


def build_encoder(args: argparse.Namespace) -> Path:
    if not BUILD_VIDEO.is_file():
        fail(f"BUILD-VIDEO.ps1 is missing below {ROOT}")
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh.exe") or shutil.which("pwsh")
    if not powershell:
        fail("PowerShell was not found")
    cmd = [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(BUILD_VIDEO),
           "-Jobs", str(args.jobs)]
    if args.cpu:
        cmd.append("-Cpu")
    if args.vs_path:
        cmd += ["-VsPath", args.vs_path]
    say("Building strata-vision ...")
    code = subprocess.run(cmd, cwd=ROOT).returncode
    if code:
        fail(f"BUILD-VIDEO.ps1 failed with exit code {code}")
    return vision_build_path(cpu=args.cpu)


def vision_build_path(cpu: bool) -> Path:
    name = "build-vision-video-cpu" if cpu else "build-vision-video"
    path = ROOT / name / "bin" / "strata-vision.exe"
    if not path.is_file():
        fail(f"the encoder build output is missing: {path}")
    return path.resolve()


def encoder_is_cpu(path: Path) -> bool:
    text = str(path).lower()
    return "build-vision-video-cpu" in text or "/cpu/" in text or "\\cpu\\" in text


def choose_encoder(args: argparse.Namespace, cfg: dict) -> tuple[Path, bool]:
    if args.build:
        return build_encoder(args), not args.cpu
    if args.vision_exe:
        path = Path(args.vision_exe).expanduser().resolve()
        if not path.is_file():
            fail(f"vision encoder not found: {path}")
        return path, not args.cpu and not encoder_is_cpu(path)
    names = ("build-vision-video-cpu",) if args.cpu else \
        ("build-vision-video", "build-vision-video-cpu")
    for name in names:
        built = ROOT / name / "bin" / "strata-vision.exe"
        if built.is_file():
            return built.resolve(), not args.cpu and not encoder_is_cpu(built)
    configured = resolved_vision_path(cfg, "exe")
    if configured and configured.is_file():
        gpu = False if args.cpu or encoder_is_cpu(configured) else \
            (cfg.get("vision") or {}).get("gpu", True) is not False
        return configured, gpu
    fail("no video encoder found. Run TEST-VIDEO.bat --build, pass --build, or pass --vision-exe PATH.")


def run_ffmpeg(command: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(command, cwd=ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        detail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
        fail(f"command failed: {' '.join(command)}\n{detail[-4000:]}")
    return proc


def make_videos(work: Path, ffmpeg: Path) -> dict:
    scenes = work / "scenes.mp4"
    long_video = work / "long.mp4"
    say("Creating synthetic videos ...")
    run_ffmpeg([
        str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=640x360:r=30:d=2",
        "-f", "lavfi", "-i", "color=c=white:s=640x360:r=30:d=2",
        "-f", "lavfi", "-i", "color=c=red:s=640x360:r=30:d=2",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-movflags", "+faststart", str(scenes),
    ])
    run_ffmpeg([
        str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", "testsrc2=s=640x360:r=10:d=60",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(long_video),
    ])
    return {"scenes": scenes, "long": long_video}


def scene_probe(ffmpeg: Path, video: Path) -> dict:
    filters = run_ffmpeg([str(ffmpeg), "-hide_banner", "-filters"]).stdout or ""
    available = bool(re.search(r"^\s*\S+\s+scdet\s", filters, re.MULTILINE))
    if not available:
        return {"scdet": False, "cuts": None}
    proc = run_ffmpeg([
        str(ffmpeg), "-hide_banner", "-v", "error", "-i", str(video),
        "-map", "0:v:0", "-an", "-vf",
        "scale=320:-2,scdet=threshold=10:sc_pass=1,metadata=print:file=-",
        "-f", "null", "-",
    ])
    cuts = len(re.findall(r"lavfi\.scd\.score=", proc.stdout or ""))
    return {"scdet": True, "cuts": cuts}


def temporary_config(base: dict, *, work: Path, variant: str,
                     vision_exe: Path, gpu: bool, ffmpeg: Path, ffprobe: Path,
                     api_key: str, port: int) -> Path:
    cfg = deepcopy(base)
    cfg["api_key"] = api_key
    cfg["host"] = "127.0.0.1"
    cfg["port"] = port
    cfg["open_browser"] = False
    cfg["parallel"] = 1
    cfg["lazy_load"] = False
    cfg["log"] = str(work / f"{variant}-server.log")
    cfg["args"], _removed = strip_engine_args(cfg.get("args") or [])
    vision = cfg.setdefault("vision", {})
    vision["exe"] = str(vision_exe)
    vision["gpu"] = gpu
    settings = {
        "enabled": True,
        "diagnostics": True,
        "ffmpeg": str(ffmpeg),
        "ffprobe": str(ffprobe),
        "fps": 2,
        "max_duration_s": 120,
        "max_frames": VARIANTS[variant]["max_frames"],
        "pair_tokens": 256,
        "max_tokens": 32768,
        "max_source_bytes": 128 << 20,
        "deadline_s": 600,
        "keyframes": VARIANTS[variant]["keyframes"],
        "fit": VARIANTS[variant]["fit"],
        "scene_threshold": 10,
        "scene_share": 0.4,
        "scene_min_gap_s": 0.5,
    }
    vision["video"] = settings
    files = cfg.get("video_files")
    if not isinstance(files, dict):
        files = {}
    files["enabled"] = True
    cfg["video_files"] = files
    target = work / f"{variant}-config.json"
    target.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(url: str, *, method: str = "GET", key: str | None = None,
              body: dict | None = None, timeout: float = 30) -> tuple[int, dict]:
    headers = {"Accept": "application/json"}
    data = None
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with HTTP.open(request, timeout=timeout) as response:
            raw = response.read()
            return int(response.status), json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as error:
        with error:
            raw = error.read()
            try:
                parsed = json.loads(raw.decode("utf-8")) if raw else {}
            except (ValueError, UnicodeError):
                parsed = {"error": {"message": raw.decode("utf-8", "replace")}}
            return int(error.code), parsed


def upload_video(base: str, key: str, path: Path, timeout: float = 120) -> tuple[int, dict]:
    data = path.read_bytes()
    request = urllib.request.Request(
        f"{base}/v1/files",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "video/mp4",
            "Content-Length": str(len(data)),
        },
    )
    try:
        with HTTP.open(request, timeout=timeout) as response:
            raw = response.read()
            return int(response.status), json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as error:
        with error:
            raw = error.read()
            try:
                parsed = json.loads(raw.decode("utf-8")) if raw else {}
            except (ValueError, UnicodeError):
                parsed = {"error": {"message": raw.decode("utf-8", "replace")}}
            return int(error.code), parsed


def response_text(body: dict) -> str:
    parts: list[str] = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                parts.append(part.get("text") or "")
    return "\n".join(part for part in parts if part)


def ask_video(base: str, key: str, file_id: str, question: str, timeout: float) -> tuple[int, dict]:
    body = {
        "model": "strata",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_video", "file_id": file_id},
                {"type": "input_text", "text": question},
            ],
        }],
        "max_output_tokens": 256,
        "reasoning": {"effort": "none"},
    }
    return http_json(f"{base}/v1/responses", method="POST", key=key, body=body, timeout=timeout)


def start_server(config: Path, log: Path, port: int, timeout: float, *, elastic: bool):
    log.write_text("", encoding="utf-8")
    handle = log.open("ab")
    cmd = [sys.executable, str(SERVER), "--engine", "strata",
           "--config", str(config), "--port", str(port)]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    kwargs = {"cwd": str(ROOT), "stdout": handle, "stderr": subprocess.STDOUT, "env": env}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(cmd, **kwargs)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            handle.close()
            tail = log.read_text(encoding="utf-8", errors="replace")[-5000:]
            fail(f"the server stopped during startup (exit {proc.returncode})\n{tail}")
        try:
            _, health = http_json(f"{base}/health", timeout=5)
            if health.get("loaded") and (health.get("videos") or elastic):
                return proc, handle, base, health
            last_error = json.dumps(health, ensure_ascii=False)
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(2)
    stop_server(proc, handle)
    tail = log.read_text(encoding="utf-8", errors="replace")[-5000:]
    fail(f"the server did not become video-ready in {timeout:g}s: {last_error}\n{tail}")


def stop_server(proc: subprocess.Popen | None, handle) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=90)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    try:
                        proc.terminate()
                        proc.wait(timeout=10)
                    except (OSError, subprocess.TimeoutExpired):
                        proc.kill()
    finally:
        try:
            handle.close()
        except (OSError, ValueError):
            pass


def latest_trace(log: Path, pattern: str) -> re.Match | None:
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = list(re.finditer(pattern, text))
    return matches[-1] if matches else None


def check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail, "at": now_iso()}


def run_variant(variant: str, base_cfg: dict, args: argparse.Namespace,
                work: Path, videos: dict, vision_exe: Path, vision_gpu: bool, ffmpeg: Path,
                ffprobe: Path, api_key: str) -> dict:
    say()
    say(f"=== {variant} ===")
    port = free_port()
    config = temporary_config(base_cfg, work=work, variant=variant,
                              vision_exe=vision_exe, gpu=vision_gpu, ffmpeg=ffmpeg,
                              ffprobe=ffprobe, api_key=api_key, port=port)
    log = work / f"{variant}-server.log"
    checks: list[dict] = []
    proc = handle = None
    started = time.monotonic()
    vision_cfg = base_cfg.get("vision") or {}
    elastic = bool(vision_cfg.get("elastic") is True or vision_cfg.get("idle_unload_s"))
    say("Vision lifecycle: elastic (the first video starts the encoder)" if elastic
        else "Vision lifecycle: encoder starts with the model")
    try:
        proc, handle, base, health = start_server(
            config, log, port, args.start_timeout, elastic=elastic)
        checks.append(check("health/loaded", True,
                            f"loaded={health.get('loaded')} videos={health.get('videos')} "
                            f"elastic={elastic} context={health.get('max_context')}"))
        video = videos["long"] if variant in ("budget-fixed", "budget") else videos["scenes"]
        status, upload = upload_video(base, api_key, video, timeout=args.request_timeout)
        checks.append(check("upload", status == 200 and upload.get("status") == "ready",
                            f"HTTP {status}; {json.dumps(upload, ensure_ascii=False)[:800]}"))
        file_id = upload.get("id")
        if not file_id:
            checks.append(check("request", False, "upload returned no file_id"))
            return {"variant": variant, "vision_elastic": elastic, "config": str(config),
                    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "log": str(log),
                    "checks": checks, "elapsed_s": round(time.monotonic() - started, 2)}

        question = ("List the main colors in the order they appear." if variant in ("uniform", "hybrid")
                    else "Describe the main motion and the final scene.")
        status, response = ask_video(base, api_key, file_id, question, args.request_timeout)
        if variant == "budget-fixed":
            message = json.dumps(response, ensure_ascii=False)
            ok = status == 400 and ("frame budget" in message or "video_sampling_failed" in message)
            checks.append(check("request/expected-over-budget", ok, f"HTTP {status}; {message[:1200]}"))
        else:
            answer = response_text(response)
            ok = status == 200 and bool(answer.strip())
            checks.append(check("request", ok, f"HTTP {status}; answer={answer[:1000]!r}; "
                                               f"usage={json.dumps(response.get('usage') or {}, ensure_ascii=False)}"))
            if status == 200:
                health_status, after = http_json(f"{base}/health", timeout=10)
                checks.append(check("health/video-ready", health_status == 200 and after.get("videos") is True,
                                    f"HTTP {health_status}; videos={after.get('videos')}"))
            if variant == "budget" and status == 200:
                trace = latest_trace(log, r"video trace=\S+ stage=prompt .*?visual_tokens=(\d+)")
                visual = int(trace.group(1)) if trace else -1
                checks.append(check("budget/visual-tokens", 0 < visual <= 32768,
                                    f"visual_tokens={visual}"))
            if variant == "uniform" and status == 200:
                checks.append(check("cache/first-miss", "video cache=miss" in log.read_text(
                    encoding="utf-8", errors="replace"), "first request used a cache miss"))
                status2, response2 = ask_video(base, api_key, file_id,
                                               "Which color appears last?", args.request_timeout)
                answer2 = response_text(response2)
                cache_text = log.read_text(encoding="utf-8", errors="replace")
                checks.append(check("cache/second-hit", status2 == 200 and "video cache=hit" in cache_text,
                                    f"HTTP {status2}; answer={answer2[:500]!r}"))
            if variant == "hybrid" and status == 200 and bool(answer.strip()):
                checks.append(check("hybrid/no-crash", True,
                                    "the request completed; current worker logs do not expose selected cut timestamps"))
    except SetupError as error:
        checks.append(check("variant/setup", False, str(error)))
    except Exception as error:  # noqa: BLE001 - report the failed manual check and continue
        checks.append(check("variant/exception", False, f"{type(error).__name__}: {error}"))
    finally:
        stop_server(proc, handle)
    return {
        "variant": variant,
        "vision_elastic": elastic,
        "config": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "log": str(log),
        "checks": checks,
        "elapsed_s": round(time.monotonic() - started, 2),
    }


def write_report(work: Path, report: dict) -> Path:
    path = work / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Windows Strata video acceptance checks without changing the real model config.")
    parser.add_argument("config_file", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--config", help="strata-<model>.json; if omitted, the script finds it")
    parser.add_argument("--vision-exe", help="strata-vision.exe built with SVF2 video support")
    parser.add_argument("--build", action="store_true", help="run BUILD-VIDEO.ps1 before testing")
    parser.add_argument("--cpu", action="store_true", help="build/use the CPU encoder")
    parser.add_argument("--jobs", type=int, default=8, help="BUILD-VIDEO.ps1 job count (default 8)")
    parser.add_argument("--vs-path", help="pass -VsPath to BUILD-VIDEO.ps1")
    parser.add_argument("--ffmpeg", help="ffmpeg executable (default: PATH)")
    parser.add_argument("--ffprobe", help="ffprobe executable (default: PATH)")
    parser.add_argument("--only", nargs="+", choices=ORDER,
                        help="run only these variants: " + ", ".join(ORDER))
    parser.add_argument("--quick", action="store_true", help="run uniform and hybrid only")
    parser.add_argument("--prepare-only", action="store_true",
                        help="create videos and temporary configs, then stop")
    parser.add_argument("--work-dir", help="where logs, videos, configs and report.json are kept")
    parser.add_argument("--start-timeout", type=float, default=900,
                        help="seconds to wait for loaded=true and videos=true (default 900)")
    parser.add_argument("--request-timeout", type=float, default=1800,
                        help="seconds per upload/request (default 1800)")
    args = parser.parse_args(argv)

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.start_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")

    try:
        if not args.prepare_only:
            check_platform()
        source = choose_config(args.config or args.config_file)
        base_cfg = valid_config(source)
        if base_cfg is None:
            fail(f"invalid model config: {source}")
        ffmpeg = executable(args.ffmpeg or "ffmpeg", "ffmpeg")
        ffprobe = executable(args.ffprobe or "ffprobe", "ffprobe")
        vision_exe, vision_gpu = choose_encoder(args, base_cfg)
        work = Path(args.work_dir).expanduser().resolve() if args.work_dir else \
            Path(tempfile.mkdtemp(prefix="strata-video-acceptance-"))
        work.mkdir(parents=True, exist_ok=True)
        say(f"Work directory: {work}")
        say(f"Encoder: {vision_exe}")

        videos = make_videos(work, ffmpeg)
        probe = scene_probe(ffmpeg, videos["scenes"])
        say(f"Scene detector: available={probe['scdet']} cuts={probe['cuts']}")

        selected = list(args.only or (("uniform", "hybrid") if args.quick else ORDER))
        api_key = "vtest-" + secrets.token_hex(12)
        report = {
            "schema_version": 1,
            "started_at": now_iso(),
            "repo": str(ROOT),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False).stdout.strip(),
            "config_source": str(source),
            "config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "vision_exe": str(vision_exe),
            "cpu": bool(args.cpu),
            "vision_gpu": bool(vision_gpu),
            "ffmpeg": str(ffmpeg),
            "ffprobe": str(ffprobe),
            "videos": {k: {"path": str(v), "bytes": v.stat().st_size} for k, v in videos.items()},
            "scene_probe": probe,
            "selected_variants": selected,
            "variants": [],
        }
        if args.prepare_only:
            for variant in selected:
                temporary_config(base_cfg, work=work, variant=variant,
                                 vision_exe=vision_exe, gpu=vision_gpu, ffmpeg=ffmpeg,
                                 ffprobe=ffprobe, api_key="vtest-prepared", port=free_port())
            report["finished_at"] = now_iso()
            path = write_report(work, report)
            say(f"Prepared {len(selected)} variant config(s). Report: {path}")
            return 0

        for variant in selected:
            report["variants"].append(run_variant(
                variant, base_cfg, args, work, videos, vision_exe, vision_gpu,
                ffmpeg, ffprobe, api_key))
        report["finished_at"] = now_iso()
        all_checks = [item for variant in report["variants"] for item in variant["checks"]]
        failed = [item for item in all_checks if not item["ok"]]
        report["ok"] = not failed
        path = write_report(work, report)

        say()
        say("=== Summary ===")
        for variant in report["variants"]:
            status = "PASS" if all(item["ok"] for item in variant["checks"]) else "FAIL"
            say(f"{status:4}  {variant['variant']:12}  {variant['elapsed_s']:8.1f}s  {variant['log']}")
            for item in variant["checks"]:
                say(f"      {'ok' if item['ok'] else 'FAIL':4} {item['name']}: {item['detail']}")
        say()
        say(f"Report: {path}")
        say(f"Logs and generated videos: {work}")
        if probe["scdet"] and probe["cuts"] == 0:
            say("Note: this FFmpeg found no scene cuts in scenes.mp4; hybrid falls back to uniform.")
        return 0 if not failed else 1
    except SetupError as error:
        say(f"ERROR: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        say("Stopped by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
