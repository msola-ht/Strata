#!/usr/bin/env python3
"""Strata's MCP server with local video analysis.

This is the management server in ``strata_mcp.py`` plus one tool:
``strata_analyze_video``. It uploads a local video through ``/v1/files`` and
asks the running ``/v1/responses`` server to analyze it. The normal
``strata_mcp.py`` tool list is unchanged, which keeps existing MCP clients and
tests stable.

The video path must be inside the Strata folder or its configured data folder.
Use ``file_id`` for follow-up questions so the same uploaded file and cached
video artifact can be reused.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
import urllib.error
import urllib.request

try:
    from strata_mcp import (
        DEFAULT_ROOT,
        McpServer as BaseMcpServer,
        Strata,
        ToolError,
        Tools as BaseTools,
        http_json,
        log,
    )
except ModuleNotFoundError:                         # imported as tools.strata_video_mcp in the repo
    from tools.strata_mcp import (
        DEFAULT_ROOT,
        McpServer as BaseMcpServer,
        Strata,
        ToolError,
        Tools as BaseTools,
        http_json,
        log,
    )


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
MAX_VIDEO_BYTES = 512 << 20

VIDEO_TOOL = {
    "name": "strata_analyze_video",
    "title": "Analyze a local video",
    "description": "Upload a local video to the running Strata server and ask the model about it. "
                   "The path must be inside the Strata folder or its data folder. Use shots mode to "
                   "give the model the merged FFmpeg shot table; use file_id for follow-up questions. "
                   "This starts a model request and can take minutes.",
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "path": {"type": "string", "maxLength": 4096,
                     "description": "absolute path to a local video inside the Strata folder or data folder"},
            "file_id": {"type": "string", "maxLength": 80,
                        "description": "an uploaded video_file id from an earlier call, for a follow-up"},
            "question": {"type": "string", "maxLength": 4000,
                         "description": "the question for the video; default: describe it shot by shot"},
            "mode": {"type": "string", "enum": ["shots", "plain"],
                     "description": "shots inserts the FFmpeg shot table into the prompt; plain keeps the "
                                    "same request without that opt-in"},
            "max_shots": {"type": "integer", "minimum": 1, "maximum": 256,
                          "description": "maximum shot rows to insert (default 64)"},
            "max_output_tokens": {"type": "integer", "minimum": 64, "maximum": 4096,
                                  "description": "answer budget (default 512)"},
            "port": {"type": "integer", "minimum": 1024, "maximum": 65535,
                     "description": "the running server's port (default: the installed model's port)"},
            "timeout": {"type": "integer", "minimum": 30, "maximum": 3600,
                        "description": "seconds to wait for the answer (default 1800)"},
        },
    },
    "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
}


def _response_text(body: dict) -> str:
    parts = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                parts.append(part.get("text") or "")
    return "\n".join(part for part in parts if part)


def _http_upload(url: str, data: bytes, content_type: str, api_key: str | None, timeout: int):
    request = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
    )
    request.add_header("Content-Type", content_type)
    request.add_header("Content-Length", str(len(data)))
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw, status = response.read(), response.status
    except urllib.error.HTTPError as error:
        raw, status = error.read(), error.code
    except (OSError, ValueError):
        return 0, None
    try:
        return status, json.loads(raw.decode("utf-8") or "null")
    except (UnicodeError, ValueError):
        return status, None


def _video_type(path: Path) -> str:
    return {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
    }.get(path.suffix.lower(), "video/mp4")


class VideoTools(BaseTools):
    def defs(self) -> list:
        return super().defs() + [VIDEO_TOOL]

    def strata_analyze_video(self, path=None, file_id=None, question=None, mode="shots",
                             max_shots=64, max_output_tokens=512, port=None, timeout=1800) -> dict:
        s = self.s
        if bool(path) == bool(file_id):
            raise ToolError("give exactly one of path or file_id")
        if mode not in ("shots", "plain"):
            raise ToolError("mode must be shots or plain")
        if not 1 <= max_shots <= 256:
            raise ToolError("max_shots must be from 1 to 256")
        if not 64 <= max_output_tokens <= 4096:
            raise ToolError("max_output_tokens must be from 64 to 4096")
        if not 30 <= timeout <= 3600:
            raise ToolError("timeout must be from 30 to 3600 seconds")
        target = None
        for candidate in s.ports(port):
            info = s.probe(candidate, deep=False)
            if info and not info.get("other"):
                target = info
                break
        if target is None:
            raise ToolError("Strata is not running: start it first (strata_start)")
        server_port = target["port"]
        key = s.api_key_for(server_port)
        base = f"http://127.0.0.1:{server_port}"
        if path:
            source = Path(path).expanduser()
            if not source.is_absolute():
                raise ToolError("path must be absolute")
            if not source.is_file():
                raise ToolError(f"video not found: {source}")
            if not s.allowed(source):
                raise ToolError("video path must be inside the Strata folder or its data folder")
            size = source.stat().st_size
            if not size:
                raise ToolError(f"video is empty: {source}")
            if size > MAX_VIDEO_BYTES:
                raise ToolError(f"video is larger than {MAX_VIDEO_BYTES // (1 << 20)} MiB")
            status, uploaded = _http_upload(
                base + "/v1/files", source.read_bytes(), _video_type(source), key, timeout)
            if status != 200 or not isinstance(uploaded, dict) or not uploaded.get("id"):
                message = (uploaded or {}).get("error", {}).get("message") if isinstance(uploaded, dict) else None
                raise ToolError(f"video upload failed (HTTP {status}): {message or 'no file id'}")
            file_id = uploaded["id"]
        if not isinstance(file_id, str) or not file_id.startswith("video_"):
            raise ToolError("file_id must be a video_file id")
        body = {
            "model": target.get("model") or "strata",
            "input": [{"role": "user", "content": [
                {"type": "input_video", "file_id": file_id},
                {"type": "input_text", "text": question or "Describe this video shot by shot."},
            ]}],
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": "none"},
            "stream": False,
        }
        if mode == "shots":
            body["video_analysis"] = {"mode": "shots", "max_shots": max_shots}
        status, response = http_json(base + "/v1/responses", "POST", body, api_key=key, timeout=timeout)
        if status != 200 or not isinstance(response, dict):
            message = response.get("error", {}).get("message") if isinstance(response, dict) else None
            raise ToolError(f"video analysis failed (HTTP {status}): {message or 'no answer'}")
        answer = _response_text(response)
        if not answer:
            raise ToolError("video analysis returned no text")
        return {
            "summary": f"analysed {file_id} on port {server_port}",
            "file_id": file_id,
            "port": server_port,
            "mode": mode,
            "answer": answer,
            "usage": response.get("usage"),
        }


class VideoMcpServer(BaseMcpServer):
    def __init__(self, strata: Strata, out=None):
        super().__init__(strata, out)
        self.tools = VideoTools(strata)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Strata's MCP server with local video analysis (stdio).")
    parser.add_argument("--root", help="the Strata folder (default: the one this file is in)")
    parser.add_argument("--src", help="where setup.py, tools/ and serve/ are")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        log("Python 3.10 or newer is needed")
        return 2
    root = Path(args.root).expanduser().resolve() if args.root else DEFAULT_ROOT
    src = Path(args.src).expanduser().resolve() if args.src else root
    if not (src / "setup.py").is_file() or not (src / "serve" / "server.py").is_file():
        log(f"{src} is not a Strata folder (no setup.py / serve/server.py)")
        return 2
    out = sys.stdout.buffer
    sys.stdout = sys.stderr
    server = VideoMcpServer(Strata(root, src), out)
    log(f"ready with video analysis (Strata folder {root})")
    try:
        server.serve(sys.stdin.buffer)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
