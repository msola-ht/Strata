# Video input pipeline

This page describes the opt-in video path end to end, the parts of the
Qwen3.8-Flash-Next contract it has to keep, and the sampling rules the code
follows. It is the design document for the `feat/video-pair-selection` branch.

## What the model expects

Qwen3.8-Flash-Next accepts video through the same vision encoder as pictures.
The important parts for this pipeline are:

- The official video input samples at 2 frames per second by default
  (`fps=2`, `do_sample_frames=True`).
- The vision encoder uses `patch_size=16`, `temporal_patch_size=2` and
  `spatial_merge_size=2`. Two consecutive frames therefore form one temporal
  patch, not two independent pictures.
- Qwen3.5-family video input separates frames with timestamps in the text, in
  the form `<t1> <vision_start> <frame1> <vision_end> <t2> ...`.
- The model has a 262,144-token context natively. The released
  `video_preprocessor_config.json` is conservative; the model card suggests
  `longest_edge=469762048` (about 224k video tokens) for hour-scale video.

Sources: the [model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), its
[config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json),
[video_preprocessor_config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/video_preprocessor_config.json),
and the `get_rope_index` notes in
[modeling_qwen4_exp.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py).

## The chain

| Stage | Code | What it owns |
| --- | --- | --- |
| Request parsing | `serve/frontend.py` | `video_url`, `input_video`, `file_id` and `video` parts; video is accepted on `POST /v1/responses` only. |
| Request budget | `serve/server.py` (`_prepare_video`) | One video per request, prompt marker, context room, model and encoder readiness. |
| Frame preparation | `serve/video.py` (`prepare` and `_worker`) | Source download, `ffprobe`, sampling, bounded decode, `SVF2` packet. |
| Frame encoding | `tools/vision/strata_vision.cpp` | Pairs frames, runs the Qwen vision encoder, writes `SVE1` records and the `S<seconds>;I<rows>` layout. |
| Prompt assembly | `serve/server.py` | Replaces the video marker with a timestamp and one vision span per pair. |
| Text engine | `src/program/generate.cpp` | Lays the visual rows over `<|image_pad|>` cells and builds the M-RoPE table. |
| Cache | `serve/video_store.py` | Owns uploaded files and encoder artifacts; keys include the video settings and encoder identity. |

The local engine follows llama.cpp's `qwen4exp` path: all visual cells are
covered as `<|image_pad|>` cells, and a video pair is one temporal-merged image
chunk. The model config also has a separate `video_token_id`, but the local
llama.cpp path does not expose it. Do not change that token id on its own: it
would require matching PLE, M-RoPE and every backend.

## Invariants

1. **A pair is one temporal patch.** The unit of selection is a pair of source
   frames, not a single frame. Inserting a single keyframe would shift every
   later pair boundary and change the model's time structure.
2. **Pairs are disjoint and chronological.** A source frame appears at most
   once. Selected pairs are sorted by their first frame and flattened in that
   order.
3. **The timestamp is the pair midpoint.** The existing `S<seconds>` label is
   the midpoint of the two frames. It stays finite and strictly increasing.
4. **The budget is shared.** Scene pairs replace uniform pairs; they never add
   frames or tokens beyond `max_frames` and `max_tokens`.
5. **The default path is unchanged.** `keyframes: "uniform"` produces the same
   frames, timestamps and packet as before.
6. **Scene detection is best effort.** A missing `scdet` filter, an unsupported
   FFmpeg build, a timeout or an unreadable scene log falls back to uniform
   sampling and is visible in diagnostics; it does not fail the request.
7. **Cache keys describe the output.** Any setting that changes frame choice
   or token count must be part of the cache key. Optional settings are only
   added to the settings dictionary when they are configured, so existing
   uniform caches keep their keys.

## Frame selection

### Uniform

`uniform` is the default and reproduces the previous nearest-PTS grid. For each
`k / fps` target it chooses the nearest source timestamp, always keeps the last
frame, and pairs the chosen frames in order. The last pair may contain one
frame; the vision encoder duplicates it.

### Hybrid

`keyframes: "hybrid"` starts with the uniform pairs and replaces some of them
with scene-aware pairs:

- The first and last frame are reserved as endpoint pairs, so a run of scene
  cuts cannot drop the beginning or end of the video. With three source frames
  the last endpoint is a one-frame pair that the encoder duplicates.
- A scene cut at time `c` contributes one pair, centred on the cut: the nearest
  frame before `c - 0.5 / fps` and the nearest frame after `c + 0.5 / fps`.
  Centring the pair keeps its time span close to a uniform pair instead of
  showing two nearly identical frames.
- A keyframe (I-frame) candidate at index `i` contributes `(i, i + 1)`, or
  `(i - 1, i)` at the end of the video.
- After the endpoint pairs, cut pairs have priority over I-frame pairs, which
  have priority over uniform pairs. Cut and I-frame pairs together may use at
  most `scene_share` of the pair budget.
- A pair that shares a frame with an already selected pair is skipped. Pairs
  whose start times are closer than `scene_min_gap_s` are also skipped.
- The remaining budget is filled with uniform pairs, so a video with no cuts
  still has full time coverage.

Scene cuts come from one low-resolution FFmpeg pass:

```sh
ffmpeg -v error -i input.mp4 -map 0:v:0 -an -sn -dn \
  -vf "scale=320:-2,scdet=threshold=10:sc_pass=1,metadata=print:file=-" \
  -f null -
```

`sc_pass=1` drops frames below the threshold, so the metadata log stays small.
The `pts_time` values are absolute PTS; subtract the first source PTS before
mapping them onto the internal time axis.

I-frames come from the existing `ffprobe -show_frames` pass. Adding
`key_frame` to that query costs no extra decode.

### Budget fit

`fit: "budget"` lets a long video use the available pair budget instead of
being refused. The pair budget is:

```text
min(max_frames // 2, max_tokens // rows_per_pair)
```

`rows_per_pair` is computed from the source size and `pair_tokens`, as before.
If the uniform grid needs more pairs than the budget, the effective FPS is
lowered until it fits. The budget path then picks the exact frame count evenly
from the source, keeping both endpoints, so there is no interior pair to drop
and no larger gap in the middle. The trace reports the effective FPS.
`fit: "fixed"` is the default and keeps the previous behaviour: excess frames
are an error, never a silent cadence change.

The server also lowers `max_tokens` to the actual room left in the request
context before preparing the video, leaving a small margin for timestamp text
and markers. This makes a 32k-context request use a smaller video budget
instead of preparing an artifact that can never fit the prompt.

## Configuration

The settings live in the model config's `vision.video` object. The existing
defaults are unchanged.

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Turns video input on. |
| `fps` | `2.0` | Uniform sampling rate. |
| `max_duration_s` | `120.0` | Longest accepted video. |
| `max_frames` | `256` | Largest selected frame count. |
| `max_tokens` | `32768` | Largest video token count. |
| `keyframes` | `"uniform"` | `"uniform"` or `"hybrid"`. |
| `fit` | `"fixed"` | `"fixed"` or `"budget"`. |
| `scene_threshold` | `10.0` | FFmpeg `scdet` threshold, 0 to 100. |
| `scene_share` | `0.4` | Maximum share of pairs used by cut and I-frame pairs. |
| `scene_min_gap_s` | `0.5` | Minimum start-time gap between scene pairs; also merges nearby shot candidates in diagnostics. |

Example for a long, scene-heavy video:

```json
{
  "vision": {
    "video": {
      "enabled": true,
      "max_duration_s": 600,
      "keyframes": "hybrid",
      "fit": "budget",
      "scene_threshold": 10
    }
  }
}
```

A long video may also need a larger `max_source_bytes` and `deadline_s`; the
defaults are sized for short uploads, and a 1080p ten-minute file can exceed
64 MiB before decoding.

### Diagnostics

With `"diagnostics": true`, the server log gets one structured sampling line:

```text
[strata] video trace=<id> stage=sampling data={"mode":"hybrid",...}
```

The `data` object has `mode`, `fit`, `fps`, `effective_fps`, `duration`,
`source_frames`, `pair_count`, `frame_count`, `pairs`, `pair_midpoints`,
`scene_scan`, `scene_fallback`, `scene_cut_count`, `scene_cuts`,
`scene_cut_scores`, `scene_cuts_truncated`, `shot_count`, `shots`,
`shots_truncated`, `keyframe_count` and `rows_per_pair`. Pair and cut times are
seconds from the first source timestamp.

`scene_scan` is `"skipped"` when `keyframes` is `"uniform"`, `"fallback"` when
the scene filter failed, and `"ok"` after a successful scan. A zero
`scene_cut_count` therefore has no meaning by itself; read `scene_scan` first.
`scene_cuts` are raw cut candidates and `scene_cut_scores` is the matching
score list. `shots` merges candidates closer than `scene_min_gap_s`, keeps the
strongest score in each burst, and reports `start`, `end`, `duration` and the
cut score that opened the shot. `shot_count` is the number of merged shots,
not the raw candidate count.

`scene_cuts`, `scene_cut_scores`, and `shots` are each capped at 256 entries.
The count and corresponding truncation flag describe the full result.
Diagnostics never change the selected frames or the cache key.

The Windows acceptance driver keeps the same object under
`variants[].sampling` in `report.json`, and prints a short line in its summary:

```text
sampling/diagnostics: mode=hybrid fit=fixed fps=2.0 effective_fps=2.0 pairs=7 frames=14 cuts=2 scan=ok shots=3 fallback=False first=[0.0, 0.033333] last=[5.933333, 5.966667]
```

Use `variants[].sampling.pairs` for every selected frame pair,
`variants[].sampling.scene_cuts` for raw candidates and
`variants[].sampling.shots` for the merged timeline. Sampling metadata is
stored with the video cache entry even when diagnostics are off. With
diagnostics on, a cache hit can emit the same `stage=sampling` line without
running FFmpeg again.

### Opt-in shot context

A `/v1/responses` request may ask the server to put the merged shot table in
the user message:

```json
{
  "video_analysis": {
    "mode": "shots",
    "max_shots": 64
  }
}
```

The table is inserted next to the video part after sampling, so the model sees
the FFmpeg boundaries and does not have to infer them from the sampled frames.
`max_shots` is a whole number from 1 to 256; its default is 64. The option is
off by default. It only has an effect when `keyframes` is `hybrid` and the
scene scan succeeded; otherwise the request continues without a shot table.
The server reserves context space for the table before preparing the video, so
enabling this option can reduce the video pair budget.

## Failure handling

- A scene pass that fails for a reason other than cancellation or timeout
  falls back to uniform sampling. The trace records `scene=fallback`.
- A missing I-frame list is not an error; the cut candidates and uniform
  candidates still work.
- A selected frame count or token count above the budget is a `400`-class
  video error, with the existing "reduce duration or server FPS" guidance.
- The worker deadline covers source download, probe, scene pass, sampling and
  decode. A long video should raise `deadline_s` together with
  `max_duration_s`.

## Validation

The video path has no historical unit tests. Validate a change in this order:

1. `keyframes: "uniform"` on the same source before and after the change:
   packet bytes, frame count, timestamps and layout must match.
2. `keyframes: "hybrid"` on a clip with hard cuts: the selected timestamps must
   include the cuts and remain strictly increasing.
3. A clip with no cuts: hybrid must fall back to uniform coverage.
4. An old FFmpeg without `scdet`, if available: hybrid must fall back, not fail.
5. A video longer than the default 120 s with `fit: "budget"`: it must prepare
   within `max_frames` and `max_tokens`.
6. Run `python -m pytest -q serve/test_server.py serve/test_images.py` and
   `python -m py_compile serve/video.py serve/server.py`.

## Changes in this branch

- Frame selection is pair-based. `keyframes: "uniform"` produced byte-identical
  `SVF2` packets before and after the refactor on a six-second, three-scene
  clip.
- The FFmpeg `select` expression is a balanced OR tree. The previous flat
  expression failed near 100 frame terms; on a synthetic 120-second, 2 fps,
  320x180 clip the old worker failed with `video codec reported a decoding
  error`, while the new worker prepared 240 frames (120 pairs).
- `keyframes: "hybrid"` adds scene-cut and I-frame pairs. On the three-scene
  clip it selected pairs around the 2.0 s and 4.0 s cuts; on a clip with no
  cuts it kept uniform coverage.
- Hybrid reserves the first and last endpoint pairs. An odd `max_frames` can
  leave one frame slot unused, so a two-frame pair never exceeds the frame
  budget.
- `fit: "budget"` prepared a synthetic 600-second, 2 fps clip as 255 frames
  (128 pairs) spanning 599.5 s, within `max_frames=256` and
  `max_tokens=32768`.
- The request's actual context room now caps the video token budget before
  preparation, with 1,024 tokens left for timestamp text and markers.

## Next phases

- A coarse-to-fine pass for hour-scale video: a low-FPS scene overview, then a
  higher-FPS look at a requested time range.
- Multiple videos and mixed image/video requests. The chat template already
  supports them; the Responses handler still refuses them.
- ASR and OCR sidecars. Qwen3.8-Flash-Next has no audio token; spoken content
  must arrive as text with timestamps.
