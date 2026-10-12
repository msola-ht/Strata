# Video files and repeated questions

This opt-in Responses extension uploads a finite video once and reuses its visual
embeddings for later questions. Responses remain stateless: send the relevant
text history and one video reference in every request. No answers, conversation
database, `previous_response_id`, multiple videos or appended video segments are
implemented. This is a visual preprocessing cache, not a KV/prefill cache; every
question still runs through prompt construction and model inference.

## Enable and storage lifecycle

Keep the existing `vision` model, projector, executable and video settings. Add
this top-level setting to the server configuration, alongside an API key:

```json
{
  "api_key": ["replace-with-a-secret-for-client-a", "replace-with-a-secret-for-client-b"],
  "video_files": {"enabled": true}
}
```

`vision.video.enabled` must also be true. Startup refuses file support without
an API key or enabled video policy. The model and SVF2 vision encoder must be
loaded before analysis. File upload itself does not load the model. Disabling or
omitting `video_files.enabled` leaves URL video preparation uncached. The existing
single-video, user-message-only, no-image-mixing and no-engine-batching restrictions
still apply. No generation or reasoning budget is added by this feature.

The store creates a private `strata-video-*` directory under the OS temporary
directory, or under the existing parent supplied in `video_files.directory`.
Only bytes and embedding artifacts live on disk; IDs, access scope and layouts
are held in memory. There is no persistent index or startup recovery. A server
restart invalidates all old IDs. Normal server close removes its own files after
active users release them. A process crash can leave an orphan temporary directory;
startup never scans, imports or deletes old directories. An administrator may
remove a confirmed orphan after confirming its process has exited. This is not
automatic migration or cleanup of pre-existing user data.

Normal command-line shutdown closes the store before stopping its producers and
removing the encoder work directory. A temporary deletion failure retains the
disk charge and cleanup retry; the cleanup thread stops only after the directory
has actually been removed. A process exit or crash can still interrupt retries.

Rollback: disable `video_files`, stop the instance through the normal deployment
procedure, and return callers to `video_url`. Old references are not portable
between instances. Disabling the feature does not scan old directories. This
change does not itself deploy or restart a server.

All settings below belong to `video_files`:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `max_bytes` | 2147483648 | Total store disk reservations, including uploads and cache |
| `max_files` | 32 | Uploaded files, including uploads in progress and pinned retired files |
| `ttl_s` | 3600 | File lifetime from start of upload; reads do not extend it |
| `cache_max_bytes` | 1073741824 | Cache disk reservations within `max_bytes` |
| `cache_max_entries` | 16 | Cache entries and pending computations |
| `cache_max_entry_bytes` | 671098880 | Maximum one embedding artifact; reserved before computation |
| `cache_ttl_s` | 3600 | Artifact lifetime from publication |
| `cache_max_workers` | 1 | Concurrent producer admission; allowed values 1 or 2 |
| `directory` | OS temporary directory | Existing parent for the private store directory |

Byte totals are capped at 8 GiB, counts at 512, entry size at 1 GiB and TTLs at
30 days. The entry limit cannot exceed either total. The source limit remains
`vision.video.max_source_bytes` (default 64 MiB). Upload reserves Content-Length
before reading. An upload has a 60-second total read deadline and uses 64 KiB
chunks. Metadata is bounded by the file/entry limits; each cached layout is at
most 64 KiB. No complete video or embeddings are retained in a Python memory cache.

Expired objects are rejected immediately. Cleanup also runs at intervals of at
most `min(60, ttl_s, cache_ttl_s)` seconds. Under capacity pressure, unused cache
entries are evicted by least recent use; live uploaded files are not silently
evicted. Pinned objects retain their disk charge until removal succeeds. Deleted
and expired IDs have bounded tombstones (at most `4 * max_files`); after those are
discarded they return `video_file_not_found`.

Decoder scratch and the generation request's embedding copy are **outside** the
store quota, under the existing vision work directory and video limits described
in [DETAILS.md](DETAILS.md). At most one file-cache decoder and one original URL
request can prepare there concurrently; encoding uses the existing inference
FIFO. Each active generation has its own embedding copy. Provision free space
for these in addition to the store quota. Old directories left by crashes are
also outside the new instance's quota.

## MCP upload, query and delete contract

This endpoint accepts **raw video bytes**, not OpenAI SDK multipart uploads.
It requires one Content-Length, no Transfer-Encoding, and Content-Type `video/*`
or `application/octet-stream`. `curl --data-binary @file` supplies the length.
Use the same API key for upload and all subsequent uses of that ID. Bearer and
`x-api-key` authentication follow the existing server rules. Two different keys
are different owners, even on the same server.
The UTF-8 and Latin-1 header encodings already accepted for the same configured
API key resolve to the same file owner.
If one header matches multiple distinct configured keys through those encodings,
file access returns 401 rather than merging their access scopes. Use keys whose
accepted wire encodings do not overlap.

```sh
curl "$STRATA_URL/v1/files" \
  -H "Authorization: Bearer $STRATA_KEY" \
  -H 'Content-Type: video/mp4' --data-binary @clip.mp4
```

Successful upload returns HTTP 200, for example:

```json
{
  "id": "video_<48 random hexadecimal characters>",
  "object": "video_file",
  "bytes": 1684,
  "created_at": 1791782142,
  "expires_at": 1791785742,
  "status": "ready"
}
```

Use `id` as the Responses `file_id`. Times are Unix seconds. `ready` means the
complete upload was stored, not that it has been decoded or analyzed. Upload is
synchronous; partial uploads never return a usable ID. No filename, local path,
owner hash or content hash is returned as a reading credential.

```sh
curl "$STRATA_URL/v1/files/$FILE_ID" -H "Authorization: Bearer $STRATA_KEY"
curl -X DELETE "$STRATA_URL/v1/files/$FILE_ID" -H "Authorization: Bearer $STRATA_KEY"
```

GET returns the same metadata. DELETE returns HTTP 200 with
`{"id":"video_...","object":"video_file","deleted":true}`. A repeated deletion
returns the corresponding 410 while its tombstone exists. Delete revokes new
uses immediately; already admitted users of the bytes can finish. Physical
removal waits for their leases. Content-derived cache entries can remain until
TTL or eviction because another authorized upload may have the same content;
a deleted ID cannot read them.

## First question and follow-up

POST this JSON to `/v1/responses` with the same Authorization header:

```json
{
  "store": false,
  "reasoning": {"effort": "high"},
  "input": [{"role": "user", "content": [
    {"type": "input_video", "file_id": "video_..."},
    {"type": "input_text", "text": "Describe the objects and their colors, with timestamps."}
  ]}]
}
```

The follow-up sends the same ID, relevant history, and the new question. It does
not resend the video bytes:

```json
{
  "store": false,
  "reasoning": {"effort": "high"},
  "input": [
    {"role": "user", "content": [
      {"type": "input_video", "file_id": "video_..."},
      {"type": "input_text", "text": "Describe the objects and their colors, with timestamps."}
    ]},
    {"role": "assistant", "content": [
      {"type": "output_text", "text": "The object looked blue."}
    ]},
    {"role": "user", "content": [
      {"type": "input_text", "text": "Recheck the video: was the color actually red? Treat the earlier answer as an unverified claim."}
    ]}
  ]
}
```

Exactly one video part is allowed across the whole request. `file_id` and
`video_url` are strictly mutually exclusive, including null or empty alternative
fields. A short discussion summary can be ordinary user text instead of full
history; there is no special `context` field. Assistant history remains assistant
content and is not converted to a system instruction or verified visual fact.
The service never chooses a recent video implicitly.

The original video marker stays in its original message position. It expands
to time labels and image markers for each encoded frame pair, followed by the
remaining messages in their original roles and order. The server checks the
expanded prompt against the current context window and configured output-budget
rules on every call, including cache hits. Do not send a new hard-coded
`max_output_tokens` or reasoning budget merely to use this extension.

## Cache, errors and cancellation

The cache key includes content SHA-256, all video policy settings, encoder
startup arguments (model, projector and visual token settings), a resident
encoder identity renewed on every restart, and a format version. Changes cause
a miss. A hit verifies artifact size, SHA-256, SVE1 records and layout limits,
then makes a request-owned copy. `os.replace` and `drop_embeddings` act on that
copy only. No cached prompt or answer is reused. Existing `video_url` calls remain
uncached and keep downloading/preparing their source.

Same-key concurrent requests share preprocessing. Each waiter has its own
preparation deadline. Disconnecting one waiter removes its interest; the last
waiter leaving cancels the producer. A producer owns its source pin independently
of the first waiter. Failed, cancelled and partial results are not published.
Once generation starts, the existing generation cancellation and engine stall
timeout apply; the video preparation deadline is not a generation time limit.
Disconnect during an encoder exchange can stop the resident encoder under the
existing video cancellation behavior; it must be loaded again before video is
accepted. There is no stored-response cancel endpoint.

The producer has a separate deadline starting when the shared task is created,
covering worker admission, decoding, encoding, artifact copying and checksum
calculation through publication. A later waiter does not extend that deadline.
Request fields, template markers and text-only context capacity are checked
before acquiring or computing the cached video. Expanded video context capacity
is still checked after the layout is available.

Errors before SSE use the usual `{"error":{"message":...,"type":...,"param":...,"code":...}}`
body. After SSE starts they use `response.failed` with the same error code.

| Code | HTTP before SSE | Caller action |
| --- | ---: | --- |
| `video_files_disabled` | 404 on files; 400 on Responses | Enable feature or use video_url |
| `video_file_not_found` | 404 | Unknown, old-instance or different-owner ID; upload with the correct key |
| `video_file_expired`, `video_file_deleted` | 410 | Upload again if analysis is still wanted |
| `video_file_not_ready` | 409 | Not consumable yet; query later (normal synchronous upload returns only ready) |
| `video_file_too_large` | 413 | Reduce source size or change server policy |
| `invalid_request` | 400 or 415 | Fix raw upload length/content type; incomplete uploads are discarded |
| `video_upload_timeout` | 408 | Retry upload; no partial ID is usable |
| `video_store_full` | 507 | Release files or wait for expiry; do not keep uploading duplicates |
| `video_cache_full` | 503 | Retry the same ID when active pins/reservations clear |
| `video_cache_entry_too_large` | 413 | Adjust cache capacity/policy, not the question |
| `video_cache_corrupt` | 503 | Retry the same ID; the corrupt cached entry is retired |
| `video_store_unavailable`, `video_store_closed` | 503 | Storage unavailable; retry after recovery |
| `video_queue_timeout` | 408 | Preparation wait expired; retry same valid ID |
| `video_preparation_timeout` | 408 | Preparing the current prompt/artifact copy exceeded its deadline |
| `video_decode_failed`, `video_sampling_failed`, `video_encode_failed` | 400 | Invalid clip, policy or encoding failure; inspect stage/message |
| `video_{decode,sampling,encode,queue}_timeout` | 400 from producer | Shared preparation deadline expired |
| `video_preparation_cancelled` | 400 or 499 if still connected | Work cancelled; no incomplete artifact is reused |
| `video_preparation_failed` | 400, 502 or 503 | Preparation failure; valid ID can be retried |
| `context_length_exceeded` | 400 | Reduce text history/output demand or change context capacity |
| `video_unavailable` | 400 or 503 | Load a compatible video-capable model/encoder |
| `video_generation_timeout`, `video_generation_failed` | 503 or 500 | Model generation failed; the file can remain usable |

Authentication failures remain 401. Different owners receive not-found even if
an ID exists; content hashes cannot grant access. Cache eviction is not file
expiry: it causes preprocessing on the next question, not re-upload.

Server logs include `video stage=upload bytes=... elapsed_s=...`, preprocessing
queue/decode-and-sampling/encoding durations, and `video cache=miss|shared|hit
wait_s=...`. They contain no video contents, API keys, URLs or conversation text.
These logs can confirm skipped preprocessing; they do not prove answer accuracy.
The separately enabled existing API monitor retains its usual request logging
behavior.

## Observed scope for this change

Manual observations on the Linux development environment used a temporary
loopback HTTP server with the repository's mock text engine **and no vision
model**. Upload/query/delete returned 200; missing authentication returned 401,
cross-key lookup 404, deleted lookup 410, mutually exclusive video fields 400,
multipart upload 415 and a source exceeding 64 MiB 413. A Responses request with
a valid file reached the existing `video_unavailable` guard; it did not generate
a visual answer.

A separate 2-second file TTL returned 410 after expiry, with no files left in
the store directory. A 4097-byte declared upload against a 4096-byte store limit
returned 507 before reading the body. Closing the temporary HTTP server removed
its private store directory. These observations used the actual file handlers;
they did not exercise native inference.

Actual FFmpeg preparation of a generated 1-second 64x64 red MP4 (1684 bytes)
produced a 36,904-byte SVF2 packet with three selected frames. Internal file input
and the existing base64 data-URL input produced identical packet bytes. A
`max_frames: 1` policy rejected it at the sampling stage; its temporary directory
was removed. Rendering user/video, assistant/incorrect color and user/correction
request preserved that sequence and one video marker in the repository template.

No real vision encoder or text model was available for this work. First encode,
cache-hit inference, correction of the wrong color claim, concurrent shared
encoding and cancellation, eviction during inference, and cached context
overflow still require observation on the deployment's actual model and template.
Static compilation is a separate check, not evidence of those behaviors. No
frozen test files were added, changed or used as delivery evidence. CUDA/HIP/SYCL
native sources were not changed or built for this Python-only extension.

A follow-up chain review manually observed these additional behaviors through
the file HTTP handlers: an upload authenticated with a Latin-1 spelling of a
configured non-ASCII key was readable with its UTF-8 spelling (200), while a
different configured key received 404. A 5000-digit Content-Length returned a
structured 413 `video_file_too_large` rather than an integer-conversion exception.
After deleting an uploaded file, a manually held independent producer lease
retained its 1684 bytes after the original lease was released. Closing the
temporary HTTP server preserved that pin and kept cleanup active; releasing the
last pin removed the directory and stopped cleanup. These are file-lifecycle
observations, not observations of shared native encoding. Producer deadline,
early context refusal and transient filesystem-deletion failure still require
runtime observation with the appropriate model or operating-system condition.

The pre-commit review also exercised overlapping accepted encodings for two
different configured keys. The ambiguous header returned 401, the other owner's
unambiguous header received 404 for the uploaded ID, and its original owner
still received 200. The temporary server removed its store on close.
