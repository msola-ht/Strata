// tools/vision/strata_vision.cpp - the image half of Strata's multimodal path.
//
// Turns an image into the embeddings the text model reads in place of its <|image_pad|> tokens, with llama.cpp's
// mtmd library and the model's mmproj file (vision encoder + projector).  The engine does the rest: it places the
// rows at the image's pad tokens and gives them their 2-D M-RoPE positions (see --serve GENI in generate.cpp).
//
//   strata-vision --mmproj <mmproj.gguf> --model <text model .gguf, first split> [--gpu] [--threads N]
//                 [--max-tokens N] [--min-tokens N] [--flash-attn on|off|auto]
// Flash attention defaults to auto, and to off on the CPU when ggml is built with AVX-512 (see main).
//
// Resident: prints "READY <n_embd> video=SVF2", then per stdin line
//   ENC <image path> <output path>   ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//   ENCV2 <max_total_tokens> <packet path> <output path>
//                                  ->  "OK <n_tokens> <n_pairs> <n_frames> <ms>" then "LAYOUT <layout>"
//   QUIT
// The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd],
// row i at grid position (x = i % nx, y = i / nx).  The text model is opened vocab-only (no weights).
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "gguf_meta.h"
#include "gpu_arch.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#elif defined(STRATA_VISION_HIP)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// the GPU architectures ggml-hip printed while it started ("  Device 0: ..., gfx1201:sramecc-:xnack- (0x1201), ...")
std::vector<std::string> g_gfx;

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_INFO && std::strstr(text, "Device ") != nullptr) {
        const std::string gfx = strata_gfx_in(text);
        if (!gfx.empty()) g_gfx.push_back(gfx);
    }
    if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

// "ENC <image> <out>": paths may contain spaces, so the image path ends at the last " " before the output path;
// the server always writes paths without spaces, but a user calling the tool by hand might not
bool parse_enc(const std::string& line, std::string& img, std::string& out) {
    if (line.rfind("ENC ", 0) != 0) return false;
    const std::string rest = line.substr(4);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    img = rest.substr(0, sp);
    out = rest.substr(sp + 1);
    return !img.empty() && !out.empty();
}

// SVF2 is prepared by the server from decoded frames and their source timestamps. All integers and doubles are
// little endian: magic, uint32 width/height/frame count, then (float64 seconds, RGB24 pixels) for each frame.
// Only one pair is held at a time. Consecutive mergeable bitmaps form one temporal patch; the final odd frame is
// copied. LAYOUT carries S<seconds>;I<rows> for each pair. Text is tokenized by the serving tokenizer, not here.
constexpr uint64_t kVideoPacketBytes = 256ULL << 20;
constexpr int kVideoMaxTokens = 65536;
constexpr int kVideoMaxFrameTokens = 1024;
constexpr int kVideoMaxFrames = 1024;
constexpr int kVideoEmbeddingWidth = 2560;
constexpr uint64_t kVideoOutputBytes = uint64_t(kVideoMaxTokens) * kVideoEmbeddingWidth * sizeof(float) +
                                        uint64_t(kVideoMaxFrames / 2) * 5 * sizeof(int32_t);

struct Encv2Args {
    int max_total_tokens = 0;
    std::string packet, out;
};

bool parse_encv2(const std::string& line, Encv2Args& a) {
    if (line.rfind("ENCV2 ", 0) != 0 || line.find('\0') != std::string::npos) return false;
    std::istringstream is(line.substr(6));
    if (!(is >> a.max_total_tokens) || a.max_total_tokens <= 0 || a.max_total_tokens > kVideoMaxTokens)
        return false;
    if (is.peek() != ' ' && is.peek() != '\t') return false;
    std::string rest;
    std::getline(is, rest);
    const size_t first = rest.find_first_not_of(" \t");
    if (first == std::string::npos) return false;
    return parse_enc("ENC " + rest.substr(first), a.packet, a.out);
}

using FilePtr = std::unique_ptr<FILE, int (*)(FILE*)>;
using BitmapPtr = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;
using ChunksPtr = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;

struct VideoOutput {
    std::string path;
    FILE* file = nullptr;
    bool committed = false;
    explicit VideoOutput(const std::string& p) : path(p) {}
    ~VideoOutput() {
        if (file) std::fclose(file);
        if (!committed && created) std::remove(path.c_str());
    }
    bool created = false;
    bool open() {
        // Exclusive creation also rejects a packet/output alias, preserving both existing files and input data.
        file = std::fopen(path.c_str(), "wbx");
        created = file != nullptr;
        return created;
    }
    bool finish() {
        FILE* closing = file;
        file = nullptr;
        if (std::fclose(closing) != 0) return false;
        committed = true;
        return true;
    }
};

uint32_t read_le_u32(const unsigned char* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

bool read_timestamp(FILE* f, double& value) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "SVF2 needs IEEE float64");
    unsigned char bytes[8];
    if (std::fread(bytes, 1, sizeof bytes, f) != sizeof bytes) return false;
    uint64_t bits = 0;
    for (size_t i = 0; i < sizeof bytes; ++i) bits |= uint64_t(bytes[i]) << (8 * i);
    std::memcpy(&value, &bits, sizeof value);
    return true;
}

// A pair must map exactly onto the engine's existing 2-D SVE1 positions. Inspect every row before encoding or
// writing, rather than accepting a rectangular last position while earlier rows use temporal or other positions.
std::string write_video_record(mtmd_context* ctx, const mtmd_input_chunk* ch, int n_embd, FILE* f,
                               int row_room, uint64_t byte_room, int& n) {
    const size_t rows = mtmd_input_chunk_get_n_tokens(ch);
    if (rows == 0 || rows > size_t(kVideoMaxFrameTokens) || rows > size_t(row_room))
        return "the video exceeds its image token budget";
    const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(ch);
    if (!it) return "the video has no image tokens";
    const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, rows - 1);
    if (last.x >= kVideoMaxFrameTokens || last.y >= kVideoMaxFrameTokens)
        return "the video has invalid image positions";
    const int nx = int(last.x) + 1, ny = int(last.y) + 1;
    if (uint64_t(nx) * ny != rows) return "the video image grid is not rectangular";
    if (mtmd_input_chunk_get_n_pos(ch) != std::max(nx, ny))
        return "the video position span cannot be represented by SVE1";
    for (size_t i = 0; i < rows; ++i) {
        const mtmd_decoder_pos pos = mtmd_image_tokens_get_decoder_pos(it, 0, i);
        if (pos.t != 0 || pos.z != 0 || pos.x != i % nx || pos.y != i / nx)
            return "the video image positions cannot be represented by SVE1";
    }
    const uint64_t bytes = 5 * sizeof(int32_t) + uint64_t(rows) * n_embd * sizeof(float);
    if (bytes > byte_room) return "the video exceeds its output byte budget";
    if (mtmd_encode_chunk(ctx, ch) != 0) return "the vision encoder failed";
    const float* embd = mtmd_get_output_embd(ctx);
    if (!embd) return "the vision encoder returned no embeddings";
    n = int(rows);
    const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
    if (std::fwrite(hdr, sizeof hdr, 1, f) != 1 ||
        std::fwrite(embd, sizeof(float) * size_t(n_embd), rows, f) != rows)
        return "short write to the video output";
    return "";
}

std::string encode_video_packet(mtmd_context* ctx, int n_embd, const Encv2Args& a,
                                int& total, int& pairs, int& frame_count, std::string& layout) {
    if (n_embd != kVideoEmbeddingWidth) return "the video projector must have 2560 embedding columns";
    FilePtr input(std::fopen(a.packet.c_str(), "rb"), &std::fclose);
    if (!input) return "cannot read the video packet";
    if (std::fseek(input.get(), 0, SEEK_END) != 0) return "cannot measure the video packet";
    const long file_bytes = std::ftell(input.get());
    if (file_bytes < 16 || uint64_t(file_bytes) > kVideoPacketBytes)
        return "the video packet size is outside its limit";
    if (std::fseek(input.get(), 0, SEEK_SET) != 0) return "cannot rewind the video packet";
    unsigned char header[16];
    if (std::fread(header, 1, sizeof header, input.get()) != sizeof header ||
        std::memcmp(header, "SVF2", 4) != 0) return "invalid SVF2 video packet header";
    const uint32_t width = read_le_u32(header + 4), height = read_le_u32(header + 8);
    const uint32_t frames = read_le_u32(header + 12);
    if (width == 0 || height == 0 || width % 32 != 0 || height % 32 != 0 ||
        uint64_t(width) * height / 1024 > kVideoMaxFrameTokens || frames == 0 || frames > kVideoMaxFrames)
        return "invalid SVF2 dimensions or frame count";
    const uint64_t rgb_bytes = uint64_t(width) * height * 3;
    const uint64_t expected_bytes = sizeof header + uint64_t(frames) * (8 + rgb_bytes);
    if (expected_bytes != uint64_t(file_bytes)) return "the SVF2 video packet length does not match its header";
    const int expected_pairs = int((frames + 1) / 2);
    const uint64_t predicted_rows = uint64_t(expected_pairs) * (uint64_t(width) * height / 1024);
    if (predicted_rows > uint64_t(a.max_total_tokens)) return "the video exceeds its image token budget";
    std::vector<double> timestamps(frames);
    for (uint32_t i = 0; i < frames; ++i) {
        if (!read_timestamp(input.get(), timestamps[i]) || !std::isfinite(timestamps[i]) || timestamps[i] < 0 ||
            (i != 0 && timestamps[i] <= timestamps[i - 1]))
            return "SVF2 timestamps must be finite, nonnegative and strictly increasing";
        if (std::fseek(input.get(), long(rgb_bytes), SEEK_CUR) != 0) return "cannot read the video packet frames";
    }
    if (std::fseek(input.get(), sizeof header, SEEK_SET) != 0) return "cannot rewind the video packet frames";
    VideoOutput output(a.out);
    if (!output.open()) return "cannot exclusively create the video output";
    std::vector<unsigned char> rgb(size_t(rgb_bytes), 0);
    const std::string marker = mtmd_default_marker();
    uint64_t output_bytes = 0;
    for (uint32_t base = 0; base < frames; base += 2) {
        BitmapPtr first(nullptr, &mtmd_bitmap_free), second(nullptr, &mtmd_bitmap_free);
        const uint32_t count = std::min(uint32_t(2), frames - base);
        for (uint32_t j = 0; j < count; ++j) {
            double timestamp = 0;
            if (!read_timestamp(input.get(), timestamp) || timestamp != timestamps[base + j] ||
                std::fread(rgb.data(), 1, rgb.size(), input.get()) != rgb.size())
                return "the video packet changed or ended while reading frames";
            BitmapPtr bm(mtmd_bitmap_init(width, height, rgb.data()), &mtmd_bitmap_free);
            if (!bm) return "cannot allocate a video frame";
            mtmd_bitmap_set_mergeable(bm.get(), true);
            if (j == 0) first = std::move(bm);
            else second = std::move(bm);
        }
        if (count == 1) {
            second.reset(mtmd_bitmap_init(width, height, rgb.data()));
            if (!second) return "cannot duplicate the final video frame";
            mtmd_bitmap_set_mergeable(second.get(), true);
        }
        const double pair_time = timestamps[base] + (timestamps[base + count - 1] - timestamps[base]) / 2;
        std::ostringstream label;
        label.imbue(std::locale::classic());
        label << std::setprecision(17) << pair_time;
        const std::string prompt = '<' + label.str() + " seconds>" + marker + marker;
        const mtmd_bitmap* bitmaps[2] = {first.get(), second.get()};
        ChunksPtr chunks(mtmd_input_chunks_init(), &mtmd_input_chunks_free);
        if (!chunks) return "cannot allocate video chunks";
        mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
        if (mtmd_tokenize(ctx, chunks.get(), &txt, bitmaps, 2) != 0)
            return "the video pair could not be preprocessed";
        size_t image_chunks = 0;
        for (size_t c = 0; c < mtmd_input_chunks_size(chunks.get()); ++c) {
            const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks.get(), c);
            const auto type = mtmd_input_chunk_get_type(ch);
            if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) ++image_chunks;
            else if (type != MTMD_INPUT_CHUNK_TYPE_TEXT) return "the video gave an unsupported chunk type";
        }
        if (image_chunks != 1) return "the video pair did not merge into exactly one image chunk";
        for (size_t c = 0; c < mtmd_input_chunks_size(chunks.get()); ++c) {
            const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks.get(), c);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                int n = 0;
                const std::string err = write_video_record(ctx, ch, n_embd, output.file,
                                                           a.max_total_tokens - total,
                                                           kVideoOutputBytes - output_bytes, n);
                if (!err.empty()) return err;
                if (!layout.empty()) layout += ';';
                layout += 'S' + label.str() + ";I" + std::to_string(n);
                total += n;
                output_bytes += 5 * sizeof(int32_t) + uint64_t(n) * n_embd * sizeof(float);
                ++pairs;
            }
        }
    }
    if (std::fgetc(input.get()) != EOF || std::ferror(input.get())) return "the video packet changed after validation";
    if (pairs != expected_pairs || total <= 0 || total > a.max_total_tokens)
        return "the video output does not match its frame count or token budget";
    if (!output.finish()) return "cannot finish writing the video output";
    frame_count = int(frames);
    return "";
}

bool video_projector_profile(const gguf_context* gg, int n_embd) {
    if (!gg || n_embd != kVideoEmbeddingWidth) return false;
    // Match mtmd's explicit projector metadata, including its mixed-modality fallback. A plain 2560-column
    // projector does not establish temporal merging. Deepstack output has additional columns SVE1 cannot carry.
    int64_t type_key = gguf_find_key(gg, "clip.projector_type");
    const char* type = nullptr;
    if (type_key >= 0 && gguf_get_kv_type(gg, type_key) == GGUF_TYPE_STRING)
        type = gguf_get_val_str(gg, type_key);
    if (!type || !*type) {
        type_key = gguf_find_key(gg, "clip.vision.projector_type");
        if (type_key >= 0 && gguf_get_kv_type(gg, type_key) == GGUF_TYPE_STRING)
            type = gguf_get_val_str(gg, type_key);
    }
    if (!type || std::strcmp(type, "qwen3vl_merger") != 0) return false;
    const int64_t patch = gguf_find_key(gg, "clip.vision.patch_size");
    if (patch < 0 || gguf_get_kv_type(gg, patch) != GGUF_TYPE_UINT32 || gguf_get_val_u32(gg, patch) != 16)
        return false;
    for (int64_t i = 0; i < gguf_get_n_tensors(gg); ++i) {
        const char* name = gguf_get_tensor_name(gg, i);
        if (name && std::strncmp(name, "v.deepstack.", 12) == 0) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32) && defined(STRATA_VISION_HIP)
    // hipblas.dll is delay-loaded (tools/vision/CMakeLists.txt): it and the ROCm DLLs behind it (rocblas.dll finds its kernel
    // library and ../.kpack beside itself) stay in rocm\bin of the engine package, which is not on the DLL search path of a
    // program started with nothing but System32 in PATH.  Put that folder on it before the first call.
    {
        wchar_t exe[MAX_PATH];
        const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir(exe, n);
        const size_t cut = dir.find_last_of(L"\\/");
        if (cut != std::wstring::npos) {
            dir.resize(cut);
            const std::wstring rocm = dir + L"\\rocm\\bin";
            if (GetFileAttributesW(rocm.c_str()) != INVALID_FILE_ATTRIBUTES) SetDllDirectoryW(rocm.c_str());
        }
    }
#endif
    std::string mmproj, model;
    bool gpu = false;
    int threads = 0, max_tokens = 0, min_tokens = 0;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    bool fa_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--mmproj") mmproj = next();
        else if (a == "--model") model = next();
        else if (a == "--gpu") gpu = true;
        else if (a == "--version") {
#ifdef STRATA_VISION_HIP
            std::printf("strata-vision hip (GPU architectures: %s)\n", STRATA_VISION_HIP_ARCHS);
#elif defined(STRATA_VISION_METAL)
            std::printf("strata-vision metal\n");
#elif defined(STRATA_VISION_SYCL)
            std::printf("strata-vision sycl (Intel GPU)\n");
#else
            std::printf("strata-vision\n");
#endif
            return 0;
        }
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--max-tokens") max_tokens = std::atoi(next().c_str());
        else if (a == "--min-tokens") min_tokens = std::atoi(next().c_str());   // mtmd image_min_tokens (#767)
        else if (a == "--flash-attn") {   // FA keeps K and V in FP16; off = the attention in FP32
            const std::string v = next();
            fa = v == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : v == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                                                                        : LLAMA_FLASH_ATTN_TYPE_AUTO;
            fa_given = true;
        }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (mmproj.empty() || model.empty()) {
        std::fprintf(stderr, "usage: strata-vision --mmproj <mmproj.gguf> --model <model.gguf> [--gpu] [--threads N] "
                             "[--max-tokens N] [--min-tokens N] [--flash-attn on|off|auto]\n");
        return 2;
    }
    // On the CPU the GPU stays unseen: a CUDA build otherwise opens a context there (measured: 0.4-0.7 GB of VRAM,
    // 150-260 expert slots less for the engine beside it).  Before anything reaches the CUDA runtime.
    if (!gpu) {
#ifdef _WIN32
        _putenv_s("CUDA_VISIBLE_DEVICES", "-1");
        _putenv_s("HIP_VISIBLE_DEVICES", "-1");
#else
        setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
        setenv("HIP_VISIBLE_DEVICES", "-1", 1);
#endif
    }
    llama_log_set(quiet_log, nullptr);
    mtmd_helper_log_set(quiet_log, nullptr);
    llama_backend_init();
#if defined(STRATA_VISION_HIP) || defined(STRATA_VISION_SYCL)
#ifdef STRATA_VISION_SYCL
#define STRATA_VISION_BUILT_FOR "Intel GPUs"
#else
#define STRATA_VISION_BUILT_FOR STRATA_VISION_HIP_ARCHS
#endif
    if (gpu) {
        // The AMD / Intel GPU reader: starting the backends prints the GPU line.  With --gpu and no GPU, mtmd would quietly use the CPU
        // (a second, slower CPU reader inside the HIP program), and a card the build has no code for would abort at its first
        // kernel.  Both end here with a line the server reads: it then starts the CPU reader (serve/server.py, Vision._start).
        bool have_gpu = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
            have_gpu = have_gpu || type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        }
        if (!have_gpu) {
            std::printf("ERR this picture reader found no GPU (it was built for: %s)\n", STRATA_VISION_BUILT_FOR);
            std::fflush(stdout);
            return 3;
        }
#ifdef STRATA_VISION_HIP
        const char* check = std::getenv("STRATA_VISION_ARCH_CHECK");
        if (!g_gfx.empty() && (check == nullptr || std::atoi(check) != 0) &&
            !strata_arch_listed(STRATA_VISION_HIP_ARCHS, g_gfx.front())) {
            std::printf("ERR this GPU (%s) is not in the picture reader's build (%s)\n", g_gfx.front().c_str(),
                        STRATA_VISION_HIP_ARCHS);
            std::fflush(stdout);
            return 3;
        }
#endif
    }
#endif
#ifdef STRATA_VISION_METAL
    if (gpu) {
        // The Mac: the same program reads on the GPU (--gpu, ggml's Metal backend) or on the CPU.  With --gpu and no Metal device (a
        // virtual machine without a GPU, say), mtmd would quietly use the CPU, which reads a picture in ~50 s: say so instead, and
        // the server starts the program again without --gpu (serve/server.py, Vision._start) with one log line.
        bool have_gpu = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
            have_gpu = have_gpu || type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        }
        if (!have_gpu) {
            std::printf("ERR this Mac has no Metal GPU for the picture reader\n");
            std::fflush(stdout);
            return 3;
        }
    }
#endif

    const auto load_t0 = std::chrono::steady_clock::now();
#if !defined(_WIN32) && !defined(__APPLE__)   // macOS has no posix_fadvise
    // ask the OS for the projector file up front (128 KiB steps) so the loader's sequential reads find it cached;
    // STRATA_READ_AHEAD=0 turns it off, as in the engine
    if (const char* ra = std::getenv("STRATA_READ_AHEAD"); ra == nullptr || std::atoi(ra) != 0) {
        const int fd = open(mmproj.c_str(), O_RDONLY);
        if (fd >= 0) {
            const off_t n = lseek(fd, 0, SEEK_END);
            for (off_t at = 0; at < n; at += (off_t) (128 << 10))
                (void) posix_fadvise(fd, at, std::min<off_t>((off_t) (128 << 10), n - at), POSIX_FADV_WILLNEED);
            close(fd);
        }
    }
#endif
    // the vocabulary and header only: the weights (and their tensor types, which can be Strata-private ones such as
    // Gyro's TQK6 / TQK7) are never touched - see gguf_meta.h
    std::string how;
    llama_model* text = strata_open_text_model(model, &how);
    if (!text) { std::printf("ERR cannot open the text model %s (%s)\n", model.c_str(), how.c_str()); std::fflush(stdout); return 1; }

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = gpu;
    cp.print_timings = false;
    cp.warmup = false;
    // On the CPU "auto" turns flash attention on.  ggml's fast (tiled) CPU kernel for it needs the head size, 72 in this
    // encoder, to be a multiple of the vector width: 8 floats with AVX2 (the release builds), 16 with AVX-512 (a build
    // from source on a CPU that has it), where ggml falls back to a kernel that is several times slower and accumulates
    // in FP16.  A 1024x1024 picture on a Ryzen 7 7700X, 8 threads: AVX2 8-10 s with it, 14-15 s without; AVX-512 44 s
    // with it (26% off the FP32 attention's output), 13 s without.
    if (!gpu && !fa_given && ggml_cpu_has_avx512()) fa = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.flash_attn_type = fa;
    // on the CPU without --threads: one per core (mtmd's own default is 4 threads)
    // (at most 16: a 256-thread server does not need 128 encoder threads; --threads overrides)
    if (threads <= 0 && !gpu) threads = std::max(1u, std::min(16u, std::thread::hardware_concurrency() / 2));
    if (threads > 0) cp.n_threads = threads;
    if (max_tokens > 0) cp.image_max_tokens = max_tokens;
    if (min_tokens > 0) cp.image_min_tokens = min_tokens;
    mtmd_context* ctx = mtmd_init_from_file(mmproj.c_str(), text, cp);
    if (!ctx || !mtmd_support_vision(ctx)) {
        std::printf("ERR cannot load the vision encoder %s\n", mmproj.c_str());
        std::fflush(stdout);
        return 1;
    }
    std::fprintf(stderr, "strata-vision: model files loaded in %.1f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - load_t0).count());
    // a vocab-only model reports no hparams, so the width comes from the projector: its output is the model's input
    int n_embd = 0;
    bool video_profile = false;
    {
        gguf_init_params gp{true, nullptr};
        gguf_context* gg = gguf_init_from_file(mmproj.c_str(), gp);
        const int64_t k = gg ? gguf_find_key(gg, "clip.vision.projection_dim") : -1;
        if (k >= 0) n_embd = (int) gguf_get_val_u32(gg, k);
        video_profile = video_projector_profile(gg, n_embd);
        if (gg) gguf_free(gg);
    }
    if (n_embd <= 0) { std::printf("ERR the vision encoder has no projection_dim\n"); std::fflush(stdout); return 1; }
    // Warm up at the LARGEST picture before READY: the encoder's GPU work buffers are allocated now, not at the first
    // real picture.  The server starts this process before the engine, so the engine sizes its expert cache from
    // what is really left; allocating ~1 GB later, on a GPU the engine has filled, made Windows page GPU memory and
    // the engine crawl to a standstill.  (A square image well above any cap; mtmd scales it to the token limit.)
    // On the CPU there is no VRAM to reserve, and the warm-up would only delay the engine's start by one encode
    // (~6 s at 1,024 tokens).
    if (!gpu) std::fprintf(stderr, "strata-vision: on the CPU, %d threads, no warm-up\n", threads);
    else {
        const auto warm_t0 = std::chrono::steady_clock::now();
        const uint32_t side = 2048;
        std::vector<unsigned char> rgb((size_t) side * side * 3, 128);
        mtmd_bitmap* bm = mtmd_bitmap_init(side, side, rgb.data());
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const std::string marker = mtmd_default_marker();
        mtmd_input_text txt{marker.c_str(), marker.size(), false, true};
        const mtmd_bitmap* bms[1] = {bm};
        int warm_tokens = 0;
        if (bm && mtmd_tokenize(ctx, chunks, &txt, bms, 1) == 0) {
            for (size_t c = 0; c < mtmd_input_chunks_size(chunks); ++c) {
                const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
                if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE && mtmd_encode_chunk(ctx, ch) == 0)
                    warm_tokens = (int) mtmd_input_chunk_get_n_tokens(ch);
            }
        }
        std::fprintf(stderr, "strata-vision: warmed up at %d image tokens in %.1f s\n", warm_tokens,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_t0).count());
        mtmd_input_chunks_free(chunks);
        if (bm) mtmd_bitmap_free(bm);
    }
    std::printf("READY %d%s\n", n_embd, video_profile ? " video=SVF2" : "");
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        if (line.rfind("ENCV2", 0) == 0) {
            Encv2Args va;
            if (!parse_encv2(line, va)) {
                std::printf("ERR expected: ENCV2 <max_total_tokens> <packet> <output>\n");
                std::fflush(stdout);
                continue;
            }
            if (!video_profile) {
                std::printf("ERR the projector does not support the SVF2 video profile\n");
                std::fflush(stdout);
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            int total = 0, pairs = 0, frames = 0;
            std::string err, layout;
            try {
                err = encode_video_packet(ctx, n_embd, va, total, pairs, frames, layout);
            } catch (const std::exception& e) {
                err = std::string("cannot encode the video packet: ") + e.what();
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (err.empty()) std::printf("OK %d %d %d %.0f\nLAYOUT %s\n", total, pairs, frames, ms, layout.c_str());
            else std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        std::string img, out;
        if (!parse_enc(line, img, out)) { std::printf("ERR expected: ENC <image> <output>\n"); std::fflush(stdout); continue; }
        const auto t0 = std::chrono::steady_clock::now();
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, img.c_str(), false,
                                                                            mtmd_helper_init_opt_default());
        if (!bw.bitmap) { std::printf("ERR cannot read the image %s\n", img.c_str()); std::fflush(stdout); continue; }
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const std::string prompt = mtmd_default_marker();
        mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
        const mtmd_bitmap* bm[1] = {bw.bitmap};
        std::string err;
        const mtmd_input_chunk* ichunk = nullptr;
        if (mtmd_tokenize(ctx, chunks, &txt, bm, 1) != 0) err = "the image could not be preprocessed";
        for (size_t c = 0; err.empty() && c < mtmd_input_chunks_size(chunks); ++c) {
            const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) ichunk = ch;
        }
        if (err.empty() && !ichunk) err = "no image chunk";
        if (err.empty() && mtmd_encode_chunk(ctx, ichunk) != 0) err = "the vision encoder failed";
        if (err.empty()) {
            const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(ichunk);
            const int n = (int) mtmd_input_chunk_get_n_tokens(ichunk);
            // the grid from the decoder positions (nx/ny getters are deprecated): x and y of the last token
            const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, (size_t) n - 1);
            const int nx = (int) last.x + 1, ny = (int) last.y + 1;
            if (nx * ny != n) err = "the image grid is not rectangular (" + std::to_string(n) + " tokens)";
            const float* embd = mtmd_get_output_embd(ctx);
            FILE* f = err.empty() ? std::fopen(out.c_str(), "wb") : nullptr;
            if (err.empty() && !f) err = "cannot write " + out;
            if (f) {
                const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
                const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                                std::fwrite(embd, sizeof(float) * (size_t) n_embd, (size_t) n, f) == (size_t) n;
                std::fclose(f);
                if (!ok) err = "short write to " + out;
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (ok) std::printf("OK %d %d %d %.0f\n", n, nx, ny, ms);
            }
        }
        if (!err.empty()) std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
        if (bw.video_ctx) mtmd_helper_video_free(bw.video_ctx);
    }
    mtmd_free(ctx);
    llama_model_free(text);
    llama_backend_free();
    return 0;
}
