/*
 * libmcraw C-API shim over upstream motioncam-decoder (C++).
 *
 * The MLV App tree previously vendored a hand-ported C version of the
 * motioncam-decoder. That port has been replaced with the upstream C++
 * sources (Decoder.cpp + Container.hpp + Decoder.hpp + RawData.cpp +
 * RawData_Legacy.cpp + nlohmann::json) and this shim, which preserves the
 * existing C ABI declared in mcraw.h.
 */

#include "Decoder.hpp"
#include "RawData.hpp"
#include "json.hpp"
#include "mcraw.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
    #define MR_FSEEK _fseeki64
#elif defined(__unix__) || defined(__linux__) || defined(__APPLE__)
    #define MR_FSEEK fseeko
#else
    #define MR_FSEEK fseek
#endif

// Compile-time guarantee that mr_buffer_offset_t and motioncam::BufferOffset
// have identical layout so we can blit between them.
static_assert(sizeof(mr_buffer_offset_t) == sizeof(motioncam::BufferOffset),
              "mr_buffer_offset_t / motioncam::BufferOffset layout mismatch");

struct mr_ctx_s {
    FILE*               fd = nullptr;     // Caller-visible handle (mr_get_file_handle)
    std::string         path;

    motioncam::Decoder* dec = nullptr;    // Lives only across mr_decoder_parse

    std::vector<mr_buffer_offset_t> video_offsets;
    std::vector<mr_buffer_offset_t> audio_offsets;

    mr_rational_t       frame_rate = {1, 1};

    double              color_matrix1[9]   = {1,0,0, 0,1,0, 0,0,1};
    double              color_matrix2[9]   = {1,0,0, 0,1,0, 0,0,1};
    double              forward_matrix1[9] = {1,0,0, 0,1,0, 0,0,1};
    double              forward_matrix2[9] = {1,0,0, 0,1,0, 0,0,1};

    int16_t             white_level = 0;
    int16_t             black_level = 0;
    uint32_t            cfa_pattern = 0;

    double              aperture     = 0.0;
    double              focal_length = 0.0;

    int32_t             audio_sample_rate = 0;
    int32_t             audio_channels    = 0;

    char                manufacturer[MR_MAX_STRING + 1] = {0};
    char                model[MR_MAX_STRING + 1]        = {0};

    mr_frame_data_t     frame_data = {};

    int                 verbose = 0;
    int                 last_error = 0;
    char                error_message[256] = {0};
};

namespace {

using nlohmann::json;

// Read a possibly-array-wrapped scalar (some metadata fields are arrays of one value).
template <typename T>
static bool json_get_scalar(const json& parent, const char* key, T& out)
{
    if (!parent.is_object()) return false;
    auto it = parent.find(key);
    if (it == parent.end()) return false;
    const json* v = &it.value();
    if (v->is_array() && !v->empty()) v = &v->at(0);

    if (v->is_number()) {
        try { out = v->get<T>(); return true; } catch (...) { return false; }
    }
    if (v->is_string()) {
        try { out = static_cast<T>(std::stod(v->get<std::string>())); return true; }
        catch (...) { return false; }
    }
    return false;
}

static bool json_get_string(const json& parent, const char* key, std::string& out)
{
    if (!parent.is_object()) return false;
    auto it = parent.find(key);
    if (it == parent.end()) return false;
    const json& v = it.value();
    if (v.is_string()) { out = v.get<std::string>(); return true; }
    if (v.is_array() && !v.empty() && v.at(0).is_string()) {
        out = v.at(0).get<std::string>(); return true;
    }
    return false;
}

static int json_copy_matrix(const json& parent, const char* key, double* dst, int max)
{
    if (!parent.is_object()) return 0;
    auto it = parent.find(key);
    if (it == parent.end() || !it.value().is_array()) return 0;
    const json& arr = it.value();
    int n = std::min<int>(static_cast<int>(arr.size()), max);
    for (int i = 0; i < n; i++) {
        const json& e = arr.at(i);
        if (e.is_number())          dst[i] = e.get<double>();
        else if (e.is_string()) {
            try { dst[i] = std::stod(e.get<std::string>()); }
            catch (...) { dst[i] = 0.0; }
        }
        else dst[i] = 0.0;
    }
    return n;
}

static void copy_string(char* dst, size_t cap, const std::string& src)
{
    size_t n = std::min(src.size(), cap);
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

// rggb -> 0x02010100, gbrg -> 0x01000201, etc.
static uint32_t parse_sensor_arrangement(const std::string& s)
{
    uint32_t pattern = 0;
    for (int i = 0; i < 4 && i < (int)s.size(); i++) {
        char c = s[i];
        uint32_t n = (c == 'g') ? 1 : (c == 'b') ? 2 : 0;
        pattern |= (n << (i * 8));
    }
    return pattern;
}

static void identity3(double* m) {
    std::memset(m, 0, 9 * sizeof(double));
    m[0] = m[4] = m[8] = 1.0;
}

static void cache_container_metadata(mr_ctx_t* ctx, const json& meta)
{
    json_get_scalar(meta, "apertures",   ctx->aperture);
    json_get_scalar(meta, "focalLengths", ctx->focal_length);
    {
        int wl = 0, bl = 0;
        if (json_get_scalar(meta, "whiteLevel", wl)) ctx->white_level = (int16_t)wl;
        if (json_get_scalar(meta, "blackLevel", bl)) ctx->black_level = (int16_t)bl;
    }

    if (meta.is_object() && meta.contains("extraData") && meta.at("extraData").is_object()) {
        const json& ex = meta.at("extraData");
        json_get_scalar(ex, "audioChannels",   ctx->audio_channels);
        json_get_scalar(ex, "audioSampleRate", ctx->audio_sample_rate);
    } else {
        json_get_scalar(meta, "audioChannels",   ctx->audio_channels);
        json_get_scalar(meta, "audioSampleRate", ctx->audio_sample_rate);
    }

    std::string s;
    if (meta.is_object() && meta.contains("deviceSpecificProfile") && meta.at("deviceSpecificProfile").is_object()) {
        if (json_get_string(meta.at("deviceSpecificProfile"), "deviceModel", s)) copy_string(ctx->model, MR_MAX_STRING, s);
    }
    if (ctx->model[0] == '\0' && json_get_string(meta, "deviceModel", s)) {
        copy_string(ctx->model, MR_MAX_STRING, s);
    }

    int cm1 = json_copy_matrix(meta, "colorMatrix1",   ctx->color_matrix1,   9);
    int cm2 = json_copy_matrix(meta, "colorMatrix2",   ctx->color_matrix2,   9);
    int fm1 = json_copy_matrix(meta, "forwardMatrix1", ctx->forward_matrix1, 9);
    int fm2 = json_copy_matrix(meta, "forwardMatrix2", ctx->forward_matrix2, 9);

    if (cm1 < 9 && cm2 >= 9) std::memcpy(ctx->color_matrix1, ctx->color_matrix2, 9*sizeof(double));
    else if (cm1 < 9)        identity3(ctx->color_matrix1);
    if (cm2 < 9 && cm1 >= 9) std::memcpy(ctx->color_matrix2, ctx->color_matrix1, 9*sizeof(double));
    else if (cm2 < 9)        identity3(ctx->color_matrix2);
    if (fm1 < 9 && fm2 >= 9) std::memcpy(ctx->forward_matrix1, ctx->forward_matrix2, 9*sizeof(double));
    else if (fm1 < 9)        identity3(ctx->forward_matrix1);
    if (fm2 < 9 && fm1 >= 9) std::memcpy(ctx->forward_matrix2, ctx->forward_matrix1, 9*sizeof(double));
    else if (fm2 < 9)        identity3(ctx->forward_matrix2);

    if (json_get_string(meta, "sensorArrangement", s) ||
        json_get_string(meta, "sensorArrangment",  s)) {
        ctx->cfa_pattern = parse_sensor_arrangement(s);
    }
}

// Compute frame_rate from sorted timestamps the same way the old port did.
static void compute_frame_rate(mr_ctx_t* ctx)
{
    ctx->frame_rate.den = 1;
    ctx->frame_rate.num = 1;
    if (ctx->video_offsets.size() < 2) return;
    int64_t total = ctx->video_offsets.back().timestamp - ctx->video_offsets.front().timestamp;
    if (total <= 0) return;
    double frame_duration_ms = ((double)total / 1e6) / (double)(ctx->video_offsets.size() - 1);
    if (frame_duration_ms <= 0.0) return;
    double fps = 1000.0 / frame_duration_ms;
    if (fps < 1.5) ctx->frame_rate.num = 1;
    else           ctx->frame_rate.num = (int)std::lround(fps);
}

// Read first frame's metadata via direct file I/O into ctx->frame_data.
static int read_first_frame_metadata(mr_ctx_t* ctx)
{
    if (ctx->video_offsets.empty()) return 0;
    if (MR_FSEEK(ctx->fd, ctx->video_offsets[0].offset, SEEK_SET) != 0) {
        return kMrErrorSeek;
    }
    mr_item_t item = {};
    if (std::fread(&item, sizeof(mr_item_t), 1, ctx->fd) != 1) return kMrErrorRead;
    if (MR_FSEEK(ctx->fd, item.size, SEEK_CUR) != 0) return kMrErrorSeek;
    return mr_read_frame_metadata(ctx->fd, &ctx->frame_data);
}

static int set_error(mr_ctx_t* ctx, int code, const char* fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    std::vsnprintf(ctx->error_message, sizeof(ctx->error_message), fmt, ap);
    va_end(ap);
    if (ctx->verbose) std::fprintf(stderr, "%s\n", ctx->error_message);
    ctx->last_error = code;
    return code;
}

} // namespace

extern "C" {

//-----------------------------------------------------------------------------
mr_ctx_t* mr_decoder_new(int verbose)
{
    auto* c = new (std::nothrow) mr_ctx_s();
    if (c) c->verbose = verbose;
    return c;
}

//-----------------------------------------------------------------------------
void mr_decoder_free(mr_ctx_t* ctx)
{
    if (!ctx) return;
    delete ctx->dec;
    // Do NOT fclose ctx->fd: the caller may continue to use the handle via
    // mr_get_file_handle() after free. mr_decoder_close() is the explicit
    // way to release the file.
    delete ctx;
}

//-----------------------------------------------------------------------------
void mr_decoder_close(mr_ctx_t* ctx)
{
    if (ctx && ctx->fd) {
        std::fclose(ctx->fd);
        ctx->fd = nullptr;
    }
}

//-----------------------------------------------------------------------------
int mr_decoder_open(mr_ctx_t* ctx, const char* filename)
{
    if (!ctx || !filename) return -1;
    ctx->path = filename;
    ctx->fd   = std::fopen(filename, "rb");
    if (!ctx->fd) {
        return set_error(ctx, -1, "Failed to open file: %s", filename);
    }
    return 0;
}

//-----------------------------------------------------------------------------
int mr_decoder_parse(mr_ctx_t* ctx)
{
    if (!ctx || !ctx->fd) return -1;

    try {
        // Upstream Decoder takes ownership of the FILE* and closes it in dtor.
        // Open a separate handle for the upstream parser so our ctx->fd remains
        // alive for the caller.
        FILE* parse_fd = std::fopen(ctx->path.c_str(), "rb");
        if (!parse_fd) {
            return set_error(ctx, -1, "Failed to open parse handle: %s", ctx->path.c_str());
        }

        ctx->dec = new motioncam::Decoder(parse_fd); // takes ownership of parse_fd

        const auto& meta = ctx->dec->getContainerMetadata();
        cache_container_metadata(ctx, meta);

        ctx->audio_sample_rate = ctx->dec->audioSampleRateHz();
        ctx->audio_channels    = ctx->dec->numAudioChannels();

        const auto& vo = ctx->dec->getVideoOffsets();
        ctx->video_offsets.resize(vo.size());
        if (!vo.empty()) {
            std::memcpy(ctx->video_offsets.data(), vo.data(),
                        vo.size() * sizeof(mr_buffer_offset_t));
        }

        const auto& ao = ctx->dec->getAudioOffsets();
        ctx->audio_offsets.resize(ao.size());
        if (!ao.empty()) {
            std::memcpy(ctx->audio_offsets.data(), ao.data(),
                        ao.size() * sizeof(mr_buffer_offset_t));
        }

        compute_frame_rate(ctx);

        // Free the parser early; we keep only ctx->fd from now on.
        delete ctx->dec;
        ctx->dec = nullptr;

        if (read_first_frame_metadata(ctx) != 0) {
            // Non-fatal: just clears frame_data.
        }
    }
    catch (const std::exception& e) {
        if (ctx->dec) { delete ctx->dec; ctx->dec = nullptr; }
        return set_error(ctx, -1, "mcraw parse error: %s", e.what());
    }
    catch (...) {
        if (ctx->dec) { delete ctx->dec; ctx->dec = nullptr; }
        return set_error(ctx, -1, "mcraw parse error: unknown");
    }

    return 0;
}

//-----------------------------------------------------------------------------
void mr_packet_free(mr_packet_t* pkt)
{
    if (pkt && pkt->data) {
        std::free(pkt->data);
        pkt->data = nullptr;
        pkt->size = 0;
        pkt->allocated = 0;
    }
}

//-----------------------------------------------------------------------------
static uint8_t* packet_resize(mr_packet_t* pkt, uint32_t size)
{
    if (size > pkt->allocated) {
        pkt->allocated = size + (size >> 2);
        pkt->data = (uint8_t*)(pkt->data ? std::realloc(pkt->data, pkt->allocated)
                                         : std::malloc(pkt->allocated));
    }
    pkt->size = size;
    return pkt->data;
}

//-----------------------------------------------------------------------------
int mr_read_video_frame(FILE* fd, int64_t offset, mr_packet_t* pkt)
{
    if (!fd || !pkt) return kMrErrorRead;
    if (MR_FSEEK(fd, offset, SEEK_SET) != 0) return kMrErrorSeek;

    mr_item_t item = {};
    if (std::fread(&item, sizeof(mr_item_t), 1, fd) != 1) return kMrErrorRead;
    if (item.type != BUFFER) return kMrErrorRead;

    if (!packet_resize(pkt, item.size)) return kMrErrorRead;
    if (std::fread(pkt->data, pkt->size, 1, fd) != 1) return kMrErrorRead;

    pkt->timestamp = -1;
    return 0;
}

//-----------------------------------------------------------------------------
int mr_read_audio_packet(FILE* fd, int64_t offset, mr_packet_t* pkt)
{
    if (!fd || !pkt) return kMrErrorRead;
    if (MR_FSEEK(fd, offset, SEEK_SET) != 0) return kMrErrorSeek;

    mr_item_t item = {};
    if (std::fread(&item, sizeof(mr_item_t), 1, fd) != 1) return kMrErrorRead;
    if (item.type != AUDIO_DATA) return kMrErrorRead;

    if (!packet_resize(pkt, item.size)) return kMrErrorRead;
    if (std::fread(pkt->data, pkt->size, 1, fd) != 1) return kMrErrorRead;

    pkt->timestamp = -1;

    // Audio metadata may follow; absent in older files.
    mr_audio_metadata_t md = {};
    if (std::fread(&md, sizeof(mr_audio_metadata_t), 1, fd) == 1) {
        if (md.item.type == AUDIO_DATA_METADATA) {
            pkt->timestamp = md.timestampNs;
        }
    }

    return 0;
}

//-----------------------------------------------------------------------------
int mr_read_frame_metadata(FILE* fd, mr_frame_data_t* frame_data)
{
    if (!fd || !frame_data) return kMrErrorRead;

    mr_item_t item = {};
    if (std::fread(&item, sizeof(mr_item_t), 1, fd) != 1) return kMrErrorRead;
    if (item.type != METADATA) return kMrErrorMetadata;

    std::vector<uint8_t> buf(item.size);
    if (item.size > 0 && std::fread(buf.data(), item.size, 1, fd) != 1) {
        return kMrErrorRead;
    }

    try {
        json j = json::parse(buf.begin(), buf.end());
        if (!j.is_object()) return kMrErrorMetadata;

        json_get_scalar(j, "iso",              frame_data->iso);
        json_get_scalar(j, "width",            frame_data->width);
        json_get_scalar(j, "height",           frame_data->height);
        json_get_scalar(j, "originalWidth",    frame_data->original_width);
        json_get_scalar(j, "originalHeight",   frame_data->original_height);
        json_get_scalar(j, "exposureTime",     frame_data->exposure_time);
        json_get_scalar(j, "recvdTimestampMs", frame_data->timestamp);
        {
            int o = 0; if (json_get_scalar(j, "orientation",     o)) frame_data->orientation     = (int16_t)o;
            int c = 0; if (json_get_scalar(j, "compressionType", c)) frame_data->compression_type = (int16_t)c;
        }

        if (json_copy_matrix(j, "asShotNeutral", frame_data->as_shot_neutral, 3) < 3) {
            frame_data->as_shot_neutral[0] = 1.0;
            frame_data->as_shot_neutral[1] = 1.0;
            frame_data->as_shot_neutral[2] = 1.0;
        }

        std::string pf;
        if (json_get_string(j, "pixelFormat", pf) && pf.compare(0, 5, "raw16") == 0) {
            frame_data->stored_pixel_format = 16;
        }
    }
    catch (const std::exception&) {
        return kMrErrorMetadata;
    }

    return 0;
}

//-----------------------------------------------------------------------------
void mr_dump(mr_ctx_t* ctx)
{
    if (!ctx || !ctx->fd) return;

    static const char* type_str[8] = {
        "BUFFER_INDEX,       ",
        "BUFFER_INDEX_DATA,  ",
        "BUFFER,             ",
        "METADATA,           ",
        "AUDIO_INDEX,        ",
        "AUDIO_DATA,         ",
        "AUDIO_DATA_METADATA,",
        "undefined,          "
    };

    MR_FSEEK(ctx->fd, 0, SEEK_END);
    int64_t file_size = ftell(ctx->fd);
    MR_FSEEK(ctx->fd, 0, SEEK_SET);

    mr_hdr_t hdr = {};
    if (std::fread(&hdr, sizeof(hdr), 1, ctx->fd) != 1) {
        std::printf("File is too short to be valid\n");
        return;
    }
    hdr.ident[6] = 0;

    std::printf("File size: %lld\n",  (long long)file_size);
    std::printf("    Ident: %s\n",    (const char*)hdr.ident);
    std::printf("  Version: %d\n\n",  (int)hdr.version);

    int64_t index_end = file_size - (int64_t)(sizeof(mr_buffer_index_t) + sizeof(mr_item_t));
    int64_t pos = ftell(ctx->fd);
    mr_item_t item = {};

    while (pos < index_end) {
        if (std::fread(&item, sizeof(mr_item_t), 1, ctx->fd) != 1) {
            std::printf("Failed to read item\n");
            return;
        }
        std::printf("type: %s size: %8u, offset: %lld\n",
                    type_str[item.type & 7], (unsigned)item.size, (long long)pos);
        pos += (int64_t)(sizeof(mr_item_t) + item.size);
        if (MR_FSEEK(ctx->fd, item.size, SEEK_CUR) != 0) break;
    }

    std::printf("Index offset: %lld\n", (long long)index_end);
}

//-----------------------------------------------------------------------------
FILE*               mr_get_file_handle(mr_ctx_t* ctx)        { return ctx ? ctx->fd : nullptr; }
uint32_t            mr_get_frame_count(mr_ctx_t* ctx)        { return ctx ? (uint32_t)ctx->video_offsets.size() : 0; }
uint32_t            mr_get_audio_packet_count(mr_ctx_t* ctx) { return ctx ? (uint32_t)ctx->audio_offsets.size() : 0; }
mr_buffer_offset_t* mr_get_index(mr_ctx_t* ctx)              { return (ctx && !ctx->video_offsets.empty()) ? ctx->video_offsets.data() : nullptr; }
mr_buffer_offset_t* mr_get_audio_index(mr_ctx_t* ctx)        { return (ctx && !ctx->audio_offsets.empty()) ? ctx->audio_offsets.data() : nullptr; }

int32_t  mr_get_width(mr_ctx_t* ctx)             { return ctx ? ctx->frame_data.width  : 0; }
int32_t  mr_get_height(mr_ctx_t* ctx)            { return ctx ? ctx->frame_data.height : 0; }
int32_t  mr_get_stored_format(mr_ctx_t* ctx)     { return ctx ? ctx->frame_data.stored_pixel_format : 0; }
int32_t  mr_get_bits_per_pixel(mr_ctx_t* ctx)
{
    if (!ctx) return 0;
    return ctx->white_level <= 0x3FF ? 10 : ctx->white_level <= 0xFFF ? 12 : 14;
}
int      mr_get_frame_rate(mr_ctx_t* ctx, int* num, int* den)
{
    if (!ctx) return -1;
    if (num) *num = ctx->frame_rate.num;
    if (den) *den = ctx->frame_rate.den;
    return 0;
}
int16_t  mr_get_black_level(mr_ctx_t* ctx)       { return ctx ? ctx->black_level : 0; }
int16_t  mr_get_white_level(mr_ctx_t* ctx)       { return ctx ? ctx->white_level : 0; }
double*  mr_get_color_matrix1(mr_ctx_t* ctx)     { return ctx ? ctx->color_matrix1   : nullptr; }
double*  mr_get_color_matrix2(mr_ctx_t* ctx)     { return ctx ? ctx->color_matrix2   : nullptr; }
double*  mr_get_forward_matrix1(mr_ctx_t* ctx)   { return ctx ? ctx->forward_matrix1 : nullptr; }
double*  mr_get_forward_matrix2(mr_ctx_t* ctx)   { return ctx ? ctx->forward_matrix2 : nullptr; }
double   mr_get_focal_length(mr_ctx_t* ctx)      { return ctx ? ctx->focal_length    : 0.0; }
double   mr_get_aperture(mr_ctx_t* ctx)          { return ctx ? ctx->aperture        : 0.0; }
int      mr_get_iso(mr_ctx_t* ctx)               { return ctx ? ctx->frame_data.iso  : 0; }
int      mr_get_exposure_time(mr_ctx_t* ctx)     { return ctx ? (int)ctx->frame_data.exposure_time : 0; }
double*  mr_get_as_shot_neutral(mr_ctx_t* ctx)   { return ctx ? ctx->frame_data.as_shot_neutral : nullptr; }
int64_t  mr_get_timestamp(mr_ctx_t* ctx)         { return ctx ? ctx->frame_data.timestamp : 0; }
const char* mr_get_manufacturer(mr_ctx_t* ctx)   { return ctx ? ctx->manufacturer : ""; }
const char* mr_get_model(mr_ctx_t* ctx)          { return ctx ? ctx->model        : ""; }
uint32_t mr_get_cfa_pattern(mr_ctx_t* ctx)       { return ctx ? ctx->cfa_pattern : 0; }
int      mr_get_compression_type(mr_ctx_t* ctx)  { return ctx ? ctx->frame_data.compression_type : 0; }
int32_t  mr_get_audio_sample_rate(mr_ctx_t* ctx) { return ctx ? ctx->audio_sample_rate : 0; }
int32_t  mr_get_audio_channels(mr_ctx_t* ctx)    { return ctx ? ctx->audio_channels    : 0; }

} // extern "C"
