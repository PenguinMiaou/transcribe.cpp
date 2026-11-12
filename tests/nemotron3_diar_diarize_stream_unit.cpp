// nemotron3_diar_diarize_stream_unit.cpp - Nemotron-3-Diarization push mode
// (transcribe_diarize_stream_*) against a real GGUF
// (TRANSCRIBE_NEMOTRON3_DIAR_GGUF; RC 77 skip when unset).
//
// Oracle: transcribe_diarize_run on the whole recording. For every split the
// fed stream must give the same frame probabilities (compared bit for bit
// through the diar.probs debug dump) and the same segments, and before
// finish the decided frames must be a prefix of the whole-run frames. With
// peeks after every feed all of that must still hold, and a peek must give
// the segments of a whole run on the audio fed so far.
//
// Optional environment:
//   TRANSCRIBE_NEMOTRON3_DIAR_STREAM_WAV      another 16 kHz mono recording;
//                                             runs only the fixed-piece
//                                             comparison and prints numbers
//   TRANSCRIBE_NEMOTRON3_DIAR_STREAM_PIECE_MS piece length for it (20000)
//   TRANSCRIBE_NEMOTRON3_DIAR_STREAM_PRESET   "low_latency" for it
//   TRANSCRIBE_NEMOTRON3_DIAR_STREAM_PEEK     "1": peek after every feed of it
//   TRANSCRIBE_NEMOTRON3_DIAR_STREAM_BACKEND  "auto" for the default backend
//                                             (CPU otherwise)

#include "transcribe.h"
#include "transcribe/diarize.h"
#include "transcribe/nemotron3_diar.h"
#include "wav.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

constexpr int k_spk      = 8;
constexpr int k_frame_ms = 10;

struct Row {
    int64_t t0, t1;
    int32_t spk;

    bool operator==(const Row & o) const { return t0 == o.t0 && t1 == o.t1 && spk == o.spk; }
};

std::vector<Row> session_rows(const transcribe_diarize_session * s) {
    std::vector<Row> rows;
    for (int i = 0; i < transcribe_diarize_n_segments(s); ++i) {
        transcribe_speaker_segment r;
        transcribe_speaker_segment_init(&r);
        transcribe_diarize_get_segment(s, i, &r);
        rows.push_back({ r.t0_ms, r.t1_ms, r.speaker_id });
    }
    return rows;
}

// The library's probs -> segments rule, on the first n_frames rows.
std::vector<Row> rows_of(const std::vector<float> & probs, size_t n_frames) {
    std::vector<Row> rows;
    for (int s = 0; s < k_spk; ++s) {
        long start = -1;
        for (size_t t = 0; t <= n_frames; ++t) {
            const bool active = t < n_frames && probs[t * k_spk + s] > 0.5f;
            if (active && start < 0) {
                start = static_cast<long>(t);
            } else if (!active && start >= 0) {
                rows.push_back({ start * k_frame_ms, static_cast<int64_t>(t) * k_frame_ms, s + 1 });
                start = -1;
            }
        }
    }
    return rows;
}

std::string g_dump_dir;

// The frame probabilities the library last dumped, row-major [n, 8].
std::vector<float> read_probs() {
    std::ifstream      f(g_dump_dir + "/diar.probs.f32", std::ios::binary | std::ios::ate);
    std::vector<float> v;
    if (f) {
        v.resize(static_cast<size_t>(f.tellg()) / sizeof(float));
        f.seekg(0);
        f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    }
    return v;
}

transcribe_diarize_params params_for(transcribe_nemotron3_diar_diarize_ext & ext,
                                     transcribe_nemotron3_diar_preset        preset) {
    transcribe_nemotron3_diar_diarize_ext_init(&ext);
    ext.preset = preset;
    transcribe_diarize_params dp;
    transcribe_diarize_params_init(&dp);
    dp.family = &ext.ext;
    return dp;
}

struct Whole {
    std::vector<float> probs;
    std::vector<Row>   rows;
    double             seconds = 0.0;
};

Whole run_whole(transcribe_diarize_session *     s,
                const std::vector<float> &       pcm,
                transcribe_nemotron3_diar_preset preset) {
    transcribe_nemotron3_diar_diarize_ext ext;
    const transcribe_diarize_params       dp = params_for(ext, preset);
    Whole                                 w;
    const auto                            t0 = std::chrono::steady_clock::now();
    CHECK(transcribe_diarize_run(s, pcm.data(), static_cast<int>(pcm.size()), &dp) == TRANSCRIBE_OK);
    w.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    w.rows    = session_rows(s);
    w.probs   = read_probs();
    CHECK(w.probs.size() == pcm.size() / 160 * k_spk);
    CHECK(w.rows == rows_of(w.probs, w.probs.size() / k_spk));
    return w;
}

struct StreamStats {
    float   max_abs_diff  = 0.0f;
    bool    bit_identical = false;
    bool    same_rows     = false;
    int     n_feeds       = 0;
    double  feed_s_sum = 0.0, feed_s_max = 0.0, finish_s = 0.0;
    // audio fed minus decided end after a feed, over the feeds from the
    // first decision on
    int64_t lag_min = std::numeric_limits<int64_t>::max(), lag_max = 0, lag_sum = 0;
    int     lag_n   = 0;
    // peeks: time, and 10 ms frames of the provisional tail whose set of
    // active speakers differs from the final result
    int     n_peeks = 0, peeks_differing = 0;
    double  peek_s_sum = 0.0, peek_s_max = 0.0;
    long    tail_frames = 0, tail_frames_differing = 0;
};

enum class Peek { None, Every, EveryVsPrefixRun };

// Active-speaker bit mask per 10 ms frame for frames [0, n).
std::vector<unsigned> masks_of(const std::vector<Row> & rows, size_t n) {
    std::vector<unsigned> m(n, 0u);
    for (const Row & r : rows) {
        for (size_t t = static_cast<size_t>(r.t0 / k_frame_ms); t < std::min(n, static_cast<size_t>(r.t1 / k_frame_ms));
             ++t) {
            m[t] |= 1u << (r.spk - 1);
        }
    }
    return m;
}

// Feed `pcm` in the given piece sizes (cycled), checking the decided prefix
// against the whole run after every feed that moved it.
StreamStats run_stream(transcribe_diarize_session *     s,
                       const std::vector<float> &       pcm,
                       transcribe_nemotron3_diar_preset preset,
                       const std::vector<size_t> &      pieces,
                       const Whole &                    whole,
                       Peek                             peek       = Peek::None,
                       transcribe_diarize_session *     prefix_dia = nullptr) {
    transcribe_nemotron3_diar_diarize_ext ext;
    const transcribe_diarize_params       dp = params_for(ext, preset);
    StreamStats                           st;
    CHECK(transcribe_diarize_stream_begin(s, &dp) == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_n_segments(s) == 0 && transcribe_diarize_stream_decided_ms(s) == 0);

    const std::vector<unsigned> final_masks = masks_of(whole.rows, whole.probs.size() / k_spk);
    int64_t                     decided     = 0;
    size_t                      pos         = 0;
    for (size_t k = 0; pos < pcm.size(); ++k) {
        const size_t n  = std::min(pieces[k % pieces.size()], pcm.size() - pos);
        const auto   t0 = std::chrono::steady_clock::now();
        const auto   rc = transcribe_diarize_stream_feed(s, pcm.data() + pos, static_cast<int>(n));
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        CHECK(rc == TRANSCRIBE_OK);
        if (rc != TRANSCRIBE_OK) {
            return st;
        }
        pos += n;
        ++st.n_feeds;
        st.feed_s_sum += dt;
        st.feed_s_max = std::max(st.feed_s_max, dt);

        const int64_t now = transcribe_diarize_stream_decided_ms(s);
        CHECK(now >= decided && now % k_frame_ms == 0);
        CHECK(now <= static_cast<int64_t>(pos / 16));
        if (now != decided) {
            decided                        = now;
            const size_t             n_dec = static_cast<size_t>(decided / k_frame_ms);
            const std::vector<float> got   = read_probs();
            CHECK(got.size() == n_dec * k_spk && got.size() <= whole.probs.size());
            CHECK(got.size() <= whole.probs.size() &&
                  std::memcmp(got.data(), whole.probs.data(), got.size() * sizeof(float)) == 0);
            CHECK(session_rows(s) == rows_of(whole.probs, n_dec));
        }
        if (decided > 0) {
            const int64_t lag = static_cast<int64_t>(pos / 16) - decided;
            st.lag_min        = std::min(st.lag_min, lag);
            st.lag_max        = std::max(st.lag_max, lag);
            st.lag_sum += lag;
            ++st.lag_n;
        }

        if (peek != Peek::None) {
            const auto   p0 = std::chrono::steady_clock::now();
            const auto   pr = transcribe_diarize_stream_peek(s);
            const double pt = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
            CHECK(pr == TRANSCRIBE_OK);
            ++st.n_peeks;
            st.peek_s_sum += pt;
            st.peek_s_max = std::max(st.peek_s_max, pt);
            CHECK(transcribe_diarize_stream_decided_ms(s) == decided);
            const std::vector<Row>      seen     = session_rows(s);
            const size_t                n_dec    = static_cast<size_t>(decided / k_frame_ms);
            const size_t                n_fed    = pos / 160;
            // The decided part of the peek is the decided result ...
            const std::vector<unsigned> got      = masks_of(seen, n_fed);
            bool                        dec_same = true;
            for (size_t t = 0; t < n_dec; ++t) {
                dec_same = dec_same && got[t] == final_masks[t];
            }
            CHECK(dec_same);
            for (const Row & r : seen) {
                CHECK(r.t1 <= static_cast<int64_t>(n_fed) * k_frame_ms);
            }
            // ... and the tail is provisional: count where it ends up different.
            long differing = 0;
            for (size_t t = n_dec; t < n_fed; ++t) {
                differing += got[t] != final_masks[t] ? 1 : 0;
            }
            st.tail_frames += static_cast<long>(n_fed - n_dec);
            st.tail_frames_differing += differing;
            st.peeks_differing += differing > 0 ? 1 : 0;
            // A peek is a whole run on the audio fed so far.
            if (peek == Peek::EveryVsPrefixRun && pos >= 160) {
                CHECK(transcribe_diarize_run(prefix_dia, pcm.data(), static_cast<int>(pos), &dp) == TRANSCRIBE_OK);
                CHECK(session_rows(prefix_dia) == seen);
            }
            if (pos == pcm.size()) {
                CHECK(seen == whole.rows);
            }
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    CHECK(transcribe_diarize_stream_finish(s) == TRANSCRIBE_OK);
    st.finish_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(transcribe_diarize_stream_decided_ms(s) == static_cast<int64_t>(whole.probs.size() / k_spk) * k_frame_ms);

    const std::vector<float> got = read_probs();
    CHECK(got.size() == whole.probs.size());
    if (got.size() == whole.probs.size()) {
        for (size_t i = 0; i < got.size(); ++i) {
            st.max_abs_diff = std::max(st.max_abs_diff, std::fabs(got[i] - whole.probs[i]));
        }
        st.bit_identical = std::memcmp(got.data(), whole.probs.data(), got.size() * sizeof(float)) == 0;
    }
    st.same_rows = session_rows(s) == whole.rows;
    CHECK(st.bit_identical);
    CHECK(st.same_rows);
    return st;
}

void report(const char * label, const StreamStats & st) {
    std::printf(
        "%s: feeds=%d max_abs_diff=%g bit_identical=%d segments_identical=%d feed_s mean=%.3f max=%.3f "
        "finish_s=%.3f",
        label, st.n_feeds, static_cast<double>(st.max_abs_diff), st.bit_identical ? 1 : 0, st.same_rows ? 1 : 0,
        st.n_feeds > 0 ? st.feed_s_sum / st.n_feeds : 0.0, st.feed_s_max, st.finish_s);
    if (st.lag_n > 0) {
        std::printf(" lag_ms min=%lld mean=%lld max=%lld", static_cast<long long>(st.lag_min),
                    static_cast<long long>(st.lag_sum / st.lag_n), static_cast<long long>(st.lag_max));
    }
    if (st.n_peeks > 0) {
        std::printf(" | peeks=%d peek_s mean=%.3f max=%.3f tail_frames=%ld differing=%ld (%.2f%%) peeks_with_diff=%d",
                    st.n_peeks, st.peek_s_sum / st.n_peeks, st.peek_s_max, st.tail_frames, st.tail_frames_differing,
                    st.tail_frames > 0 ?
                        100.0 * static_cast<double>(st.tail_frames_differing) / static_cast<double>(st.tail_frames) :
                        0.0,
                    st.peeks_differing);
    }
    std::printf("\n");
}

bool abort_now(void *) {
    return true;
}

}  // namespace

int main() {
    const char * path = std::getenv("TRANSCRIBE_NEMOTRON3_DIAR_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::fprintf(stderr, "nemotron3_diar_diarize_stream_unit: TRANSCRIBE_NEMOTRON3_DIAR_GGUF not set; skipping.\n");
        return 77;
    }
    const char *       other_wav = std::getenv("TRANSCRIBE_NEMOTRON3_DIAR_STREAM_WAV");
    const bool         stress    = other_wav != nullptr && other_wav[0] != '\0';
    std::vector<float> pcm;
    std::string        err;
    if (!transcribe_cli::load_wav_mono_16k(
            stress ? std::string(other_wav) : std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/nemotron3-diar-8spk-mix.wav",
            pcm, err)) {
        std::fprintf(stderr, "nemotron3_diar_diarize_stream_unit: wav: %s\n", err.c_str());
        return 77;
    }
    ::unsetenv("TRANSCRIBE_NEMOTRON3_DIAR_PRESET");

    // The frame probabilities are read back through the debug dump; the
    // dumper latches its directory on the first run of the process.
    {
        std::error_code ec;
        const auto      dir =
            std::filesystem::temp_directory_path(ec) /
            ("transcribe_n3d_stream_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            std::fprintf(stderr, "nemotron3_diar_diarize_stream_unit: no temp dir; skipping.\n");
            return 77;
        }
        g_dump_dir = dir.string();
        ::setenv("TRANSCRIBE_DUMP_DIR", g_dump_dir.c_str(), 1);
    }

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    const char * backend = std::getenv("TRANSCRIBE_NEMOTRON3_DIAR_STREAM_BACKEND");
    mp.backend =
        (backend != nullptr && std::string(backend) == "auto") ? TRANSCRIBE_BACKEND_AUTO : TRANSCRIBE_BACKEND_CPU;
    transcribe_model * model = nullptr;
    if (transcribe_model_load_file(path, &mp, &model) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL: model load\n");
        return EXIT_FAILURE;
    }
    transcribe_diarize_session * dia = nullptr;
    CHECK(transcribe_diarize_session_init(model, nullptr, &dia) == TRANSCRIBE_OK);
    std::printf("backend=%s audio_s=%.2f\n", transcribe_model_backend(model),
                static_cast<double>(pcm.size()) / 16000.0);

    const auto DEF = TRANSCRIBE_NEMOTRON3_DIAR_PRESET_DEFAULT;
    const auto LOW = TRANSCRIBE_NEMOTRON3_DIAR_PRESET_LOW_LATENCY;

    if (stress) {
        const char * piece_env  = std::getenv("TRANSCRIBE_NEMOTRON3_DIAR_STREAM_PIECE_MS");
        const long   piece_ms   = piece_env != nullptr && piece_env[0] != '\0' ? std::atol(piece_env) : 20000;
        const char * preset_env = std::getenv("TRANSCRIBE_NEMOTRON3_DIAR_STREAM_PRESET");
        const auto   preset     = (preset_env != nullptr && std::string(preset_env) == "low_latency") ? LOW : DEF;
        const Whole  whole      = run_whole(dia, pcm, preset);
        std::printf("whole run: %.3f s, %zu frames, %zu segments\n", whole.seconds, whole.probs.size() / k_spk,
                    whole.rows.size());
        char label[64];
        std::snprintf(label, sizeof(label), "%ld ms pieces", piece_ms);
        const char * peek_env = std::getenv("TRANSCRIBE_NEMOTRON3_DIAR_STREAM_PEEK");
        const Peek   peek     = (peek_env != nullptr && std::string(peek_env) == "1") ? Peek::Every : Peek::None;
        report(label, run_stream(dia, pcm, preset, { static_cast<size_t>(std::max(1L, piece_ms)) * 16 }, whole, peek));
    } else {
        // 1. Default operating point: 20 s, 1 s and uneven pieces (hop
        //    multiples, off-hop sizes, single samples, a long one).
        const Whole whole = run_whole(dia, pcm, DEF);
        std::printf("whole run: %.3f s, %zu frames, %zu segments\n", whole.seconds, whole.probs.size() / k_spk,
                    whole.rows.size());
        report("20 s pieces", run_stream(dia, pcm, DEF, { 320000 }, whole));
        report("1 s pieces", run_stream(dia, pcm, DEF, { 16000 }, whole));
        report("uneven pieces",
               run_stream(dia, pcm, DEF, { 1, 159, 160, 161, 255, 257, 7919, 48000, 3, 100003, 4801, 512, 640000 },
                          whole));
        // The whole recording in one feed, and a run after a stream.
        report("one piece", run_stream(dia, pcm, DEF, { pcm.size() }, whole));
        CHECK(run_whole(dia, pcm, DEF).rows == whole.rows);

        // 2. Low-latency operating point (0.72 s chunks): many decisions
        //    before finish, each checked as a prefix of the whole run.
        const Whole whole_low = run_whole(dia, pcm, LOW);
        report("low_latency, 1 s pieces", run_stream(dia, pcm, LOW, { 16000 }, whole_low));
        report("low_latency, uneven pieces", run_stream(dia, pcm, LOW, { 1601, 7919, 23, 33333, 160 }, whole_low));

        // 3. Peek after every feed: the stream's frames and segments are
        //    still those of the whole run, and each peek equals a whole run
        //    (on a second session) of the audio fed so far.
        transcribe_diarize_session * dia2 = nullptr;
        CHECK(transcribe_diarize_session_init(model, nullptr, &dia2) == TRANSCRIBE_OK);
        report("peek, 20 s pieces", run_stream(dia, pcm, DEF, { 320000 }, whole, Peek::EveryVsPrefixRun, dia2));
        report("peek, uneven pieces", run_stream(dia, pcm, DEF, { 100, 159, 48000, 7919, 100003, 257, 640000 }, whole,
                                                 Peek::EveryVsPrefixRun, dia2));
        report("peek, low_latency, 4 s pieces",
               run_stream(dia, pcm, LOW, { 64000 }, whole_low, Peek::EveryVsPrefixRun, dia2));
        report("peek, low_latency, 1 s pieces", run_stream(dia, pcm, LOW, { 16000 }, whole_low, Peek::Every));
        transcribe_diarize_session_free(dia2);

        // 3b. Call order and input rejection.
        const int n_before = transcribe_diarize_n_segments(dia);
        CHECK(n_before > 0);
        CHECK(transcribe_diarize_stream_feed(dia, pcm.data(), 16000) == TRANSCRIBE_ERR_INVALID_ARG);  // no stream
        CHECK(transcribe_diarize_stream_peek(dia) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(transcribe_diarize_stream_finish(dia) == TRANSCRIBE_ERR_INVALID_ARG);
        transcribe_nemotron3_diar_diarize_ext bad;
        transcribe_diarize_params bad_dp = params_for(bad, static_cast<transcribe_nemotron3_diar_preset>(99));
        CHECK(transcribe_diarize_stream_begin(dia, &bad_dp) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(transcribe_diarize_n_segments(dia) == n_before);  // rejected before the commit point

        CHECK(transcribe_diarize_stream_begin(dia, nullptr) == TRANSCRIBE_OK);
        CHECK(transcribe_diarize_n_segments(dia) == 0);
        CHECK(transcribe_diarize_stream_feed(dia, nullptr, 0) == TRANSCRIBE_OK);
        CHECK(transcribe_diarize_stream_feed(dia, nullptr, 160) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(transcribe_diarize_stream_feed(dia, pcm.data(), -1) == TRANSCRIBE_ERR_INVALID_ARG);
        const std::vector<float> nan_pcm(160, std::numeric_limits<float>::quiet_NaN());
        CHECK(transcribe_diarize_stream_feed(dia, nan_pcm.data(), 160) == TRANSCRIBE_ERR_INVALID_ARG);
        // ... none of which ended the stream: sub-hop audio is an empty OK result.
        CHECK(transcribe_diarize_stream_feed(dia, pcm.data(), 100) == TRANSCRIBE_OK);
        CHECK(transcribe_diarize_stream_peek(dia) == TRANSCRIBE_OK);
        CHECK(transcribe_diarize_n_segments(dia) == 0);
        CHECK(transcribe_diarize_stream_finish(dia) == TRANSCRIBE_OK);
        CHECK(transcribe_diarize_n_segments(dia) == 0 && transcribe_diarize_stream_decided_ms(dia) == 0);
        CHECK(transcribe_diarize_stream_finish(dia) == TRANSCRIBE_ERR_INVALID_ARG);

        // 4. begin restarts a stream in progress; a run ends one.
        CHECK(transcribe_diarize_stream_begin(dia, nullptr) == TRANSCRIBE_OK);
        CHECK(transcribe_diarize_stream_feed(dia, pcm.data(), 640000) == TRANSCRIBE_OK);
        CHECK(transcribe_diarize_stream_decided_ms(dia) > 0);
        report("restart, 20 s pieces", run_stream(dia, pcm, DEF, { 320000 }, whole));
        CHECK(transcribe_diarize_stream_begin(dia, nullptr) == TRANSCRIBE_OK);
        CHECK(transcribe_diarize_stream_feed(dia, pcm.data(), 16000) == TRANSCRIBE_OK);
        CHECK(transcribe_diarize_run(dia, pcm.data(), 16000, nullptr) == TRANSCRIBE_OK);
        CHECK(transcribe_diarize_stream_feed(dia, pcm.data(), 16000) == TRANSCRIBE_ERR_INVALID_ARG);

        // 5. Abort: the feed that would run a chunk returns ERR_ABORTED and
        //    ends the stream with an empty result.
        CHECK(transcribe_diarize_stream_begin(dia, nullptr) == TRANSCRIBE_OK);
        CHECK(transcribe_diarize_stream_feed(dia, pcm.data(), 640000) == TRANSCRIBE_OK);
        CHECK(transcribe_diarize_n_segments(dia) > 0);
        // An aborted peek leaves the stream and the segments in place.
        const int n_dec_rows = transcribe_diarize_n_segments(dia);
        transcribe_diarize_set_abort_callback(dia, abort_now, nullptr);
        CHECK(transcribe_diarize_stream_peek(dia) == TRANSCRIBE_ERR_ABORTED);
        CHECK(transcribe_diarize_n_segments(dia) == n_dec_rows && transcribe_diarize_stream_decided_ms(dia) > 0);
        transcribe_diarize_set_abort_callback(dia, nullptr, nullptr);
        CHECK(transcribe_diarize_stream_peek(dia) == TRANSCRIBE_OK);
        transcribe_diarize_set_abort_callback(dia, abort_now, nullptr);
        CHECK(transcribe_diarize_stream_feed(dia, pcm.data() + 640000, 16000) == TRANSCRIBE_ERR_ABORTED);
        CHECK(transcribe_diarize_n_segments(dia) == 0 && transcribe_diarize_stream_decided_ms(dia) == 0);
        transcribe_diarize_set_abort_callback(dia, nullptr, nullptr);
        CHECK(transcribe_diarize_stream_feed(dia, pcm.data(), 16000) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(transcribe_diarize_stream_begin(dia, nullptr) == TRANSCRIBE_OK);
        transcribe_diarize_set_abort_callback(dia, abort_now, nullptr);
        CHECK(transcribe_diarize_stream_finish(dia) == TRANSCRIBE_ERR_ABORTED);
        transcribe_diarize_set_abort_callback(dia, nullptr, nullptr);
        report("after abort, 20 s pieces", run_stream(dia, pcm, DEF, { 320000 }, whole));
    }

    transcribe_diarize_session_free(dia);
    transcribe_model_free(model);
    {
        std::error_code ec;
        std::filesystem::remove_all(g_dump_dir, ec);
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("nemotron3_diar_diarize_stream_unit: ok\n");
    return EXIT_SUCCESS;
}
