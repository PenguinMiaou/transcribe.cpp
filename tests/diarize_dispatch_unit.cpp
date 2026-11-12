// diarize_dispatch_unit.cpp - DIARIZE role dispatcher (transcribe-diarize.cpp)
// against a fake arch: role checks, pre-clear validation, probs -> segments,
// accessors, push mode (stream_begin / feed / finish).

#include "transcribe-arch.h"
#include "transcribe-diarize.h"
#include "transcribe-model.h"
#include "transcribe.h"
#include "transcribe/diarize.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

constexpr uint32_t k_fake_kind = 0x4B414644;  // 'DFAK'

int               g_run_calls   = 0;
transcribe_status g_run_status  = TRANSCRIBE_OK;
transcribe_status g_validate_st = TRANSCRIBE_OK;

int fake_max_speakers(const transcribe_model *) {
    return 3;
}

transcribe_diarize_session * fake_new_session() {
    return new transcribe_diarize_session();
}

transcribe_status fake_run_validate(const transcribe_diarize_params *) {
    return g_validate_st;
}

// Frames at 80 ms: speaker 1 active in frames 1-2, speaker 2 in frames 2-3.
transcribe_status fake_run(transcribe_diarize_session *,
                           const float *,
                           int,
                           const transcribe_diarize_params *,
                           transcribe::DiarizeProbs & out) {
    ++g_run_calls;
    if (g_run_status != TRANSCRIBE_OK) {
        return g_run_status;
    }
    out.probs      = { 0.1f, 0.0f, 0.9f, 0.0f, 0.8f, 0.7f, 0.2f, 0.6f };
    out.n_frames   = 4;
    out.n_speakers = 2;
    out.frame_ms   = 80.0;
    return TRANSCRIBE_OK;
}

bool fake_accepts(const transcribe_model *, transcribe_ext_slot slot, uint32_t kind) {
    return slot == TRANSCRIBE_EXT_SLOT_DIARIZE_RUN && kind == k_fake_kind;
}

// Push mode of the fake: one 80 ms frame per 1280 fed samples, speaker 1
// active in every second frame; finish adds one closing frame for speaker 2.
std::vector<float> g_stream_probs;
int                g_stream_samples = 0;
transcribe_status  g_stream_status  = TRANSCRIBE_OK;

transcribe_status fake_stream_begin(transcribe_diarize_session *, const transcribe_diarize_params *) {
    g_stream_probs.clear();
    g_stream_samples = 0;
    return TRANSCRIBE_OK;
}

transcribe_status fake_stream_feed(transcribe_diarize_session *,
                                   const float *,
                                   int                            n_samples,
                                   transcribe::DiarizeProbsView & decided) {
    if (g_stream_status != TRANSCRIBE_OK) {
        return g_stream_status;
    }
    g_stream_samples += n_samples;
    while (static_cast<int>(g_stream_probs.size() / 2) < g_stream_samples / 1280) {
        const bool on = (g_stream_probs.size() / 2) % 2 == 0;
        g_stream_probs.push_back(on ? 0.9f : 0.1f);
        g_stream_probs.push_back(0.0f);
    }
    decided.probs      = g_stream_probs.data();
    decided.n_frames   = static_cast<int>(g_stream_probs.size() / 2);
    decided.n_speakers = 2;
    decided.frame_ms   = 80.0;
    return TRANSCRIBE_OK;
}

transcribe_status fake_stream_finish(transcribe_diarize_session *, transcribe::DiarizeProbs & out) {
    if (g_stream_status != TRANSCRIBE_OK) {
        return g_stream_status;
    }
    out.probs = g_stream_probs;
    out.probs.push_back(0.0f);
    out.probs.push_back(0.8f);
    out.n_frames   = static_cast<int>(out.probs.size() / 2);
    out.n_speakers = 2;
    out.frame_ms   = 80.0;
    return TRANSCRIBE_OK;
}

// Peek of the fake: the frames so far plus one provisional frame for speaker 2.
int g_peek_calls = 0;

transcribe_status fake_stream_peek(transcribe_diarize_session *, transcribe::DiarizeProbs & out) {
    ++g_peek_calls;
    if (g_stream_status != TRANSCRIBE_OK) {
        return g_stream_status;
    }
    out.probs = g_stream_probs;
    out.probs.push_back(0.0f);
    out.probs.push_back(0.8f);
    out.n_frames   = static_cast<int>(out.probs.size() / 2);
    out.n_speakers = 2;
    out.frame_ms   = 80.0;
    return TRANSCRIBE_OK;
}

const transcribe::DiarizeOps k_ops = {
    fake_max_speakers, fake_new_session, fake_run_validate, fake_run, nullptr, nullptr, nullptr, nullptr,
};
const transcribe::DiarizeOps k_stream_ops = {
    fake_max_speakers, fake_new_session, fake_run_validate,  fake_run,
    fake_stream_begin, fake_stream_feed, fake_stream_finish, fake_stream_peek,
};

const transcribe::Arch k_arch = {
    /* .name             = */ "fake-diarize",
    /* .load             = */ nullptr,
    /* .init_context     = */ nullptr,
    /* .run              = */ nullptr,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ fake_accepts,
    /* .run_validate     = */ nullptr,
    /* .diarize          = */ &k_ops,
};

const transcribe::Arch k_stream_arch = {
    /* .name             = */ "fake-diarize-stream",
    /* .load             = */ nullptr,
    /* .init_context     = */ nullptr,
    /* .run              = */ nullptr,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ fake_accepts,
    /* .run_validate     = */ nullptr,
    /* .diarize          = */ &k_stream_ops,
};

struct Fixture {
    transcribe_model             model;
    transcribe_diarize_session * session = nullptr;
    std::vector<float>           pcm     = std::vector<float>(1600, 0.0f);

    explicit Fixture(const transcribe::Arch * arch = &k_arch) {
        model.arch  = arch;
        model.roles = TRANSCRIBE_ROLE_DIARIZE;
        CHECK(transcribe_diarize_session_init(&model, nullptr, &session) == TRANSCRIBE_OK);
    }

    ~Fixture() { transcribe_diarize_session_free(session); }

    transcribe_status run(const transcribe_diarize_params * p = nullptr) {
        return transcribe_diarize_run(session, pcm.data(), static_cast<int>(pcm.size()), p);
    }
};

void test_probs_to_segments() {
    // [frames=5, speakers=2]; speaker 2 still active at the end.
    const float probs[] = { 0.6f, 0.0f, 0.6f, 0.0f, 0.4f, 0.0f, 0.9f, 0.51f, 0.0f, 0.7f };
    std::vector<transcribe::SpeakerSegmentEntry> out;
    transcribe::probs_to_segments(probs, 5, 2, 80.0, out);
    CHECK(out.size() == 3);
    CHECK(out[0].speaker_id == 1 && out[0].t0_ms == 0 && out[0].t1_ms == 160);
    CHECK(out[1].speaker_id == 1 && out[1].t0_ms == 240 && out[1].t1_ms == 320);
    CHECK(out[2].speaker_id == 2 && out[2].t0_ms == 240 && out[2].t1_ms == 400);
    CHECK(std::isnan(out[0].p));
}

void test_role_checks() {
    transcribe_model model;
    model.arch  = &k_arch;
    model.roles = TRANSCRIBE_ROLE_ASR;  // no DIARIZE bit

    transcribe_diarize_session * s = reinterpret_cast<transcribe_diarize_session *>(0x1);
    CHECK(transcribe_diarize_session_init(&model, nullptr, &s) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);
    CHECK(s == nullptr);

    transcribe_diarize_info info;
    transcribe_diarize_info_init(&info);
    CHECK(transcribe_diarize_get_info(&model, &info) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);

    model.roles = TRANSCRIBE_ROLE_DIARIZE;
    CHECK(transcribe_diarize_get_info(&model, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000 && info.max_speakers == 3);
    CHECK(transcribe_diarize_get_info(nullptr, &info) == TRANSCRIBE_ERR_INVALID_ARG);
    info.struct_size = 0;
    CHECK(transcribe_diarize_get_info(&model, &info) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);

    transcribe_diarize_session_params sp;
    transcribe_diarize_session_params_init(&sp);
    sp.n_threads = -1;
    CHECK(transcribe_diarize_session_init(&model, &sp, &s) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(s == nullptr);
    sp.n_threads = 2;
    CHECK(transcribe_diarize_session_init(&model, &sp, &s) == TRANSCRIBE_OK);
    CHECK(s != nullptr);
    transcribe_diarize_session_free(s);
    transcribe_diarize_session_free(nullptr);
}

void test_run_and_accessors() {
    Fixture f;
    CHECK(transcribe_diarize_n_segments(f.session) == 0);
    CHECK(f.run() == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_n_segments(f.session) == 2);

    transcribe_speaker_segment row;
    transcribe_speaker_segment_init(&row);
    CHECK(transcribe_diarize_get_segment(f.session, 1, &row) == TRANSCRIBE_OK);
    CHECK(row.speaker_id == 2 && row.t0_ms == 160 && row.t1_ms == 320);
    CHECK(transcribe_diarize_get_segment(f.session, 2, &row) == TRANSCRIBE_OK);  // out of range: zeroed
    CHECK(row.speaker_id == 0 && row.t0_ms == 0 && row.t1_ms == 0);
    CHECK(transcribe_diarize_get_segment(f.session, 0, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);

    transcribe_timings tm;
    transcribe_timings_init(&tm);
    CHECK(transcribe_diarize_get_timings(f.session, &tm) == TRANSCRIBE_OK);
}

// Malformed input and family-rejected params leave the previous result alone
// and never reach the family run.
void test_pre_clear_rejections_preserve_result() {
    g_validate_st = TRANSCRIBE_OK;
    g_run_status  = TRANSCRIBE_OK;
    Fixture f;
    CHECK(f.run() == TRANSCRIBE_OK);
    g_run_calls = 0;

    std::vector<float> bad(1600, 0.0f);
    bad[7] = std::numeric_limits<float>::quiet_NaN();
    CHECK(transcribe_diarize_run(f.session, bad.data(), 1600, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_run(f.session, nullptr, 1600, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_run(f.session, f.pcm.data(), 0, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);

    transcribe_ext            ext{ sizeof(transcribe_ext), 0x12345678u };
    transcribe_diarize_params p;
    transcribe_diarize_params_init(&p);
    p.family = &ext;
    CHECK(f.run(&p) == TRANSCRIBE_ERR_INVALID_ARG);  // kind not accepted on DIARIZE_RUN
    ext.size = 4;
    CHECK(f.run(&p) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    ext           = { sizeof(transcribe_ext), k_fake_kind };
    g_validate_st = TRANSCRIBE_ERR_INVALID_ARG;
    CHECK(f.run(&p) == TRANSCRIBE_ERR_INVALID_ARG);
    p.struct_size = 0;
    CHECK(f.run(&p) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);

    CHECK(g_run_calls == 0);
    CHECK(transcribe_diarize_n_segments(f.session) == 2);

    // A family failure after the commit point clears the result.
    g_validate_st = TRANSCRIBE_OK;
    g_run_status  = TRANSCRIBE_ERR_ABORTED;
    p.struct_size = sizeof(p);
    CHECK(f.run(&p) == TRANSCRIBE_ERR_ABORTED);
    CHECK(transcribe_diarize_n_segments(f.session) == 0);
    g_run_status = TRANSCRIBE_OK;
}

// A family without push mode refuses all three calls and keeps its result.
void test_stream_unsupported() {
    Fixture f;
    CHECK(f.run() == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_stream_begin(f.session, nullptr) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(transcribe_diarize_stream_feed(f.session, f.pcm.data(), 1600) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(transcribe_diarize_stream_peek(f.session) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(transcribe_diarize_stream_finish(f.session) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(transcribe_diarize_n_segments(f.session) == 2);
    CHECK(transcribe_diarize_stream_decided_ms(f.session) == 0);
    CHECK(transcribe_diarize_stream_decided_ms(nullptr) == 0);
}

void test_stream_calls() {
    g_validate_st   = TRANSCRIBE_OK;
    g_run_status    = TRANSCRIBE_OK;
    g_stream_status = TRANSCRIBE_OK;
    Fixture f(&k_stream_arch);
    CHECK(transcribe_diarize_stream_begin(nullptr, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_stream_feed(nullptr, f.pcm.data(), 1600) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_stream_finish(nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_stream_peek(nullptr) == TRANSCRIBE_ERR_INVALID_ARG);

    // No stream yet: feed / peek / finish are refused and the run result stays.
    CHECK(f.run() == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_stream_peek(f.session) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_stream_feed(f.session, f.pcm.data(), 1600) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_stream_finish(f.session) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_n_segments(f.session) == 2);

    // begin validates params like run, before the previous result is touched.
    transcribe_ext            ext{ sizeof(transcribe_ext), 0x12345678u };
    transcribe_diarize_params p;
    transcribe_diarize_params_init(&p);
    p.family = &ext;
    CHECK(transcribe_diarize_stream_begin(f.session, &p) == TRANSCRIBE_ERR_INVALID_ARG);
    ext.size = 4;
    CHECK(transcribe_diarize_stream_begin(f.session, &p) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    ext           = { sizeof(transcribe_ext), k_fake_kind };
    g_validate_st = TRANSCRIBE_ERR_INVALID_ARG;
    CHECK(transcribe_diarize_stream_begin(f.session, &p) == TRANSCRIBE_ERR_INVALID_ARG);
    g_validate_st = TRANSCRIBE_OK;
    CHECK(transcribe_diarize_n_segments(f.session) == 2);

    CHECK(transcribe_diarize_stream_begin(f.session, &p) == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_n_segments(f.session) == 0);
    CHECK(transcribe_diarize_stream_decided_ms(f.session) == 0);

    // Malformed feeds leave the stream as it was; an empty feed is a no-op.
    std::vector<float> bad(1600, 0.0f);
    bad[7] = std::numeric_limits<float>::infinity();
    CHECK(transcribe_diarize_stream_feed(f.session, bad.data(), 1600) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_stream_feed(f.session, nullptr, 1600) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_stream_feed(f.session, f.pcm.data(), -1) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_stream_feed(f.session, nullptr, 0) == TRANSCRIBE_OK);
    CHECK(g_stream_samples == 0);

    // 1600 samples: one frame decided (speaker 1 open at the decided end).
    CHECK(transcribe_diarize_stream_feed(f.session, f.pcm.data(), 1600) == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_stream_decided_ms(f.session) == 80);
    CHECK(transcribe_diarize_n_segments(f.session) == 1);
    // 3200: two frames; 4800: three frames, speaker 1 in frames 0 and 2.
    CHECK(transcribe_diarize_stream_feed(f.session, f.pcm.data(), 1600) == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_stream_feed(f.session, f.pcm.data(), 1600) == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_stream_decided_ms(f.session) == 240);
    CHECK(transcribe_diarize_n_segments(f.session) == 2);
    transcribe_speaker_segment row;
    transcribe_speaker_segment_init(&row);
    CHECK(transcribe_diarize_get_segment(f.session, 1, &row) == TRANSCRIBE_OK);
    CHECK(row.speaker_id == 1 && row.t0_ms == 160 && row.t1_ms == 240);

    // peek: decided frames plus the provisional one; the decided end stays,
    // a failed peek keeps the segments, and the next feed is back on the
    // decided frames.
    CHECK(transcribe_diarize_stream_peek(f.session) == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_n_segments(f.session) == 3);
    CHECK(transcribe_diarize_get_segment(f.session, 2, &row) == TRANSCRIBE_OK);
    CHECK(row.speaker_id == 2 && row.t0_ms == 240 && row.t1_ms == 320);
    CHECK(transcribe_diarize_stream_decided_ms(f.session) == 240);
    g_stream_status = TRANSCRIBE_ERR_ABORTED;
    CHECK(transcribe_diarize_stream_peek(f.session) == TRANSCRIBE_ERR_ABORTED);
    CHECK(transcribe_diarize_n_segments(f.session) == 3);
    g_stream_status = TRANSCRIBE_OK;
    CHECK(transcribe_diarize_stream_feed(f.session, f.pcm.data(), 100) == TRANSCRIBE_OK);  // still active
    CHECK(transcribe_diarize_n_segments(f.session) == 2);
    CHECK(transcribe_diarize_stream_decided_ms(f.session) == 240);

    CHECK(transcribe_diarize_stream_finish(f.session) == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_stream_decided_ms(f.session) == 320);
    CHECK(transcribe_diarize_n_segments(f.session) == 3);
    CHECK(transcribe_diarize_get_segment(f.session, 2, &row) == TRANSCRIBE_OK);
    CHECK(row.speaker_id == 2 && row.t0_ms == 240 && row.t1_ms == 320);
    // The stream is over; its result stays readable.
    CHECK(transcribe_diarize_stream_feed(f.session, f.pcm.data(), 1600) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_stream_finish(f.session) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_stream_peek(f.session) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_n_segments(f.session) == 3);

    // A run ends a stream in progress.
    CHECK(transcribe_diarize_stream_begin(f.session, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_stream_feed(f.session, f.pcm.data(), 1600) == TRANSCRIBE_OK);
    CHECK(f.run() == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_stream_decided_ms(f.session) == 0);
    CHECK(transcribe_diarize_stream_feed(f.session, f.pcm.data(), 1600) == TRANSCRIBE_ERR_INVALID_ARG);

    // A family failure in feed or finish ends the stream with an empty result.
    CHECK(transcribe_diarize_stream_begin(f.session, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_stream_feed(f.session, f.pcm.data(), 1600) == TRANSCRIBE_OK);
    g_stream_status = TRANSCRIBE_ERR_ABORTED;
    CHECK(transcribe_diarize_stream_feed(f.session, f.pcm.data(), 1600) == TRANSCRIBE_ERR_ABORTED);
    CHECK(transcribe_diarize_n_segments(f.session) == 0);
    CHECK(transcribe_diarize_stream_decided_ms(f.session) == 0);
    CHECK(transcribe_diarize_stream_finish(f.session) == TRANSCRIBE_ERR_INVALID_ARG);
    g_stream_status = TRANSCRIBE_OK;
    CHECK(transcribe_diarize_stream_begin(f.session, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_stream_feed(f.session, f.pcm.data(), 1600) == TRANSCRIBE_OK);
    g_stream_status = TRANSCRIBE_ERR_BACKEND;
    CHECK(transcribe_diarize_stream_finish(f.session) == TRANSCRIBE_ERR_BACKEND);
    CHECK(transcribe_diarize_n_segments(f.session) == 0);
    g_stream_status = TRANSCRIBE_OK;
}

}  // namespace

int main() {
    transcribe_log_set(nullptr, nullptr);

    test_probs_to_segments();
    test_role_checks();
    test_run_and_accessors();
    test_pre_clear_rejections_preserve_result();
    test_stream_unsupported();
    test_stream_calls();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("diarize_dispatch_unit: ok\n");
    return EXIT_SUCCESS;
}
