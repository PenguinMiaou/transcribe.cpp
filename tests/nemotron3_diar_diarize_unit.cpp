// nemotron3_diar_diarize_unit.cpp - Nemotron-3-Diarization on the DIARIZE
// role, against a real GGUF (TRANSCRIBE_NEMOTRON3_DIAR_GGUF; RC 77 skip when
// unset).

#include "transcribe.h"
#include "transcribe/diarize.h"
#include "transcribe/nemotron3_diar.h"
#include "wav.h"

#include <cstdio>
#include <cstdlib>
#include <set>
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

struct Row {
    int64_t t0, t1;
    int32_t spk;

    bool operator==(const Row & o) const { return t0 == o.t0 && t1 == o.t1 && spk == o.spk; }
};

std::vector<Row> diarize_rows(const transcribe_diarize_session * s) {
    std::vector<Row> rows;
    for (int i = 0; i < transcribe_diarize_n_segments(s); ++i) {
        transcribe_speaker_segment r;
        transcribe_speaker_segment_init(&r);
        transcribe_diarize_get_segment(s, i, &r);
        rows.push_back({ r.t0_ms, r.t1_ms, r.speaker_id });
    }
    return rows;
}

std::vector<Row> diarize(transcribe_diarize_session *     s,
                         const std::vector<float> &       pcm,
                         transcribe_nemotron3_diar_preset preset) {
    transcribe_nemotron3_diar_diarize_ext ext;
    transcribe_nemotron3_diar_diarize_ext_init(&ext);
    ext.preset = preset;
    transcribe_diarize_params dp;
    transcribe_diarize_params_init(&dp);
    dp.family = &ext.ext;
    CHECK(transcribe_diarize_run(s, pcm.data(), static_cast<int>(pcm.size()), &dp) == TRANSCRIBE_OK);
    return diarize_rows(s);
}

bool abort_now(void *) {
    return true;
}

}  // namespace

int main() {
    const char * path = std::getenv("TRANSCRIBE_NEMOTRON3_DIAR_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::fprintf(stderr, "nemotron3_diar_diarize_unit: TRANSCRIBE_NEMOTRON3_DIAR_GGUF not set; skipping.\n");
        return 77;
    }
    std::vector<float> pcm;
    std::string        err;
    if (!transcribe_cli::load_wav_mono_16k(std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/nemotron3-diar-8spk-mix.wav",
                                           pcm, err)) {
        std::fprintf(stderr, "nemotron3_diar_diarize_unit: wav: %s\n", err.c_str());
        return 77;
    }
    ::unsetenv("TRANSCRIBE_NEMOTRON3_DIAR_PRESET");

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend               = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * model = nullptr;
    if (transcribe_model_load_file(path, &mp, &model) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL: model load\n");
        return EXIT_FAILURE;
    }

    // 1. Roles, info, extension probe.
    CHECK(transcribe_model_roles(model) == TRANSCRIBE_ROLE_DIARIZE);
    CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_CANCELLATION));
    transcribe_diarize_info info;
    transcribe_diarize_info_init(&info);
    CHECK(transcribe_diarize_get_info(model, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000 && info.max_speakers == 8);
    CHECK(transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_DIARIZE_RUN,
                                            TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_DIARIZE));
    CHECK(
        !transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_DIARIZE));
    CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                             TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_DIARIZE));

    transcribe_diarize_session * dia = nullptr;
    CHECK(transcribe_diarize_session_init(model, nullptr, &dia) == TRANSCRIBE_OK);

    // 2. Golden segments (ms, speaker) for the 8-speaker oracle mix at the
    //    default preset, recorded on CPU with the BF16 reference GGUF (the
    //    rows of tests/golden/batch/Nemotron-3-Diarization.cpu.json). The
    //    low-latency preset still finds all 8 speakers.
    const std::vector<Row> golden_default = {
        { 0,     1870,  1 },
        { 2960,  4130,  1 },
        { 43580, 48490, 1 },
        { 88200, 90110, 1 },
        { 91140, 92000, 1 },
        { 5500,  9530,  2 },
        { 54500, 57980, 2 },
        { 58290, 59480, 2 },
        { 9980,  16000, 3 },
        { 48980, 54000, 3 },
        { 16500, 19100, 4 },
        { 19170, 21470, 4 },
        { 80980, 82060, 4 },
        { 82520, 86990, 4 },
        { 21990, 24680, 5 },
        { 25220, 26980, 5 },
        { 71020, 74800, 5 },
        { 75380, 76000, 5 },
        { 26560, 28000, 6 },
        { 29450, 30000, 6 },
        { 30790, 32000, 6 },
        { 65520, 67310, 6 },
        { 67690, 70500, 6 },
        { 32530, 37330, 7 },
        { 60000, 64960, 7 },
        { 37970, 43000, 8 },
        { 76470, 81600, 8 },
    };
    CHECK(diarize(dia, pcm, TRANSCRIBE_NEMOTRON3_DIAR_PRESET_DEFAULT) == golden_default);
    const std::vector<Row> low = diarize(dia, pcm, TRANSCRIBE_NEMOTRON3_DIAR_PRESET_LOW_LATENCY);
    {
        std::set<int32_t> ids;
        for (const Row & r : low) {
            ids.insert(r.spk);
        }
        CHECK(ids.size() == 8);
    }

    // 3. Ext preset == env preset (the validation hook route); NULL params
    //    and a DEFAULT ext agree.
    ::setenv("TRANSCRIBE_NEMOTRON3_DIAR_PRESET", "low_latency", 1);
    CHECK(transcribe_diarize_run(dia, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    CHECK(diarize_rows(dia) == low);
    ::unsetenv("TRANSCRIBE_NEMOTRON3_DIAR_PRESET");
    CHECK(transcribe_diarize_run(dia, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    CHECK(diarize_rows(dia) == golden_default);
    CHECK(diarize(dia, pcm, TRANSCRIBE_NEMOTRON3_DIAR_PRESET_VERY_HIGH_LATENCY) == golden_default);

    // 4. Pre-clear rejection of a bad preset keeps the previous result.
    const int                             n_before = transcribe_diarize_n_segments(dia);
    transcribe_nemotron3_diar_diarize_ext bad;
    transcribe_nemotron3_diar_diarize_ext_init(&bad);
    bad.preset = static_cast<transcribe_nemotron3_diar_preset>(99);
    transcribe_diarize_params dp;
    transcribe_diarize_params_init(&dp);
    dp.family = &bad.ext;
    CHECK(transcribe_diarize_run(dia, pcm.data(), static_cast<int>(pcm.size()), &dp) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_n_segments(dia) == n_before);

    // 5. Sub-hop input (< 160 samples) is an empty OK result; an abort
    //    returns ERR_ABORTED.
    CHECK(transcribe_diarize_run(dia, pcm.data(), 100, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_diarize_n_segments(dia) == 0);
    transcribe_diarize_set_abort_callback(dia, abort_now, nullptr);
    CHECK(transcribe_diarize_run(dia, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_ERR_ABORTED);
    transcribe_diarize_set_abort_callback(dia, nullptr, nullptr);

    transcribe_diarize_session_free(dia);
    transcribe_model_free(model);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("nemotron3_diar_diarize_unit: ok\n");
    return EXIT_SUCCESS;
}
