/*
 * include/transcribe/nemotron3_diar.h - Nemotron-3-Diarization public
 * extension.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs.
 * Nemotron-3-Diarization (up to 8 speakers, 10 ms output) serves the DIARIZE
 * role (include/transcribe/diarize.h). Its one extension picks the streaming
 * operating point for transcribe_diarize_run; probe with
 * transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_DIARIZE_RUN,
 * TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_DIARIZE).
 *
 * The compute core is streaming (arrival-order speaker cache + FIFO);
 * transcribe_diarize_run drives it over the whole recording in one call.
 *
 * FourCC kinds are reserved in docs/extension-kinds.md.
 */

#ifndef TRANSCRIBE_NEMOTRON3_DIAR_H
#define TRANSCRIBE_NEMOTRON3_DIAR_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'N3DR' little-endian = 0x5244334E (DIARIZE_RUN slot) */
#define TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_DIARIZE 0x5244334Eu

/*
 * Streaming operating point (latency / accuracy trade-off).
 *
 * The model processes audio in fixed chunks with a fixed lookahead and
 * carries speaker identity across chunks in a bounded cache. Each named
 * preset is a jointly-tuned bundle published on the upstream model card
 * (chunk length, lookahead, FIFO and speaker-cache geometry); the menu is
 * discrete, not a continuous latency dial. Latency = chunk + lookahead.
 *
 *   DEFAULT             The GGUF-shipped configuration (= VERY_HIGH_LATENCY).
 *   VERY_HIGH_LATENCY   30.4 s (chunk 340 + lookahead 40 frames @ 80 ms).
 *                       Highest accuracy; the offline operating point.
 *   LOW_LATENCY         1.04 s (chunk 9 + lookahead 4).
 *   VERY_LOW_LATENCY    0.64 s (chunk 6 + lookahead 2).
 *   ULTRA_LOW_LATENCY   0.32 s (chunk 3 + lookahead 1).
 *
 * Smaller chunks cost more compute per audio second (the whole speaker
 * cache is re-encoded every chunk). Values outside the enum range are
 * rejected by transcribe_diarize_run with TRANSCRIBE_ERR_INVALID_ARG before
 * the previous result is cleared.
 */
typedef enum {
    TRANSCRIBE_NEMOTRON3_DIAR_PRESET_DEFAULT           = 0,
    TRANSCRIBE_NEMOTRON3_DIAR_PRESET_VERY_HIGH_LATENCY = 1,
    TRANSCRIBE_NEMOTRON3_DIAR_PRESET_LOW_LATENCY       = 2,
    TRANSCRIBE_NEMOTRON3_DIAR_PRESET_VERY_LOW_LATENCY  = 3,
    TRANSCRIBE_NEMOTRON3_DIAR_PRESET_ULTRA_LOW_LATENCY = 4,
} transcribe_nemotron3_diar_preset;

/* transcribe_diarize_params::family: the operating point for
 * transcribe_diarize_run. */
struct transcribe_nemotron3_diar_diarize_ext {
    struct transcribe_ext            ext;
    transcribe_nemotron3_diar_preset preset;
};

/* Fills ext.size/kind and preset = DEFAULT. */
TRANSCRIBE_API void transcribe_nemotron3_diar_diarize_ext_init(struct transcribe_nemotron3_diar_diarize_ext * ext);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_NEMOTRON3_DIAR_H */
