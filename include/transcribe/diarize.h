/*
 * include/transcribe/diarize.h - DIARIZE role: who spoke when.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. For models whose
 * product is speaker turns rather than text (transcribe_model_roles() has
 * TRANSCRIBE_ROLE_DIARIZE). ASR models that attribute speakers inside a
 * transcript keep using transcribe_run_params::diarize instead.
 *
 * Open a transcribe_diarize_session on a loaded model, run, then read the
 * speaker segments. A recording still in progress is fed piece by piece
 * with transcribe_diarize_stream_*. Threading and lifetime rules:
 * docs/roles.md.
 */

#ifndef TRANSCRIBE_DIARIZE_H
#define TRANSCRIBE_DIARIZE_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

struct transcribe_diarize_session;

/* Static facts about a diarization model. */
struct transcribe_diarize_info {
    uint64_t struct_size;
    int32_t  sample_rate;  /* input PCM rate (16000) */
    int32_t  max_speakers; /* speaker_id is in [1, max_speakers] */
};

struct transcribe_diarize_session_params {
    uint64_t struct_size;
    int32_t  n_threads; /* 0 = library default */
};

struct transcribe_diarize_params {
    uint64_t                      struct_size;
    /* Typed family extension on TRANSCRIBE_EXT_SLOT_DIARIZE_RUN, or NULL. */
    const struct transcribe_ext * family;
};

TRANSCRIBE_API void transcribe_diarize_info_init(struct transcribe_diarize_info * out);
TRANSCRIBE_API void transcribe_diarize_session_params_init(struct transcribe_diarize_session_params * params);
TRANSCRIBE_API void transcribe_diarize_params_init(struct transcribe_diarize_params * params);

/* UNSUPPORTED_ROLE when the model does not serve TRANSCRIBE_ROLE_DIARIZE. */
TRANSCRIBE_API transcribe_status transcribe_diarize_get_info(const struct transcribe_model *  model,
                                                             struct transcribe_diarize_info * out);

/* params may be NULL for defaults. UNSUPPORTED_ROLE when the model does not
 * serve TRANSCRIBE_ROLE_DIARIZE. On failure *out_session is NULL. */
TRANSCRIBE_API transcribe_status
transcribe_diarize_session_init(struct transcribe_model *                        model,
                                const struct transcribe_diarize_session_params * params,
                                struct transcribe_diarize_session **             out_session);

/* NULL is a no-op. */
TRANSCRIBE_API void transcribe_diarize_session_free(struct transcribe_diarize_session * session);

/* Polled during a run; returning true stops it with TRANSCRIBE_ERR_ABORTED. */
TRANSCRIBE_API void transcribe_diarize_set_abort_callback(struct transcribe_diarize_session * session,
                                                          transcribe_abort_callback           cb,
                                                          void *                              user_data);

/*
 * Diarize one recording: 16 kHz mono float32 PCM, every sample finite.
 * params may be NULL for defaults. Replaces the previous result. Malformed
 * input (NULL pointers, n_samples <= 0, NaN / Inf, a bad extension) returns
 * an error before the previous result is touched.
 */
TRANSCRIBE_API transcribe_status transcribe_diarize_run(struct transcribe_diarize_session *      session,
                                                        const float *                            pcm,
                                                        int                                      n_samples,
                                                        const struct transcribe_diarize_params * params);

/*
 * Push mode: diarize one recording as its audio arrives. The result of
 * begin + feed(piece)... + finish is that of transcribe_diarize_run on the
 * concatenated pieces, for any split, and speaker ids are stable from the
 * first feed to the end.
 *
 * transcribe_diarize_stream_begin starts a recording; params as for
 * transcribe_diarize_run (NULL for defaults). It replaces the previous
 * result and any stream in progress; a bad params struct or extension
 * returns an error before either is touched. NOT_IMPLEMENTED (from all
 * three calls) when the model has no push mode.
 *
 * transcribe_diarize_stream_feed appends 16 kHz mono float32 PCM (every
 * sample finite) and computes every frame that later audio can no longer
 * change. The segment accessors then describe the frames decided so far,
 * [0, transcribe_diarize_stream_decided_ms): a segment that ends at the
 * decided end may still grow, earlier ones are final. The decided end
 * trails the fed audio by the model's look-ahead (chunk plus right context
 * of the preset in use) and moves in whole chunks. n_samples == 0 is a
 * no-op. INVALID_ARG without an active stream or for malformed input
 * (negative n_samples, NULL pcm, NaN / Inf), leaving the stream as it was.
 *
 * transcribe_diarize_stream_finish ends the recording: the rest of the fed
 * audio is processed and the segments are the final ones. INVALID_ARG
 * without an active stream.
 *
 * The abort callback is polled during feed and finish. A feed or finish
 * that fails after input validation (aborted, backend error) ends the
 * stream with an empty result; start again with begin.
 */
TRANSCRIBE_API transcribe_status transcribe_diarize_stream_begin(struct transcribe_diarize_session *      session,
                                                                 const struct transcribe_diarize_params * params);
TRANSCRIBE_API transcribe_status transcribe_diarize_stream_feed(struct transcribe_diarize_session * session,
                                                                const float *                       pcm,
                                                                int                                 n_samples);
TRANSCRIBE_API transcribe_status transcribe_diarize_stream_finish(struct transcribe_diarize_session * session);

/* End time in ms of the last decided frame of the current stream, or of the
 * last finished one; 0 after begin, after a run, and after a failed feed or
 * finish. */
TRANSCRIBE_API int64_t transcribe_diarize_stream_decided_ms(const struct transcribe_diarize_session * session);

/*
 * Speaker segments of the last run or of the stream so far, grouped by
 * speaker and time-ordered within a speaker; segments of different speakers
 * may overlap. 0 before any run, and 0 after a run that passed input
 * validation but then failed (aborted, backend error). An out-of-range index
 * returns OK with a zeroed row.
 */
TRANSCRIBE_API int               transcribe_diarize_n_segments(const struct transcribe_diarize_session * session);
TRANSCRIBE_API transcribe_status transcribe_diarize_get_segment(const struct transcribe_diarize_session * session,
                                                                int                                       i,
                                                                struct transcribe_speaker_segment *       out);

/* load_ms plus the last run's stage times (summed over the feeds of a stream). */
TRANSCRIBE_API transcribe_status transcribe_diarize_get_timings(const struct transcribe_diarize_session * session,
                                                                struct transcribe_timings *               out);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_DIARIZE_H */
