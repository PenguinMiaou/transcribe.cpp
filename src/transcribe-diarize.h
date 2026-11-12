// transcribe-diarize.h - internal DIARIZE role surface: the session base,
// the per-arch ops table, and the shared probs -> segments step.
//
// Families compute frame probabilities; the role dispatcher
// (transcribe-diarize.cpp) turns them into speaker segments.

#pragma once

#include "transcribe-session-core.h"
#include "transcribe/diarize.h"

#include <vector>

struct transcribe_diarize_session : transcribe::SessionCore {
    std::vector<transcribe::SpeakerSegmentEntry> segments;  // last successful run's rows

    // Push mode (transcribe_diarize_stream_*).
    bool    stream_active     = false;  // between a successful begin and finish / failure / run
    int64_t stream_decided_ms = 0;      // end of the last decided frame of the current / last stream
};

namespace transcribe {

// One run's family output: probs is row-major [n_frames, n_speakers].
struct DiarizeProbs {
    std::vector<float> probs;
    int                n_frames   = 0;
    int                n_speakers = 0;
    double             frame_ms   = 0.0;
};

// Borrowed view of frame probabilities, row-major [n_frames, n_speakers].
struct DiarizeProbsView {
    const float * probs      = nullptr;
    int           n_frames   = 0;
    int           n_speakers = 0;
    double        frame_ms   = 0.0;
};

struct DiarizeOps {
    int (*max_speakers)(const transcribe_model * model);
    transcribe_diarize_session * (*new_session)();
    // Pure pre-clear check of params->family (already kind/size-checked);
    // NULL = nothing to check.
    transcribe_status (*run_validate)(const transcribe_diarize_params * params);
    transcribe_status (*run)(transcribe_diarize_session *      session,
                             const float *                     pcm,
                             int                               n_samples,
                             const transcribe_diarize_params * params,
                             DiarizeProbs &                    out);
    // Push mode; all four NULL when the family has none. params is already
    // validated like run's. stream_feed advances every chunk whose result
    // cannot change with later audio and points `decided` at the frames
    // decided so far (owned by the session, valid until its next call);
    // stream_finish runs the rest as the tail of a whole-recording run and
    // hands over all frames. Feeding in any split gives run's frames.
    transcribe_status (*stream_begin)(transcribe_diarize_session * session, const transcribe_diarize_params * params);
    transcribe_status (*stream_feed)(transcribe_diarize_session * session,
                                     const float *                pcm,
                                     int                          n_samples,
                                     DiarizeProbsView &           decided);
    transcribe_status (*stream_finish)(transcribe_diarize_session * session, DiarizeProbs & out);
    // The frames stream_finish would hand over if the recording ended now,
    // computed on a copy: the stream itself is left exactly as it was.
    transcribe_status (*stream_peek)(transcribe_diarize_session * session, DiarizeProbs & out);
};

// Append one segment per contiguous run of probs > 0.5 per speaker,
// speaker-major and time-ordered (speaker_id 1-based, p = NaN).
void probs_to_segments(const float *                      probs,
                       int                                n_frames,
                       int                                n_speakers,
                       double                             frame_ms,
                       std::vector<SpeakerSegmentEntry> & out);

}  // namespace transcribe
