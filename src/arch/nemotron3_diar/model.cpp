// arch/nemotron3_diar/model.cpp - Nemotron-3-Diarization load / graph /
// streaming driver / Arch instance.
//
// Every product path runs NeMo's synchronous streaming loop
// (SortformerEncLabelModel.forward_streaming, streaming_mode=True,
// async_streaming=False): per chunk, one graph computes
//   chunk_embs = FeatureStacking(chunk mel)                       [T_diar, 512]
//   x = [spkcache | fifo | chunk_embs] -> embed_norm -> 31 pre-LN RoPE
//       blocks -> final_norm -> encoder_proj -> subpixel x8 -> head
//   probs = sigmoid(logits)                                     [8*T_cat, 8]
// and the host state machine (stream.cpp) updates the cache/FIFO from the
// x8 average-pooled probs. transcribe_diarize_run drives the per-chunk
// driver (advance_chunks) over the whole recording.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "nemotron3_diar.h"
#include "transcribe-abi.h"
#include "transcribe-arch.h"
#include "transcribe-backend.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-mel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace transcribe::nemotron3_diar {

extern const Arch arch;

static constexpr char k_default_variant[] = "Nemotron-3-Diarization";

Nemotron3DiarModel::~Nemotron3DiarModel() {
    if (f32_ctx != nullptr) {
        ggml_free(f32_ctx);
    }
    if (f32_buffer != nullptr) {
        transcribe::safe_buffer_free(f32_buffer);
    }
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
    }
    weight_storage.reset();
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        transcribe::safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary = nullptr;
}

namespace {

bool env_set(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

// Upcast weight slots to exact F32 copies on `backend` and repoint the slots
// (see the promotion policy in load()).
transcribe_status promote_to_f32(const std::vector<ggml_tensor **> & slots,
                                 ggml_backend_t                      backend,
                                 ggml_context **                     out_ctx,
                                 ggml_backend_buffer_t *             out_buffer) {
    if (slots.empty()) {
        return TRANSCRIBE_OK;
    }
    ggml_init_params ip{ slots.size() * ggml_tensor_overhead() + 1024, nullptr, /*no_alloc=*/true };
    *out_ctx = ggml_init(ip);
    if (*out_ctx == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    std::vector<ggml_tensor *> dst(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        const ggml_tensor * src = *slots[i];
        dst[i]                  = ggml_new_tensor(*out_ctx, GGML_TYPE_F32, ggml_n_dims(src), src->ne);
        ggml_set_name(dst[i], ggml_get_name(src));
    }
    *out_buffer = ggml_backend_alloc_ctx_tensors(*out_ctx, backend);
    if (*out_buffer == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_buffer_set_usage(*out_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<uint8_t> raw;
    std::vector<float>   f32;
    for (size_t i = 0; i < slots.size(); ++i) {
        ggml_tensor * src = *slots[i];
        const int64_t n   = ggml_nelements(src);
        raw.resize(ggml_nbytes(src));
        ggml_backend_tensor_get(src, raw.data(), 0, raw.size());
        f32.resize(static_cast<size_t>(n));
        const auto * tt = ggml_get_type_traits(src->type);
        if (src->type == GGML_TYPE_F32) {
            std::memcpy(f32.data(), raw.data(), raw.size());
        } else if (tt != nullptr && tt->to_float != nullptr) {
            tt->to_float(raw.data(), f32.data(), n);
        } else {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: cannot upcast %s (%s)", ggml_get_name(src),
                    ggml_type_name(src->type));
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_tensor_set(dst[i], f32.data(), 0, f32.size() * sizeof(float));
        *slots[i] = dst[i];
    }
    return TRANSCRIBE_OK;
}

// ---- graph helpers ----

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    return ggml_add(ctx, x, b);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * W, ggml_tensor * x, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, W, x);
    return b != nullptr ? ggml_add(ctx, y, b) : y;
}

// Pre-LN RoPE block (TransformerBlock.forward). x ne = [d, T, B].
ggml_tensor * build_block(ggml_context *               ctx,
                          const Nemotron3DiarHParams & hp,
                          const Nemotron3DiarBlock &   b,
                          ggml_tensor *                x,
                          ggml_tensor *                positions) {
    const int64_t d  = hp.enc_d_model;
    const int64_t H  = hp.enc_n_heads;
    const int64_t D  = hp.head_dim();
    const int64_t T  = x->ne[1];
    const int64_t Bn = x->ne[2];

    // ---- attention ----
    ggml_tensor * h   = layer_norm(ctx, x, b.norm1_w, b.norm1_b, hp.enc_ln_eps);
    ggml_tensor * qkv = ggml_mul_mat(ctx, b.attn_qkv_w, h);  // [3d, T, B], rows [q | k | v]

    // NeMo: w_qkv(x).view(B, T, 3, H, D) -> q/k/v each head-major [D, H, T, B].
    // q and k are adjacent in every qkv row, so one strided view covers both
    // ([D, 2H, T, B]) and a single RoPE rotates them: rotate_half (NEOX) over
    // all D dims, positions 0..T-1 over the concat (t_q == t_k, no cache
    // offset), theta rope_base. Reading strided views instead of contiguous
    // copies is value-identical (same per-element math, same dot products).
    const size_t  es = ggml_element_size(qkv);
    ggml_tensor * qk = ggml_view_4d(ctx, qkv, D, 2 * H, T, Bn, D * es, qkv->nb[1], qkv->nb[2], 0);
    qk = ggml_rope_ext(ctx, qk, positions, nullptr, static_cast<int>(D), GGML_ROPE_TYPE_NEOX, 0, hp.enc_rope_base, 1.0f,
                       0.0f, 1.0f, 0.0f, 0.0f);  // contiguous [D, 2H, T, B]
    // [D, T, H, B] views into the rotated q|k (heads 0..H-1 = q, H..2H-1 = k).
    ggml_tensor * q = ggml_view_4d(ctx, qk, D, T, H, Bn, qk->nb[2], qk->nb[1], qk->nb[3], 0);
    ggml_tensor * k = ggml_view_4d(ctx, qk, D, T, H, Bn, qk->nb[2], qk->nb[1], qk->nb[3], H * qk->nb[1]);

    // Full attention (sync batch 1: no padding mask), scale 1/sqrt(D).
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);  // [T_k, T_q, H, B]
    kq               = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / std::sqrt(static_cast<float>(D)), 0.0f);
    // v^T [T, D, H, B] straight from the qkv view in one copy.
    ggml_tensor * v =
        ggml_view_4d(ctx, qkv, D, H, T, Bn, D * es, qkv->nb[1], qkv->nb[2], static_cast<size_t>(2) * d * es);
    ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [T, D, H, B]
    ggml_tensor * o   = ggml_mul_mat(ctx, v_t, kq);                        // [D, T_q, H, B]
    o                 = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));  // [D, H, T, B]
    o                 = ggml_reshape_3d(ctx, o, d, T, Bn);
    o                 = linear(ctx, b.attn_out_w, o, b.attn_out_b);
    x                 = ggml_add(ctx, x, o);

    // ---- FFN: Linear -> GELU (exact erf) -> Linear ----
    ggml_tensor * f = layer_norm(ctx, x, b.norm2_w, b.norm2_b, hp.enc_ln_eps);
    f               = linear(ctx, b.ff_in_w, f, b.ff_in_b);
    f               = ggml_gelu_erf(ctx, f);
    f               = linear(ctx, b.ff_out_w, f, b.ff_out_b);
    return ggml_add(ctx, x, f);
}

// One streaming step for B utterances sharing the same step geometry.
struct StepBuild {
    ggml_cgraph * graph     = nullptr;
    ggml_tensor * prev_in   = nullptr;  // [d, S+F, B] (null when S+F == 0)
    ggml_tensor * mel_in    = nullptr;  // [sub*n_mels, T_diar, B]
    ggml_tensor * positions = nullptr;  // I32 [T_cat]
    ggml_tensor * embs      = nullptr;  // [d, T_diar, B]
    ggml_tensor * probs     = nullptr;  // [n_spk, up*T_cat, B]

    // Encoder-stage parity tensors (step 0 dumps).
    ggml_tensor *                              embed_norm = nullptr;
    ggml_tensor *                              final_norm = nullptr;
    ggml_tensor *                              enc_proj   = nullptr;
    ggml_tensor *                              sub_conv   = nullptr;
    ggml_tensor *                              upsampled  = nullptr;
    ggml_tensor *                              logits     = nullptr;
    std::vector<std::pair<int, ggml_tensor *>> layer_out;
};

StepBuild build_step_graph(ggml_context *               ctx,
                           const Nemotron3DiarHParams & hp,
                           const Nemotron3DiarWeights & w,
                           int                          n_prev,
                           int                          T_diar,
                           int                          Bn,
                           bool                         want_layers) {
    StepBuild     sb;
    const int64_t d     = hp.enc_d_model;
    const int64_t fin   = static_cast<int64_t>(hp.enc_feat_in) * hp.enc_subsampling_factor;
    const int64_t T_cat = n_prev + T_diar;
    const int64_t hd    = hp.head_d_model;
    const int64_t up    = hp.upsample_factor;

    sb.mel_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, fin, T_diar, Bn);
    ggml_set_name(sb.mel_in, "step.mel.in");
    ggml_set_input(sb.mel_in);

    // FeatureStacking: 8 consecutive 128-dim mel frames per row (frame-major),
    // stacked on the host; Linear 1024 -> 512, no bias.
    sb.embs = ggml_mul_mat(ctx, w.pre_encode_proj_w, sb.mel_in);  // [d, T_diar, B]

    ggml_tensor * x = sb.embs;
    if (n_prev > 0) {
        sb.prev_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_prev, Bn);
        ggml_set_name(sb.prev_in, "step.prev.in");
        ggml_set_input(sb.prev_in);
        x = ggml_concat(ctx, sb.prev_in, sb.embs, 1);  // [d, T_cat, B]
    }

    sb.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_cat);
    ggml_set_name(sb.positions, "step.positions");
    ggml_set_input(sb.positions);

    // rope branch of forward_internal: no xscale; embed_norm (pre_block_norm).
    x             = layer_norm(ctx, x, w.embed_norm_w, w.embed_norm_b, hp.enc_ln_eps);
    sb.embed_norm = x;
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        x = build_block(ctx, hp, w.blocks[static_cast<size_t>(i)], x, sb.positions);
        if (want_layers) {
            sb.layer_out.emplace_back(i, x);
        }
    }
    x             = layer_norm(ctx, x, w.final_norm_w, w.final_norm_b, hp.enc_ln_eps);
    sb.final_norm = x;

    // frontend_encoder: encoder_proj 512 -> 192.
    ggml_tensor * e = linear(ctx, w.enc_proj_w, x, w.enc_proj_b);  // [hd, T_cat, B]
    sb.enc_proj     = e;

    // upsample_hidden: Conv1d(hd, hd*up, k=3, pad=1) over time, then
    // (T, up, hd) -> (up*T, hd). im2col (F32) + mul_mat keeps channels in ne0.
    ggml_tensor * e_t  = ggml_cont(ctx, ggml_permute(ctx, e, 1, 0, 2, 3));                             // [T_cat, hd, B]
    ggml_tensor * cols = ggml_im2col(ctx, w.upsample_w, e_t, 1, 0, 1, 0, 1, 0, false, GGML_TYPE_F32);  // [3*hd, T, B]
    ggml_tensor * kern =
        ggml_reshape_2d(ctx, w.upsample_w, w.upsample_w->ne[0] * w.upsample_w->ne[1], w.upsample_w->ne[2]);
    ggml_tensor * c = ggml_mul_mat(ctx, kern, cols);  // [hd*up, T_cat, B]
    c               = ggml_add(ctx, c, w.upsample_b);
    sb.sub_conv     = c;
    // Channel block j of frame t is output frame up*t + j: a pure reshape.
    ggml_tensor * u = ggml_reshape_3d(ctx, c, hd, up * T_cat, Bn);
    sb.upsampled    = u;

    // forward_speaker_logits: relu -> fc1 -> relu -> single_hidden_to_spks.
    ggml_tensor * h = ggml_relu(ctx, u);
    h               = linear(ctx, w.fc1_w, h, w.fc1_b);
    h               = ggml_relu(ctx, h);
    sb.logits       = linear(ctx, w.spk_head_w, h, w.spk_head_b);  // [n_spk, up*T_cat, B]
    sb.probs        = ggml_sigmoid(ctx, sb.logits);

    // Both are read back after compute: keep the allocator from reusing
    // their buffers for later nodes.
    ggml_set_output(sb.embs);
    ggml_set_output(sb.probs);

    sb.graph = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(sb.graph, sb.probs);
    ggml_build_forward_expand(sb.graph, sb.embs);
    return sb;
}

transcribe_status ensure_sched(transcribe::SessionCore * s, Nemotron3DiarModel * pm) {
    if (s->sched == nullptr) {
        s->sched = ggml_backend_sched_new(pm->plan.scheduler_list.data(), nullptr,
                                          static_cast<int>(pm->plan.scheduler_list.size()),
                                          /*graph_size=*/16384, /*parallel=*/false, /*op_offload=*/true);
        if (s->sched == nullptr) {
            return TRANSCRIBE_ERR_BACKEND;
        }
    }
    return TRANSCRIBE_OK;
}

// Chunk geometry of the next step (NeMo streaming_feat_loader).
struct ChunkGeom {
    int64_t win_lo = 0, win_hi = 0, end = 0;
    int     lc = 0, rc = 0, T_diar = 0;
};

// Geometry of the next chunk of `u`, or false when the recording (feat_len
// mel frames) is exhausted.
bool next_chunk(const Utterance & u, int sub, int64_t feat_len, ChunkGeom & g) {
    const StreamParams & P = u.P;
    if (u.stt >= feat_len) {
        return false;
    }
    const int64_t left  = std::min<int64_t>(static_cast<int64_t>(P.chunk_left_context) * sub, u.stt);
    g.end               = std::min<int64_t>(u.stt + static_cast<int64_t>(P.chunk_len) * sub, feat_len);
    const int64_t right = std::min<int64_t>(static_cast<int64_t>(P.chunk_right_context) * sub, feat_len - g.end);
    g.win_lo            = u.stt - left;
    g.win_hi            = g.end + right;
    g.lc                = static_cast<int>((left + sub / 2) / sub);   // round (left is a multiple of sub)
    g.rc                = static_cast<int>((right + sub - 1) / sub);  // ceil
    g.T_diar            = static_cast<int>((g.win_hi - g.win_lo + sub - 1) / sub);
    return true;
}

// Run chunks for a group of utterances in lockstep (identical step
// geometry is required: same S, F, T_diar, lc, rc). Returns after one step
// for every member. The DIARIZE role drives groups of one.
transcribe_status run_step(Nemotron3DiarSession *           pc,
                           Nemotron3DiarModel *             pm,
                           const std::vector<Utterance *> & group,
                           const std::vector<ChunkGeom> &   geo,
                           bool                             dump_encoder) {
    const Nemotron3DiarHParams & hp  = pm->hparams;
    const Nemotron3DiarWeights & w   = pm->weights;
    const int                    d   = hp.enc_d_model;
    const int                    sub = hp.enc_subsampling_factor;
    const int                    nm  = hp.fe_num_mels;
    const int                    ns  = hp.max_speakers;
    const int                    up  = hp.upsample_factor;
    const int                    Bn  = static_cast<int>(group.size());
    const int                    S   = group[0]->st.spkcache_n;
    const int                    F   = group[0]->st.fifo_n;
    const int                    Td  = geo[0].T_diar;
    const int                    Tc  = S + F + Td;

    if (pc->compute_ctx != nullptr) {
        ggml_free(pc->compute_ctx);
        pc->compute_ctx = nullptr;
    }
    ggml_init_params ip{ 32 * 1024 * 1024, nullptr, /*no_alloc=*/true };
    pc->compute_ctx = ggml_init(ip);
    if (pc->compute_ctx == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    StepBuild sb = build_step_graph(pc->compute_ctx, hp, w, S + F, Td, Bn, dump_encoder);
    if (dump_encoder) {
        for (auto & lo : sb.layer_out) {
            ggml_build_forward_expand(sb.graph, lo.second);
            transcribe::debug::mark_tensor_for_dump(lo.second);
        }
        for (ggml_tensor * t : { sb.embed_norm, sb.final_norm, sb.enc_proj, sb.sub_conv, sb.upsampled, sb.logits }) {
            ggml_build_forward_expand(sb.graph, t);
            transcribe::debug::mark_tensor_for_dump(t);
        }
        transcribe::debug::mark_tensor_for_dump(sb.embs);
    }
    if (const transcribe_status st = ensure_sched(pc, pm); st != TRANSCRIBE_OK) {
        return st;
    }
    ggml_backend_sched_reset(pc->sched);
    if (!TRANSCRIBE_ALLOC_SCHED_GRAPH_EXTENT(pc->sched, sb.graph, "nemotron3_diar", Tc)) {
        return TRANSCRIBE_ERR_OOM;
    }

    // Inputs: stacked chunk mel (zero-padded to a multiple of sub, like
    // FeatureStacking's F.pad), previous cache/FIFO embeddings, positions.
    const size_t row = static_cast<size_t>(sub) * nm;
    pc->step_mel.assign(row * Td * Bn, 0.0f);
    for (int b = 0; b < Bn; ++b) {
        const Utterance & u = *group[static_cast<size_t>(b)];
        const int64_t     M = geo[static_cast<size_t>(b)].win_hi - geo[static_cast<size_t>(b)].win_lo;
        const float * src = u.mel_tm.data() + static_cast<size_t>(geo[static_cast<size_t>(b)].win_lo - u.mel_base) * nm;
        std::copy(src, src + static_cast<size_t>(M) * nm, pc->step_mel.begin() + row * Td * b);
    }
    ggml_backend_tensor_set(sb.mel_in, pc->step_mel.data(), 0, pc->step_mel.size() * sizeof(float));
    if (sb.prev_in != nullptr) {
        pc->step_prev.resize(static_cast<size_t>(S + F) * d * Bn);
        for (int b = 0; b < Bn; ++b) {
            const StreamState & st  = group[static_cast<size_t>(b)]->st;
            float *             dst = pc->step_prev.data() + static_cast<size_t>(S + F) * d * b;
            std::copy(st.spkcache.begin(), st.spkcache.begin() + static_cast<size_t>(S) * d, dst);
            std::copy(st.fifo.begin(), st.fifo.begin() + static_cast<size_t>(F) * d, dst + static_cast<size_t>(S) * d);
        }
        ggml_backend_tensor_set(sb.prev_in, pc->step_prev.data(), 0, pc->step_prev.size() * sizeof(float));
    }
    pc->positions.resize(static_cast<size_t>(Tc));
    for (int i = 0; i < Tc; ++i) {
        pc->positions[static_cast<size_t>(i)] = i;
    }
    ggml_backend_tensor_set(sb.positions, pc->positions.data(), 0, pc->positions.size() * sizeof(int32_t));

    transcribe::configure_sched_n_threads(pc->sched, pc->n_threads);
    if (ggml_backend_sched_graph_compute(pc->sched, sb.graph) != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: step compute failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    if (dump_encoder) {
        transcribe::debug::dump_tensor("enc.pre_encode.out", sb.embs, "encoder");
        transcribe::debug::dump_tensor("enc.embed_norm.out", sb.embed_norm, "encoder");
        for (auto & lo : sb.layer_out) {
            char name[48];
            std::snprintf(name, sizeof(name), "enc.layers.%d.out", lo.first);
            transcribe::debug::dump_tensor(name, lo.second, "encoder");
        }
        transcribe::debug::dump_tensor("enc.final_norm.out", sb.final_norm, "encoder");
        transcribe::debug::dump_tensor("diar.encoder_proj.out", sb.enc_proj, "encoder");
        transcribe::debug::dump_tensor("diar.subpixel_conv.out", sb.sub_conv, "encoder");
        transcribe::debug::dump_tensor("diar.upsample.out", sb.upsampled, "encoder");
        transcribe::debug::dump_tensor("diar.logits", sb.logits, "encoder");
    }

    pc->step_embs.resize(static_cast<size_t>(Td) * d * Bn);
    ggml_backend_tensor_get(sb.embs, pc->step_embs.data(), 0, pc->step_embs.size() * sizeof(float));
    const size_t hi_per = static_cast<size_t>(up) * Tc * ns;
    pc->step_hi.resize(hi_per * Bn);
    ggml_backend_tensor_get(sb.probs, pc->step_hi.data(), 0, pc->step_hi.size() * sizeof(float));

    // Host update per utterance.
    std::vector<float> embs_b, pooled(static_cast<size_t>(Tc) * ns);
    for (int b = 0; b < Bn; ++b) {
        Utterance &       u  = *group[static_cast<size_t>(b)];
        const ChunkGeom & g  = geo[static_cast<size_t>(b)];
        const float *     hi = pc->step_hi.data() + hi_per * b;

        // downsample_preds(hi, up): avg_pool1d(kernel=stride=up); T_cat*up
        // is an exact multiple so every window is full.
        for (int t = 0; t < Tc; ++t) {
            for (int s = 0; s < ns; ++s) {
                float acc = 0.0f;
                for (int j = 0; j < up; ++j) {
                    acc += hi[(static_cast<size_t>(t) * up + j) * ns + s];
                }
                pooled[static_cast<size_t>(t) * ns + s] = acc / static_cast<float>(up);
            }
        }
        embs_b.assign(pc->step_embs.begin() + static_cast<size_t>(Td) * d * b,
                      pc->step_embs.begin() + static_cast<size_t>(Td) * d * (b + 1));

        const int S0 = u.st.spkcache_n;
        const int F0 = u.st.fifo_n;
        const int C  = Td - g.lc - g.rc;
        streaming_update(u.st, u.P, ns, d, embs_b, Td, pooled, g.lc, g.rc, pm->sil_emb_host);

        // Chunk output at 10 ms: hi[(S+F+lc)*up : (S+F+lc+C)*up].
        const size_t off = static_cast<size_t>(S0 + F0 + g.lc) * up * ns;
        u.st.total_preds.insert(u.st.total_preds.end(), hi + off, hi + off + static_cast<size_t>(C) * up * ns);
        u.st.total_n += C * up;

        u.stt = g.end;
        ++u.chunk_idx;
        // Drop consumed mel (lc = 0 for every preset; keep lc context anyway).
        const int64_t keep_from =
            std::max<int64_t>(u.mel_base, u.stt - static_cast<int64_t>(u.P.chunk_left_context) * sub);
        if (keep_from > u.mel_base) {
            u.mel_tm.erase(u.mel_tm.begin(), u.mel_tm.begin() + static_cast<size_t>(keep_from - u.mel_base) * nm);
            u.mel_base = keep_from;
        }
    }
    return TRANSCRIBE_OK;
}

// Drive one utterance over the chunks of its mel. final: u.mel_frames is the
// whole recording, run every chunk and apply the end trim. Otherwise more mel
// will follow: run only the chunks whose full window (chunk + right context)
// is already there, so their geometry cannot depend on the final length.
transcribe_status advance_chunks(Nemotron3DiarSession * pc, Nemotron3DiarModel * pm, Utterance & u, bool final) {
    const int     sub  = pm->hparams.enc_subsampling_factor;
    const int64_t full = static_cast<int64_t>(u.P.chunk_len + u.P.chunk_right_context) * sub;
    ChunkGeom     g;
    while ((final || u.stt + full <= u.mel_frames) && next_chunk(u, sub, u.mel_frames, g)) {
        if (pc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const bool dump_enc =
            u.chunk_idx == 0 && transcribe::debug::enabled() && env_set("TRANSCRIBE_NEMOTRON3_DIAR_ENCODER_DUMP");
        std::vector<Utterance *> grp{ &u };
        std::vector<ChunkGeom>   geo{ g };
        if (const transcribe_status st = run_step(pc, pm, grp, geo, dump_enc); st != TRANSCRIBE_OK) {
            return st;
        }
    }
    if (!final) {
        return TRANSCRIBE_OK;
    }
    // NeMo forward_streaming: total_preds[:, :ceil(feat_len / output_subsampling_factor)].
    const int64_t keep = std::min<int64_t>(u.st.total_n, u.mel_frames);
    u.st.total_preds.resize(static_cast<size_t>(keep) * pm->hparams.max_speakers);
    u.st.total_n = static_cast<int>(keep);
    return TRANSCRIBE_OK;
}

double ms_per_output_frame(const Nemotron3DiarHParams & hp) {
    return 1000.0 * static_cast<double>(hp.output_hop) / static_cast<double>(hp.fe_sample_rate);
}

void dump_probs(const Utterance & u, int n_spk) {
    if (!transcribe::debug::enabled()) {
        return;
    }
    const long long shape[2] = { u.st.total_n, n_spk };
    transcribe::debug::dump_host_f32("diar.probs", u.st.total_preds.data(),
                                     static_cast<long long>(u.st.total_n) * n_spk, shape, 2, "diarize");
}

// Append mel frames [0, n_valid) of a row-major [n_mels, stride] buffer as
// time-major rows to u.mel_tm, starting at column `col0`.
void append_mel(Utterance & u, const std::vector<float> & mel, int n_mels, int stride, int col0, int n) {
    const size_t old = u.mel_tm.size();
    u.mel_tm.resize(old + static_cast<size_t>(n) * n_mels);
    float * dst = u.mel_tm.data() + old;
    for (int t = 0; t < n; ++t) {
        for (int m = 0; m < n_mels; ++m) {
            dst[static_cast<size_t>(t) * n_mels + m] = mel[static_cast<size_t>(m) * stride + col0 + t];
        }
    }
    u.mel_frames += n;
}

}  // namespace

// ---- load ----

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = ggml_time_us();

    auto m     = std::make_unique<Nemotron3DiarModel>();
    m->arch    = &arch;
    m->variant = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    if (const transcribe_status st = read_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }
    const Nemotron3DiarHParams & hp = m->hparams;

    gguf_init_params init_params{};
    init_params.no_alloc     = true;
    init_params.ctx          = &m->ctx_meta;
    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    struct GgufGuard {
        gguf_context * g;

        ~GgufGuard() { gguf_free(g); }
    } guard{ gguf_data };

    if (const transcribe_status st = build_weights(m->ctx_meta, hp, m->weights); st != TRANSCRIBE_OK) {
        return st;
    }

    // Mel frontend with the checkpoint's BF16-stored window + filterbank
    // (NeMo's reference mel uses them; an fp32 recompute differs by 1.9e-3).
    {
        transcribe::MelConfig cfg{};
        cfg.sample_rate  = hp.fe_sample_rate;
        cfg.num_mels     = hp.fe_num_mels;
        cfg.n_fft        = hp.fe_n_fft;
        cfg.win_length   = hp.fe_win_length;
        cfg.hop_length   = hp.fe_hop_length;
        cfg.pre_emphasis = hp.fe_pre_emphasis;
        cfg.normalize    = hp.fe_normalize;
        cfg.pad_mode     = "constant";
        cfg.window_type  = "hann_symmetric";
        namespace lc     = transcribe::load_common;
        if (lc::read_f32_tensor_checked(gguf_data, loader.path(), "frontend.window",
                                        static_cast<size_t>(hp.fe_win_length), "nemotron3_diar",
                                        cfg.window) != lc::ReadF32Result::Ok ||
            lc::read_f32_tensor_checked(gguf_data, loader.path(), "frontend.mel_filterbank",
                                        static_cast<size_t>(hp.fe_num_mels) * (hp.fe_n_fft / 2 + 1), "nemotron3_diar",
                                        cfg.filterbank) != lc::ReadF32Result::Ok) {
            return TRANSCRIBE_ERR_GGUF;
        }
        m->mel.emplace(cfg);
    }

    const transcribe_backend_request backend_req = (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, (params != nullptr) ? params->device : nullptr, "nemotron3_diar", m->plan);
        st != TRANSCRIBE_OK) {
        return st;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    if (const transcribe_status st = transcribe::load_common::load_weights(
            loader.path(), gguf_data, m->ctx_meta, m->plan, params, "nemotron3_diar", m->weight_storage);
        st != TRANSCRIBE_OK) {
        return st;
    }

    // Host copy of the learned AOSC silence embedding (compress runs on host).
    m->sil_emb_host.resize(static_cast<size_t>(hp.enc_d_model));
    ggml_backend_tensor_get(m->weights.sil_emb, m->sil_emb_host.data(), 0, m->sil_emb_host.size() * sizeof(float));

    // F32 promotion of the BF16 / F16 matmul weights on CPU (exact upcast).
    // Quantized weights (Q8_0, Q*_K) are left alone: they are lossy by design
    // and run on ggml's native quantized kernels, so an upcast would only
    // discard their memory / speed benefit. ggml-cpu rounds the F32
    // activations to the weight's dtype inside each BF16 / F16 dot product;
    // the reference computes in fp32 over
    // the same BF16-exact weights. That rounding is not benign here: the AOSC
    // cache compression is a top-k selection, so a 1e-3 probability shift
    // flips picks and every later chunk diverges (measured on the 8-speaker
    // oracle: BF16 compute flips 43 picks at compression #1, probs drift
    // 3.2e-2; F32 compute keeps every pick, probs 1.0e-5 over 92 s). ggml's
    // BF16 CPU matmul is also ~7x slower than F32 here. Costs ~400 MB RAM.
    // TRANSCRIBE_NEMOTRON3_DIAR_NATIVE_BF16=1 opts out on CPU;
    // TRANSCRIBE_NEMOTRON3_DIAR_F32_WEIGHTS=1 forces it on other backends.
    {
        std::vector<ggml_tensor **> slots;
        const bool                  on_cpu  = m->plan.primary_kind == transcribe::BackendKind::Cpu;
        const bool                  promote = env_set("TRANSCRIBE_NEMOTRON3_DIAR_F32_WEIGHTS") ||
                                              (on_cpu && !env_set("TRANSCRIBE_NEMOTRON3_DIAR_NATIVE_BF16"));
        if (on_cpu && m->weights.upsample_w->type != GGML_TYPE_F32) {
            slots.push_back(&m->weights.upsample_w);  // F16 conv kernel: always on CPU
        }
        if (promote) {
            auto add = [&](ggml_tensor ** t) {
                const bool half = (*t)->type == GGML_TYPE_BF16 || (*t)->type == GGML_TYPE_F16;
                if (half && std::find(slots.begin(), slots.end(), t) == slots.end()) {
                    slots.push_back(t);
                }
            };
            add(&m->weights.pre_encode_proj_w);
            for (auto & b : m->weights.blocks) {
                add(&b.attn_qkv_w);
                add(&b.attn_out_w);
                add(&b.ff_in_w);
                add(&b.ff_out_w);
            }
            add(&m->weights.enc_proj_w);
            add(&m->weights.upsample_w);
            add(&m->weights.fc1_w);
            add(&m->weights.spk_head_w);
        }
        if (const transcribe_status st = promote_to_f32(slots, m->plan.primary, &m->f32_ctx, &m->f32_buffer);
            st != TRANSCRIBE_OK) {
            return st;
        }
    }

    m->roles = TRANSCRIBE_ROLE_DIARIZE;
    // The abort callback is honored (transcribe_diarize_set_abort_callback).
    transcribe::set_feature(m.get(), TRANSCRIBE_FEATURE_CANCELLATION, true);
    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

// ---- DIARIZE role ----

// The caller's preset, read as a raw int (see transcribe::enum_field_raw):
// an out-of-range value must be rejected before it is loaded as the enum.
static int ext_preset_raw(const transcribe_diarize_params * params) {
    return transcribe::enum_field_raw(
        &reinterpret_cast<const transcribe_nemotron3_diar_diarize_ext *>(params->family)->preset);
}

static int diarize_max_speakers(const transcribe_model * model) {
    return static_cast<const Nemotron3DiarModel *>(model)->hparams.max_speakers;
}

static transcribe_diarize_session * diarize_new_session() {
    return new Nemotron3DiarSession();
}

static transcribe_status diarize_run_validate(const transcribe_diarize_params * params) {
    if (const transcribe_status st = transcribe_ext_check(params->family, TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_DIARIZE,
                                                          sizeof(struct transcribe_nemotron3_diar_diarize_ext));
        st != TRANSCRIBE_OK) {
        return st;
    }
    return preset_is_valid(ext_preset_raw(params)) ? TRANSCRIBE_OK : TRANSCRIBE_ERR_INVALID_ARG;
}

// Whole recording: mel -> streaming chunk loop -> 10 ms frame probabilities.
static transcribe_status diarize_run(transcribe_diarize_session *      session,
                                     const float *                     pcm,
                                     int                               n_samples,
                                     const transcribe_diarize_params * params,
                                     DiarizeProbs &                    out) {
    auto *                       pc = static_cast<Nemotron3DiarSession *>(session);
    auto *                       pm = static_cast<Nemotron3DiarModel *>(session->model);
    const Nemotron3DiarHParams & hp = pm->hparams;

    if (pc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    // Range-checked by diarize_run_validate before the dispatcher got here.
    const transcribe_nemotron3_diar_preset preset =
        params->family != nullptr ? static_cast<transcribe_nemotron3_diar_preset>(ext_preset_raw(params)) :
                                    TRANSCRIBE_NEMOTRON3_DIAR_PRESET_DEFAULT;
    transcribe::debug::init();

    out.n_speakers = hp.max_speakers;
    out.frame_ms   = ms_per_output_frame(hp);

    // Fewer than one hop of audio: NeMo emits floor(n / hop) = 0 frames, an
    // empty (not failed) result.
    if (n_samples < hp.fe_hop_length) {
        return TRANSCRIBE_OK;
    }

    const int64_t t_mel0   = ggml_time_us();
    int           n_mels   = 0;
    int           n_frames = 0;
    if (const transcribe_status mst =
            pm->mel->compute(pcm, static_cast<size_t>(n_samples), pc->mel_buf, n_mels, n_frames, pc->n_threads);
        mst != TRANSCRIBE_OK) {
        return mst;
    }
    // NeMo get_seq_len: floor(n / hop) valid frames (the STFT's trailing
    // center frame is dropped).
    const int valid = std::min(n_frames, n_samples / hp.fe_hop_length);
    pc->t_mel_us    = ggml_time_us() - t_mel0;

    Utterance & u = pc->utt;
    u             = Utterance{};
    u.P           = resolve_stream_params(hp, preset);
    append_mel(u, pc->mel_buf, n_mels, n_frames, 0, valid);

    if (transcribe::debug::enabled()) {
        std::vector<float> mel_valid(static_cast<size_t>(n_mels) * valid);
        for (int mm = 0; mm < n_mels; ++mm) {
            std::copy(pc->mel_buf.begin() + static_cast<size_t>(mm) * n_frames,
                      pc->mel_buf.begin() + static_cast<size_t>(mm) * n_frames + valid,
                      mel_valid.begin() + static_cast<size_t>(mm) * valid);
        }
        const long long shape[2] = { n_mels, valid };
        transcribe::debug::dump_host_f32("enc.mel.in", mel_valid.data(), static_cast<long long>(mel_valid.size()),
                                         shape, 2, "frontend");
    }
    // The chunk loop reads u.mel_tm; drop the row-major copy (it grows with
    // the recording).
    std::vector<float>().swap(pc->mel_buf);

    const int64_t t_enc0 = ggml_time_us();
    if (const transcribe_status st = advance_chunks(pc, pm, u, /*final=*/true); st != TRANSCRIBE_OK) {
        return st;
    }
    pc->t_encode_us = ggml_time_us() - t_enc0;

    dump_probs(u, hp.max_speakers);
    out.n_frames = u.st.total_n;
    out.probs.swap(u.st.total_preds);
    u = Utterance{};
    return TRANSCRIBE_OK;
}

// ---- DIARIZE role, push mode ----
//
// Feeding a recording in pieces gives the frames of diarize_run on the whole
// recording. Two things make that hold:
//   - mel: the frontend has no cross-frame normalization, so frame t depends
//     only on the pre-emphasized samples [t*hop - n_fft/2, t*hop + n_fft/2).
//     A frame is computed once all of them have been fed (at finish the
//     missing ones are the zero padding of the whole-recording STFT), from a
//     PCM slice that starts a whole number of hops earlier so the slice's own
//     left padding and its un-emphasized first sample fall in frames that are
//     dropped;
//   - chunks: advance_chunks(final=false) runs a chunk only when its window
//     no longer depends on the recording length.

// Turn fed PCM into mel frames of `u`. final: every frame of the recording
// (floor(n / hop)); otherwise only the frames whose samples are all there.
// Reads the session's fed PCM without changing it.
static transcribe_status stream_mel(Nemotron3DiarSession * pc, Nemotron3DiarModel * pm, Utterance & u, bool final) {
    const Nemotron3DiarHParams & hp   = pm->hparams;
    const int64_t                hop  = hp.fe_hop_length;
    const int64_t                pad  = hp.fe_n_fft / 2;
    const int64_t                back = (pad + hop) / hop;  // leading slice frames touched by the slice start
    const int64_t                n    = pc->stream_n_fed;

    const int64_t valid = n / hop;
    const int64_t ready = final ? valid : std::min<int64_t>(valid, n >= pad ? (n - pad) / hop + 1 : 0);
    if (ready <= u.mel_frames) {
        return TRANSCRIBE_OK;
    }
    const int64_t first    = std::max<int64_t>(0, u.mel_frames - back);  // global frame of slice frame 0
    const size_t  off      = static_cast<size_t>(first * hop - pc->stream_pcm_base);
    const int64_t t0       = ggml_time_us();
    int           n_mels   = 0;
    int           n_frames = 0;
    if (const transcribe_status st = pm->mel->compute(pc->stream_pcm.data() + off, pc->stream_pcm.size() - off,
                                                      pc->mel_buf, n_mels, n_frames, pc->n_threads);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (ready - first > n_frames) {
        return TRANSCRIBE_ERR_BACKEND;  // unreachable: the slice holds every sample of frames < ready
    }
    append_mel(u, pc->mel_buf, n_mels, n_frames, static_cast<int>(u.mel_frames - first),
               static_cast<int>(ready - u.mel_frames));
    pc->t_mel_us += ggml_time_us() - t0;
    return TRANSCRIBE_OK;
}

// Drop the fed PCM no later mel slice needs.
static void stream_trim_pcm(Nemotron3DiarSession * pc, const Nemotron3DiarHParams & hp) {
    const int64_t hop       = hp.fe_hop_length;
    const int64_t back      = (hp.fe_n_fft / 2 + hop) / hop;
    const int64_t keep_from = std::max<int64_t>(0, pc->utt.mel_frames - back) * hop;
    if (keep_from > pc->stream_pcm_base) {
        pc->stream_pcm.erase(pc->stream_pcm.begin(),
                             pc->stream_pcm.begin() + static_cast<ptrdiff_t>(keep_from - pc->stream_pcm_base));
        pc->stream_pcm_base = keep_from;
    }
}

static transcribe_status diarize_stream_begin(transcribe_diarize_session *      session,
                                              const transcribe_diarize_params * params) {
    auto *                                 pc = static_cast<Nemotron3DiarSession *>(session);
    auto *                                 pm = static_cast<Nemotron3DiarModel *>(session->model);
    // Range-checked by diarize_run_validate before the dispatcher got here.
    const transcribe_nemotron3_diar_preset preset =
        params->family != nullptr ? static_cast<transcribe_nemotron3_diar_preset>(ext_preset_raw(params)) :
                                    TRANSCRIBE_NEMOTRON3_DIAR_PRESET_DEFAULT;
    transcribe::debug::init();
    pc->utt   = Utterance{};
    pc->utt.P = resolve_stream_params(pm->hparams, preset);
    pc->stream_pcm.clear();
    pc->stream_pcm_base = 0;
    pc->stream_n_fed    = 0;
    return TRANSCRIBE_OK;
}

static transcribe_status stream_advance(Nemotron3DiarSession * pc, Nemotron3DiarModel * pm, bool final) {
    if (pc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    if (const transcribe_status st = stream_mel(pc, pm, pc->utt, final); st != TRANSCRIBE_OK) {
        return st;
    }
    stream_trim_pcm(pc, pm->hparams);
    const int64_t t_enc0 = ggml_time_us();
    if (const transcribe_status st = advance_chunks(pc, pm, pc->utt, final); st != TRANSCRIBE_OK) {
        return st;
    }
    pc->t_encode_us += ggml_time_us() - t_enc0;
    if (pc->utt.st.total_n > 0) {
        dump_probs(pc->utt, pm->hparams.max_speakers);  // the frames so far, rewritten per call
    }
    return TRANSCRIBE_OK;
}

static transcribe_status diarize_stream_feed(transcribe_diarize_session * session,
                                             const float *                pcm,
                                             int                          n_samples,
                                             DiarizeProbsView &           decided) {
    auto * pc = static_cast<Nemotron3DiarSession *>(session);
    auto * pm = static_cast<Nemotron3DiarModel *>(session->model);
    pc->stream_pcm.insert(pc->stream_pcm.end(), pcm, pcm + n_samples);
    pc->stream_n_fed += n_samples;
    if (const transcribe_status st = stream_advance(pc, pm, /*final=*/false); st != TRANSCRIBE_OK) {
        return st;
    }
    decided.probs      = pc->utt.st.total_preds.data();
    decided.n_frames   = pc->utt.st.total_n;
    decided.n_speakers = pm->hparams.max_speakers;
    decided.frame_ms   = ms_per_output_frame(pm->hparams);
    return TRANSCRIBE_OK;
}

static transcribe_status diarize_stream_finish(transcribe_diarize_session * session, DiarizeProbs & out) {
    auto * pc      = static_cast<Nemotron3DiarSession *>(session);
    auto * pm      = static_cast<Nemotron3DiarModel *>(session->model);
    out.n_speakers = pm->hparams.max_speakers;
    out.frame_ms   = ms_per_output_frame(pm->hparams);
    if (const transcribe_status st = stream_advance(pc, pm, /*final=*/true); st != TRANSCRIBE_OK) {
        return st;
    }
    out.n_frames = pc->utt.st.total_n;
    out.probs.swap(pc->utt.st.total_preds);
    pc->utt = Utterance{};
    std::vector<float>().swap(pc->stream_pcm);
    std::vector<float>().swap(pc->mel_buf);
    return TRANSCRIBE_OK;
}

// The tail as finish would compute it, on a copy of the utterance (operating
// point, speaker cache, FIFO, pending mel, frames so far). The committed
// utterance and the fed PCM are only read; the session's graph scratch is
// shared, which holds no stream state.
static transcribe_status diarize_stream_peek(transcribe_diarize_session * session, DiarizeProbs & out) {
    auto * pc      = static_cast<Nemotron3DiarSession *>(session);
    auto * pm      = static_cast<Nemotron3DiarModel *>(session->model);
    out.n_speakers = pm->hparams.max_speakers;
    out.frame_ms   = ms_per_output_frame(pm->hparams);
    if (pc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    Utterance u = pc->utt;
    if (const transcribe_status st = stream_mel(pc, pm, u, /*final=*/true); st != TRANSCRIBE_OK) {
        return st;
    }
    const int64_t t_enc0 = ggml_time_us();
    if (const transcribe_status st = advance_chunks(pc, pm, u, /*final=*/true); st != TRANSCRIBE_OK) {
        return st;
    }
    pc->t_encode_us += ggml_time_us() - t_enc0;
    out.n_frames = u.st.total_n;
    out.probs.swap(u.st.total_preds);
    return TRANSCRIBE_OK;
}

// Kind+slot probe: the operating-point preset on the DIARIZE_RUN slot.
static bool accepts_ext_kind(const transcribe_model * model, transcribe_ext_slot slot, uint32_t kind) {
    return model != nullptr && slot == TRANSCRIBE_EXT_SLOT_DIARIZE_RUN &&
           kind == TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_DIARIZE;
}

static const DiarizeOps k_diarize_ops = {
    /* .max_speakers  = */ diarize_max_speakers,
    /* .new_session   = */ diarize_new_session,
    /* .run_validate  = */ diarize_run_validate,
    /* .run           = */ diarize_run,
    /* .stream_begin  = */ diarize_stream_begin,
    /* .stream_feed   = */ diarize_stream_feed,
    /* .stream_finish = */ diarize_stream_finish,
    /* .stream_peek   = */ diarize_stream_peek,
};

extern const Arch arch = {
    /* .name             = */ "nemotron3_diar",
    /* .load             = */ load,
    /* .init_context     = */ nullptr,
    /* .run              = */ nullptr,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ accepts_ext_kind,
    /* .run_validate     = */ nullptr,
    /* .diarize          = */ &k_diarize_ops,
};

}  // namespace transcribe::nemotron3_diar

extern "C" void transcribe_nemotron3_diar_diarize_ext_init(struct transcribe_nemotron3_diar_diarize_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size = sizeof(*p);
    p->ext.kind = TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_DIARIZE;
    p->preset   = TRANSCRIBE_NEMOTRON3_DIAR_PRESET_DEFAULT;
}
