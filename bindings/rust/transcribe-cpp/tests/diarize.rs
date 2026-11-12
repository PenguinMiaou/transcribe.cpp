//! DIARIZE role on Sortformer (TRANSCRIBE_SMOKE_SORTFORMER_MODEL) and whisper
//! (TRANSCRIBE_SMOKE_MODEL); the Busy path is covered in `src/diarize.rs`.

mod common;

use transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_SORTFORMER_DIARIZE as SFDR;
use transcribe_cpp::{
    Backend, CancelToken, DiarizeExtension, DiarizeOptions, Error, ExtSlot, Model, ModelOptions,
    Role, SortformerDiarizeOptions, SortformerPreset,
};

#[test]
fn sortformer_is_diarize_only() {
    let Some((model_path, _)) = common::smoke_sortformer_fixtures("sortformer_is_diarize_only")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let roles = model.roles();
    assert!(!roles.contains(Role::Asr) && roles.contains(Role::Diarize));
    let info = model.diarize_info().unwrap();
    assert_eq!((info.sample_rate, info.max_speakers), (16000, 4));
    assert!(model.accepts_ext(ExtSlot::DiarizeRun, SFDR));
    assert!(!model.accepts_ext(ExtSlot::Run, SFDR));
    assert!(matches!(
        model.capabilities(),
        Err(Error::UnsupportedRole(_))
    ));
    assert!(matches!(model.session(), Err(Error::UnsupportedRole(_))));
}

#[test]
fn diarize_run_returns_turns_and_applies_preset() {
    let Some((model_path, pcm)) =
        common::smoke_sortformer_fixtures("diarize_run_returns_turns_and_applies_preset")
    else {
        return;
    };
    // Exact segments are pinned in C (tests/sortformer_diarize_unit.cpp); here
    // the binding only has to return well-formed turns and pass the preset on.
    let cpu = ModelOptions {
        backend: Backend::Cpu,
        ..Default::default()
    };
    let model = Model::load_with(&model_path, &cpu).unwrap();
    let max_speakers = model.diarize_info().unwrap().max_speakers;
    let mut diarize = model.diarize_session().unwrap();
    let rows = |turns: Vec<transcribe_cpp::SpeakerSegment>| -> Vec<_> {
        turns
            .iter()
            .map(|s| (s.t0_ms, s.t1_ms, s.speaker_id))
            .collect()
    };

    let default = rows(diarize.run(&pcm, &DiarizeOptions::default()).unwrap());
    assert!(!default.is_empty());
    for &(t0, t1, speaker) in &default {
        assert!(t0 < t1, "{t0} >= {t1}");
        assert!((1..=max_speakers).contains(&speaker), "speaker {speaker}");
    }
    assert!(diarize.timings().encode_ms > 0.0);

    let low_latency = DiarizeOptions {
        family: Some(DiarizeExtension::Sortformer(SortformerDiarizeOptions {
            preset: Some(SortformerPreset::LowLatency),
        })),
    };
    assert_ne!(rows(diarize.run(&pcm, &low_latency).unwrap()), default);
}

#[test]
fn nemotron3_diar_diarize_run_and_preset() {
    use transcribe_cpp::{Nemotron3DiarDiarizeOptions, Nemotron3DiarPreset};
    let Some((model_path, pcm)) =
        common::smoke_nemotron3_diar_fixtures("nemotron3_diar_diarize_run_and_preset")
    else {
        return;
    };
    // Exact segments are pinned in C (tests/nemotron3_diar_diarize_unit.cpp).
    let model = Model::load(&model_path).unwrap();
    assert_eq!(model.diarize_info().unwrap().max_speakers, 8);
    let mut diarize = model.diarize_session().unwrap();
    let speakers = |turns: Vec<transcribe_cpp::SpeakerSegment>| -> Vec<i32> {
        let mut ids: Vec<i32> = turns
            .iter()
            .inspect(|s| assert!(s.t0_ms < s.t1_ms))
            .map(|s| s.speaker_id)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    assert_eq!(
        speakers(diarize.run(&pcm, &DiarizeOptions::default()).unwrap()),
        [1, 2]
    );
    let low_latency = DiarizeOptions {
        family: Some(DiarizeExtension::Nemotron3Diar(
            Nemotron3DiarDiarizeOptions {
                preset: Some(Nemotron3DiarPreset::LowLatency),
            },
        )),
    };
    assert_eq!(speakers(diarize.run(&pcm, &low_latency).unwrap()), [1, 2]);
    // A foreign family's extension is rejected for this model.
    let foreign = DiarizeOptions {
        family: Some(DiarizeExtension::Sortformer(
            SortformerDiarizeOptions::default(),
        )),
    };
    assert!(diarize.run(&pcm, &foreign).is_err());
}

#[test]
fn nemotron3_diar_stream_feed_in_pieces_equals_run() {
    use transcribe_cpp::{Nemotron3DiarDiarizeOptions, Nemotron3DiarPreset};
    let Some((model_path, pcm)) =
        common::smoke_nemotron3_diar_fixtures("nemotron3_diar_stream_feed_in_pieces_equals_run")
    else {
        return;
    };
    // Frame-level equality is pinned in C (tests/nemotron3_diar_diarize_stream_unit.cpp).
    let model = Model::load(&model_path).unwrap();
    let mut diarize = model.diarize_session().unwrap();
    let rows = |turns: &[transcribe_cpp::SpeakerSegment]| -> Vec<(i64, i64, i32)> {
        turns
            .iter()
            .map(|s| (s.t0_ms, s.t1_ms, s.speaker_id))
            .collect()
    };
    let audio_ms = (pcm.len() / 16) as i64;
    let low_latency = DiarizeOptions {
        family: Some(DiarizeExtension::Nemotron3Diar(
            Nemotron3DiarDiarizeOptions {
                preset: Some(Nemotron3DiarPreset::LowLatency),
            },
        )),
    };

    // No stream yet.
    assert!(matches!(
        diarize.stream_feed(&pcm[..160]),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        diarize.stream_finish(),
        Err(Error::InvalidArgument(_))
    ));

    for options in [DiarizeOptions::default(), low_latency] {
        let whole = rows(&diarize.run(&pcm, &options).unwrap());
        assert!(!whole.is_empty());
        for pieces in [&[16_000usize][..], &[4_801, 1, 160, 31_999, 257][..]] {
            diarize.stream_begin(&options).unwrap();
            assert_eq!(diarize.stream_decided_ms(), 0);
            assert!(diarize.stream_feed(&[]).unwrap().is_empty());
            let (mut pos, mut k, mut decided) = (0usize, 0usize, 0i64);
            while pos < pcm.len() {
                let n = pieces[k % pieces.len()].min(pcm.len() - pos);
                let so_far = rows(&diarize.stream_feed(&pcm[pos..pos + n]).unwrap());
                pos += n;
                k += 1;
                let now = diarize.stream_decided_ms();
                assert!(now >= decided && now <= (pos / 16) as i64);
                decided = now;
                // The decided part is the whole-run result cut at the decided end.
                let cut: Vec<_> = whole
                    .iter()
                    .filter(|r| r.0 < decided)
                    .map(|r| (r.0, r.1.min(decided), r.2))
                    .collect();
                assert_eq!(so_far, cut, "decided {decided} ms after {pos} samples");
            }
            assert_eq!(rows(&diarize.stream_finish().unwrap()), whole);
            assert_eq!(diarize.stream_decided_ms(), audio_ms / 10 * 10);
        }
    }

    // Malformed input leaves the stream usable; a cancelled feed ends it.
    diarize.stream_begin(&DiarizeOptions::default()).unwrap();
    assert!(matches!(
        diarize.stream_feed(&[f32::NAN; 16]),
        Err(Error::InvalidArgument(_))
    ));
    diarize.stream_feed(&pcm[..16_000]).unwrap();
    let token = CancelToken::new();
    diarize.set_cancel_token(&token);
    token.cancel();
    let err = diarize.stream_feed(&pcm[16_000..32_000]).unwrap_err();
    assert!(matches!(err, Error::Aborted { .. }), "{err:?}");
    diarize.clear_cancel_token();
    assert!(matches!(
        diarize.stream_feed(&pcm[..160]),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn diarize_session_rejects_bad_input_and_cancels() {
    let Some((model_path, pcm)) =
        common::smoke_sortformer_fixtures("diarize_session_rejects_bad_input_and_cancels")
    else {
        return;
    };
    // The session keeps its model alive after the last Model handle drops.
    let mut diarize = Model::load(&model_path).unwrap().diarize_session().unwrap();
    let opts = DiarizeOptions::default();
    for bad in [&[][..], &[f32::NAN; 16][..]] {
        let err = diarize.run(bad, &opts).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }
    let token = CancelToken::new();
    diarize.set_cancel_token(&token);
    token.cancel();
    let err = diarize.run(&pcm, &opts).unwrap_err();
    assert!(matches!(err, Error::Aborted { .. }), "{err:?}");
    diarize.clear_cancel_token();
    assert!(!diarize.run(&pcm, &opts).unwrap().is_empty());
}

#[test]
fn asr_only_model_refuses_diarize() {
    let Some(model_path) = common::smoke_model() else {
        eprintln!("skip asr_only_model_refuses_diarize: smoke model absent");
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let roles = model.roles();
    assert!(roles.contains(Role::Asr) && !roles.contains(Role::Diarize));
    assert!(matches!(
        model.diarize_info(),
        Err(Error::UnsupportedRole(_))
    ));
    assert!(matches!(
        model.diarize_session(),
        Err(Error::UnsupportedRole(_))
    ));
    assert!(!model.accepts_ext(ExtSlot::DiarizeRun, SFDR));
}
