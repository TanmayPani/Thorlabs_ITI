import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    import torch

    import whisperx
    from kokoro import KPipeline
    from pydub import AudioSegment

    import marimo as mo
    from IPython.display import Audio, display


    device = "cpu"

    audio_file = mo.notebook_dir() / "test" / "combined.mp3"


@app.function
def crossfade_combine(a1, a2, cross_ms, sample_rate=24000):
    crossfade_samples = min(int(cross_ms/1000.0)*sample_rate, len(a1), len(a2))
    if crossfade_samples <= 0:
        return torch.cat([a1, a2])

    fade_out = torch.linspace(1.0, 0.0, crossfade_samples)
    fade_in = torch.linspace(0.0, 1.0, crossfade_samples)

    pre_fade = a1[:-crossfade_samples]
    fade1 = a1[-crossfade_samples:]
    fade2 = a2[:crossfade_samples]
    post_fade = a2[crossfade_samples:]

    fade = fade1*fade_out + fade2*fade_in

    return torch.cat([pre_fade, fade, post_fade])


@app.cell
def _(time_margin):
    def word_align_seg_kkr_to_wspx(
        seg_aligned_wspx, seg_kkr_pred, sample_rate=24000, init_offset=0
    ):
        kkr_audio_tensor = seg_kkr_pred.audio
        # kkr_pred_dur_tensor = seg_kkr_pred.pred_dur

        wspx_word_start_times = tuple(word["start"] for word in seg_aligned_wspx["words"])

        aligned_kkr_word_segs = []
        total_offset = int(init_offset)

        for itoken, token in enumerate(seg_kkr_pred.tokens):
            if len(wspx_word_start_times) > itoken:
                offset = round(wspx_word_start_times[itoken] * sample_rate)
                if offset > total_offset:
                    aligned_kkr_word_segs.append(torch.zeros(offset - total_offset))
                    total_offset = offset - int(time_margin * sample_rate)
            start_time = token.start_ts - time_margin
            end_time = token.end_ts + time_margin
            start_idx = round(start_time * sample_rate)
            end_idx = round(end_time * sample_rate)
            token_audio = kkr_audio_tensor[start_idx:end_idx]
            total_offset += token_audio.shape[0]

            aligned_kkr_word_segs.append(token_audio)

        return aligned_kkr_word_segs, total_offset

    return (word_align_seg_kkr_to_wspx,)


@app.function
def step_slicer(whisper_result):
    start_triggers = ("start",)
    end_triggers = ("end", "and", "finish", "stop")

    segments = whisper_result["segments"]
    word_segments = whisper_result["word_segments"]
    total_words = len(word_segments)

    steps = []
    iword = 0
    start_triggered = False
    last_filled = {"segment" : -1, "word" : -1}
    for iseg, seg in enumerate(segments):            
        for iw, w in enumerate(seg["words"]):
            if iword < total_words - 2:
                maybe_step = "step" in word_segments[iword + 1]["word"].casefold()
                if (
                    maybe_step
                    and any(trg in w["word"].casefold() for trg in start_triggers)
                    and not start_triggered
                ):
                    start_triggered = True

                    steps.append({"segments": [], "word_segments": []})

                if (
                    maybe_step
                    and any(trg in w["word"].casefold() for trg in end_triggers)
                    and start_triggered
                ):
                    if last_filled["segment"] < iseg:
                        steps[-1]["segments"].append({"words": []})

                    steps[-1]["segments"][-1]["words"].extend(
                        word_segments[iword : iword + 3]
                    )
                    steps[-1]["word_segments"].extend(word_segments[iword : iword + 3])

                    steps[-1]["start"] = steps[-1]["word_segments"][0]["start"]
                    steps[-1]["end"] = steps[-1]["word_segments"][-1]["end"]
                    #steps[-1]["text"] = " ".join(s["text"] for s in steps[-1]["segments"])

                    start_triggered = False

            if start_triggered:
                steps[-1]["word_segments"].append(w)
                if last_filled["segment"] < iseg:
                    steps[-1]["segments"].append({"words": []})
                steps[-1]["segments"][-1]["words"].append(w)
                last_filled["segment"] = iseg 
                last_filled["word"] = iw 

            iword += 1

        if len(steps[-1]["segments"][-1]["words"]) > 0:
            steps[-1]["segments"][-1]["start"] = steps[-1]["segments"][-1]["words"][0][
                "start"
            ]

            steps[-1]["segments"][-1]["end"] = steps[-1]["segments"][-1]["words"][-1]["end"]
            steps[-1]["segments"][-1]["text"] = " ".join(
                w["word"] for w in steps[-1]["segments"][-1]["words"]
            )

    for istep in range(len(steps)):
        steps[istep]["text"] = " ".join(s["text"] for s in steps[istep]["segments"])

    return steps


@app.cell
def _():
    _model = whisperx.load_model(
        "distil-large-v3", device, compute_type="int8", language="en", vad_method="silero"
    )
    _audio = whisperx.load_audio(str(audio_file))
    _oresult = _model.transcribe(_audio, batch_size=12, chunk_size=30)
    _model_a, _metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(
        _oresult["segments"],
        _model_a,
        _metadata,
        _audio,
        device,
        return_char_alignments=False,
    )

    # display(Audio(_audio, rate=24000, autoplay=False))
    return (result,)


@app.cell
def _(result):
    import json
    _res = step_slicer(result)
    print(type(_res))
    print(json.dumps(_res, indent=4))
    return (json,)


@app.cell
def _():
    import re
    def normalize(word):
        # Lowercase and strip punctuation for the alignment math
        return re.sub(r'[^\w\s]', '', word).lower()

    return normalize, re


@app.cell
def _(normalize, re):
    import difflib 
    def map_step_with_word_segments(original_step, edited_step_text):
        # 1. Unroll using WhisperX's pre-defined word segments
        orig_words_tracked = []  
        for seg_idx, seg in enumerate(original_step["segments"]):
            for word_dict in seg.get('words', []):
                orig_words_tracked.append({
                    "word": word_dict["word"],
                    "seg_idx": seg_idx
                })

        orig_norm = [normalize(item["word"]) for item in orig_words_tracked]

        edit_words = edited_step_text.split()
        edit_norm = [normalize(w) for w in edit_words]

        # 2. Sequence Alignment
        matcher = difflib.SequenceMatcher(None, orig_norm, edit_norm)
        opcodes = matcher.get_opcodes()

        new_segment_words = [[] for _ in range(len(original_step["segments"]))]

        # 3. Distribute edits into segment buckets
        for tag, i1, i2, j1, j2 in opcodes:
            if tag in ('equal', 'replace'):
                for orig_idx, edit_idx in zip(range(i1, i2), range(j1, j2)):
                    seg_id = orig_words_tracked[orig_idx]["seg_idx"]
                    new_segment_words[seg_id].append(edit_words[edit_idx])

            elif tag == 'insert':
                seg_id = orig_words_tracked[i1 - 1]["seg_idx"] if i1 > 0 else 0
                for edit_idx in range(j1, j2):
                    new_segment_words[seg_id].append(edit_words[edit_idx])

        # 4. Re-roll the segments and apply Kokoro Regex
        edit_mapped_step = []
        for i, seg in enumerate(original_step["segments"]):
            if new_segment_words[i]:
                # The raw text as edited by the user
                raw_segment_text = " ".join(new_segment_words[i])

                # --- APPLY KOKORO REGEX ---
                # 1. Remove bracketed text and replace hyphens
                kokoro_text = re.sub(r" ?\[.*?\]", "", raw_segment_text).replace("-", " dash ")
                # 2. Add spaces between digits for serial number reading
                kokoro_text = re.sub(r"(\d)", r"\1 ", kokoro_text)
                # Clean up any accidental double spaces created by the regex
                kokoro_text = re.sub(r"\s+", " ", kokoro_text).strip()

                edit_mapped_step.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "orig_text": seg["text"],
                    "text": raw_segment_text,      # Use this for UI / Subtitles
                    "kokoro_text": kokoro_text     # Feed this directly into Kokoro
                })

        return edit_mapped_step

    return (map_step_with_word_segments,)


@app.cell
def _(json, map_step_with_word_segments, result):
    _res = step_slicer(result)
    _new_text = "Stout step twain. We're going to clean the surface with alcohol[, 70%]. Clean the entire surface and there are some dark marks that are difficult to remove that you will need nit to use acetone. Do not scuba too hard or you will take off the powder coating. Ending step."


    _rec_segs = map_step_with_word_segments(_res[0], _new_text)

    print(json.dumps(_rec_segs, indent=4))
    return


app._unparsable_cell(
    r"""
    def match_target_amplitude(sound, target_dBFS_level):
        change_in_dBFS = target_dBFS_level - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    def normalize_audio(sudio target_dBFS_level=14):
        try:
            sound_file = AudioSegment.from_file(file_address)
            normalized_sound = match_target_amplitude(sound_file, target_dBFS_level)
    """,
    name="_"
)


@app.cell
def _(result, word_align_seg_kkr_to_wspx):
    _pipeline = KPipeline(lang_code="a")
    _sample_rate = 24000
    _total_offset = 0
    _aligned_words = []
    for _segment in result["segments"]:
        _segment_kkr_pred = next(
            _pipeline(_segment["text"], voice="af_heart", speed=1.2)
        )
        _aligned_segment_words, _total_offset = word_align_seg_kkr_to_wspx(
            _segment, _segment_kkr_pred, _sample_rate, _total_offset, time_margin=0.0
        )
        _word_aligned_segment = torch.cat(_aligned_segment_words)
        display(
            _segment["text"],
            Audio(_word_aligned_segment, rate=_sample_rate, autoplay=False),
        )

        _aligned_words.extend(_aligned_segment_words)



    _word_aligned_audio = torch.cat(_aligned_words)

    _crossfaded = _aligned_words[0]
    for iw in range(1, len(_aligned_words)):
        _crossfaded = crossfade_combine(_crossfaded, _aligned_words[iw], 50)

    _normalizedsound = _crossfaded/(_crossfaded.abs().max())
    print(_crossfaded.dtype)
    display(Audio(_normalizedsound , rate=_sample_rate, autoplay=False))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
