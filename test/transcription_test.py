import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    import os
    import gc
    import sys
    from pathlib import Path

    import torch 
    import numpy as np

    import whisperx
    import soundfile as sf
    from kokoro import KPipeline

    import marimo as mo
    from IPython.display import Audio, display


    device = "cpu"
    batch_size = 12
    compute_type = "int8"
    audio_file = mo.notebook_dir()/"combined.mp3"

    # return Audio, KPipeline, Path, display, os, whisperx


@app.function
def add_silence(audio_segments, pause_duration=2.0, sample_rate=24000):
    """Merge audio segments with silence between them"""
    out_audio_segments = []
    total_pause_duration = 0
    for isegment, segment in enumerate(audio_segments, start=1):
        out_audio_segments.append(segment)
        if isegment < len(audio_segments):
            num_silence_samples = int(pause_duration * sample_rate)
            out_audio_segments.append(np.zeros(num_silence_samples))
            total_pause_duration += pause_duration

    # Concatenate all segments
    final_audio = np.concatenate(out_audio_segments)

    # Save
    #sf.write(output_file, final_audio, sample_rate)
    print(f"Created audio with total {total_pause_duration}s pauses")
    print(f"Total duration: {len(final_audio)/sample_rate:.2f} seconds")

    return final_audio


@app.function
def word_align_seg_kkr_to_wspx(seg_aligned_wspx, seg_kkr_pred, sample_rate=24000, init_offset=0, time_margin = 0.001):
    kkr_audio_tensor = seg_kkr_pred.audio
    #kkr_pred_dur_tensor = seg_kkr_pred.pred_dur

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
        start_idx = round(start_time*sample_rate) 
        end_idx = round(end_time*sample_rate)
        token_audio = kkr_audio_tensor[start_idx:end_idx]
        total_offset += token_audio.shape[0]

        aligned_kkr_word_segs.append(token_audio)

    return aligned_kkr_word_segs, total_offset


@app.function
def TimeSlicer(word_segments):
    # word_segments = self.WordSegments
    StartSeg = -1
    EndSeg = -1
    Step = []
    for i in range(len(word_segments)):
        if i + 2 < len(word_segments):
            if (
                "start" in word_segments[i]["word"].casefold()
                and ("step" in word_segments[i + 1]["word"].casefold())
                # and ("step" in word_segments[i + 1]["word"].casefold() or "step" in word_segments[i + 2]["word"].casefold())
            ):
                StartSeg = i
            if (
                "end" in word_segments[i]["word"].casefold()
                or "and" in word_segments[i]["word"].casefold()
                or "finish" in word_segments[i]["word"].casefold()
                or "stop" in word_segments[i]["word"].casefold()
                # ) and ("step" in word_segments[i + 1]["word"].casefold() or "step" in word_segments[i + 2]["word"].casefold()):
            ) and ("step" in word_segments[i + 1]["word"].casefold()):
                EndSeg = i + 3
        if StartSeg != -1 and EndSeg != -1:
            Step.append(word_segments[StartSeg:EndSeg])
            StartSeg = -1
            EndSeg = -1
    if len(Step) == 0:
        Step.append(word_segments)
        StichedStep = [" ".join([j["word"] for j in i]) for i in Step]
        # print('End Stitching. Attempting any corrections')
        for i in StichedStep:
            if "and step" in i.casefold() or "And step" in i.casefold():
                i.casefold().replace("and step", "end step")
        # print('Corrections successful!')
        CompleteStepTiming = [Step]
    else:
        StichedStep = [" ".join([j["word"] for j in i]) for i in Step]
        # print('End Stitching. Attempting any corrections')
        for i in StichedStep:
            if "and step" in i.casefold() or "And step" in i.casefold():
                i.casefold().replace("and step", "end step")
        # print('Corrections successful!')

        CompleteStepTiming = [
            [Step[i][0]["start"], Step[i][len(Step[i]) - 1]["end"]]
            for i in range(len(Step))
        ]

    return Step, StichedStep, CompleteStepTiming


@app.cell
def _():
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    return (model,)


@app.cell
def _():
    print(audio_file)
    return


@app.cell
def _(model):
    _audio = whisperx.load_audio(audio_file)
    _oresult = model.transcribe(_audio, batch_size=batch_size, language="en")
    _model_a, _metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(
        _oresult["segments"],
        _model_a,
        _metadata,
        _audio,
        device,
        return_char_alignments=False,
    )
    return (result,)


@app.cell
def _(result):
    import json as _json
    #print(_json.dumps(result, indent=4))
    _step , _stiched_step, _complete_step_timing = TimeSlicer(result["word_segments"])
    print(_json.dumps(result["word_segments"], indent=4))
    #print(_json.dumps(_step, indent=4))
    #print(len(_step))

    #for index, (stepText, fullStep) in enumerate(zip(_stiched_step, _step)):
    #    print(index)
    #    print(stepText)
    #    print(_json.dumps(fullStep, indent=4))
    return


@app.cell
def _(result):
    pipeline = KPipeline(lang_code="a")
    _sample_rate = 24000
    _total_offset = 0
    _aligned_words = []
    for _segment in result["segments"]:
        _segment_kkr_pred = next(pipeline(_segment["text"], voice="af_heart", speed=0.9))
        _aligned_segment_words, _total_offset = word_align_seg_kkr_to_wspx(_segment, _segment_kkr_pred, _sample_rate, _total_offset, time_margin=0.)
        _word_aligned_segment = torch.cat(_aligned_segment_words)
        display(_segment["text"], Audio(_word_aligned_segment, rate=_sample_rate, autoplay=False))

        _aligned_words.extend(_aligned_segment_words)

    _word_aligned_audio = torch.cat(_aligned_words)

    #display(Audio(_segment_kkr_pred[0].audio, rate=_sample_rate, autoplay=False))
    display(Audio(_word_aligned_audio, rate=_sample_rate, autoplay=False))

    #sf.write("test.mp3", _word_aligned_audio, _sample_rate)

    #print(int(_token.start_ts*_sample_rate), int(_token.end_ts*_sample_rate))


    #print(_audio[0].text_index)


    #_audio_segments = []
    #for _segment in result["segments"]:
        # for text in result["segments"]:
    #    _audio = pipeline(_segment["text"], voice="af_heart")
    #    print(next(_generator))
        #for _iword, _word in enumerate(_segment["words"]):
        #    _word_start_rel = _word["start"] - _segment["start"]


        #for _i, (_gs, _ps, _audio) in enumerate(_generator):
        #    #print(_i, _gs, _ps)
        #    _audio_segments.append(_audio)
        #    #display(Audio(_audio, rate=24000, autoplay=False))

    #_full_audio_no_pauses = np.concatenate(_audio_segments)
    #display(Audio(_full_audio_no_pauses, rate=24000, autoplay=False))

    #_full_audio_with_pauses = add_silence(_audio_segments, pause_duration=3.0, sample_rate=24000)
    #display(Audio(_full_audio_with_pauses, rate=24000, autoplay=False))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
