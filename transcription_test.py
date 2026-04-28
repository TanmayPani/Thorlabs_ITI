import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    import json
    import torch
    from copy import deepcopy
    from collections import defaultdict
    #import whisperx
    #from kokoro import KPipeline
    import marimo as mo
    from IPython.display import Audio, display
    from pathlib import Path
    import difflib
    import re

    from snippets import (
        process_bom_data,
        read_and_combine_videos,
        speech_to_text,
        step_slicer,
        step_text_to_speech,
        video_step_slicer, map_step_with_word_segments,
        normalize_segment_text
    )



    device = "cpu"

    audio_file = mo.notebook_dir() / "test" / "combined.mp3"
    out_dir = mo.notebook_dir() / "test"


@app.cell
def _():
    return


@app.cell
def _():
    step_files = step_slicer(out_dir / "combined.json", out_dir)
    len(step_files)
    return (step_files,)


@app.cell
def _(step_files):

    _new_text = normalize_segment_text("Stout step twain. We're going to clean the surface with alcohol, [70%]. Clean the entire surface and there are some dark marks that are difficult to remove that you will need nit to use acetone. Do not scuba too hard or you will take off the powder coating. Ending step.")

    with step_files[0].open("r") as fin:
        _step = json.load(fin)


    diffs =  difflib.ndiff(_step["text"].split(), _new_text.split())

    new_word_segments = [[] for _ in _step["segments"]]
    #print(json.dumps(_step["word_segments"], indent=4))
    #orig_word_segments = []

    iword = 0

    for d in diffs:
        #print(d[:2], d[2:])
        match d[:2]:
            case "? ":
                continue
            case "+ ":
                seg_idx = _step["word_segments"][iword - 1]["seg_idx"]
                new_word_segments[seg_idx].append({"word" : d[2:], "iword_orig":iword-1})
                print(new_word_segments[seg_idx][-1]["word"], _step["word_segments"][iword - 1]["word"], iword-1)
            case "- ":
                #orig_word_segments.append(d[2:])
                iword += 1
            case "  ":
                #orig_word_segments.append(d[2:])
                seg_idx = _step["word_segments"][iword]["seg_idx"]
                new_word_segments[seg_idx].append({"word" : d[2:], "iword_orig":iword})
                iword += 1

                print(new_word_segments[seg_idx][-1]["word"],  _step["word_segments"][iword - 1]["word"], iword-1)
            case _:
                raise ValueError("diff prefix can only be one of \"? \", \"+ \", \"- \", \"  \"!!")
    #print(json.dumps(new_word_segments, indent=4))

    repeats = defaultdict(list)
    for iseg, seg in enumerate(new_word_segments):
        #print(iseg)
        for iwrd, wrd in enumerate(seg):
            #print(iseg, iwrd, wrd)
            repeats[wrd["iword_orig"]].append((iseg, iwrd))

    for iwrd_orig, new_word_idx in repeats.items():
        if len(new_word_idx) == 1:
            iseg, iwrd = new_word_idx[0]
            new_word_segments[iseg][iwrd]["start"] = _step["word_segments"][iwrd_orig]["start"]
            new_word_segments[iseg][iwrd]["end"] = _step["word_segments"][iwrd_orig]["end"]

        if len(new_word_idx) > 1:
            num_wrds = len(new_word_idx)
            tot_dur = _step["word_segments"][iwrd_orig]["end"] - _step["word_segments"][iwrd_orig]["start"]
            dur_per_wrd = tot_dur / float(num_wrds + 1)

            for i, (iseg, iwrd) in enumerate(new_word_idx):
                new_word_segments[iseg][iwrd]["start"] = _step["word_segments"][iwrd_orig]["start"] + i*dur_per_wrd
                new_word_segments[iseg][iwrd]["end"] = _step["word_segments"][iwrd_orig]["start"] + (i+1)*dur_per_wrd

    print(json.dumps(new_word_segments, indent=4))       

    #for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    #    print(tag, i1, i2, j1, j2)



    #_rerendered_step = map_step_with_word_segments(_step, _new_text)

    #print(json.dumps(_rerendered_step, indent=4))
    return


@app.cell
def _(KPipeline, crossfade_combine, result, word_align_seg_kkr_to_wspx):
    _pipeline = KPipeline(lang_code="a")
    _sample_rate = 24000
    _total_offset = 0
    _aligned_words = []
    for _segment in result["segments"]:
        _segment_kkr_pred = next(
            _pipeline(_segment["text"], voice="af_heart", speed=1.2)
        )
        print(len(_segment_kkr_pred.tokens), len(_segment["words"]))
        #continue 
        _aligned_segment_words, _total_offset = word_align_seg_kkr_to_wspx(
            _segment, _segment_kkr_pred, _sample_rate, _total_offset
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
