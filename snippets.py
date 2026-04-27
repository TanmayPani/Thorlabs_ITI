import gc
import sys
import pickle
import re
import difflib
import json
from functools import lru_cache, reduce
from pathlib import Path

from pandas import read_excel
from natsort import natsorted

import torch

import soundfile as sf
import whisperx
from moviepy import concatenate_videoclips, VideoFileClip, AudioFileClip
from kokoro.pipeline import KPipeline

torch.serialization.add_safe_globals(
    ["omegaconf.listconfig.ListConfig", "omegaconf.dictconfig.DictConfig"]
)


def process_bom_data(fpath, out_dir):
    bom_file_path = list(fpath.glob("*.xls[mx]"))[0]
    df_dict = read_excel(bom_file_path, sheet_name=None)
    # keys = list(df_dict.keys())
    matches = []
    steps = [[], [], []]
    # count = 0
    for idf, (df_key, df) in enumerate(df_dict.items()):
        if df_key == "Master_BOM":
            steps.append(df_key, df[1:], [])
        elif df_key != "BOM_Export":
            if len(df["Item number"]) > 0:
                mask = df["Item number"] == "Tool"
                df["grouper"] = mask.cumsum()
                group_df_dict = {
                    group_key: group_df for group_key, group_df in df.groupby("grouper")
                }

                group_df_dict[1].columns = group_df_dict[1].iloc[0]
                group_df_dict[1] = group_df_dict[1].loc[:, :"Quantity"]
                tmpKeys = group_df_dict[0].keys()
                matches = [j for j in tmpKeys if "unnamed" in j.casefold()]
                matches.append("grouper")
                steps[0].append(df_key)
                steps[1].append(
                    group_df_dict[0].drop(matches, axis=1).reset_index(drop=True)
                )
                steps[2].append(
                    group_df_dict[1]["Tool Description"]
                    .dropna()
                    .reset_index(drop=True)[1:],
                )

    with (out_dir / "AFHeartTxt.pkl").open(mode="wb") as fout:
        pickle.dump(steps[1:], fout)


def read_and_combine_videos(video_dir, audio_dir, thumbnail_path=None):
    if not (video_dir / "combined.mp4").exists():
        print("Creating combined.mp4...")
        video_files = list(video_dir.glob("*.mp4"))

        if len(video_files) > 1:
            sorted_video_files = [fvid for fvid in natsorted(video_files)]
            combined_video = concatenate_videoclips(sorted_video_files)

            combined_video.write_videofile(
                str(video_dir / "combined.mp4"),
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                audio_codec="aac",
                # codec=self.get_best_codec(),
                codec="libx264",
                threads=8,
                logger=None,
                preset="veryfast",
            )
        elif len(video_files) == 1:
            print(f"Creating combined.mp4 from {video_files[0]}...")
            video_files[0].rename(video_dir / "combined.mp4")
            combined_video = VideoFileClip(str(video_dir / "combined.mp4"))
        else:
            raise FileNotFoundError(f"No video files found in {str(video_dir)}!")

    else:
        combined_video = VideoFileClip(video_dir / "combined.mp4")

    if thumbnail_path is not None:
        combined_video.save_frame(thumbnail_path, t=0)
    combined_video.audio.write_audiofile(audio_dir / "combined.mp3")
    combined_video.audio.close()
    combined_video.close()


def video_step_slicer(steps, video_path, step_audio_files, out_dir):
    assert len(steps) != len(step_audio_files), (
        f"Expected one audio file per step, but got {len(step_audio_files)} audio files for {len(steps)} steps!"
    )

    full_video = VideoFileClip(video_path).resized(0.5)
    full_video_no_audio = VideoFileClip(video_path).resized(0.5).without_audio()

    step_video_clips = []
    for istep, step_file in enumerate(steps):
        with step_file.open("r") as fin:
            step = json.load(fin)

        step_audio = AudioFileClip(step_audio_files[istep])
        step_video = full_video[step["start"] : step["end"]]
        step_video_new_audio = full_video_no_audio[
            step["start"] : step["end"]
        ].with_audio(step_audio)

        step_video_clips.append(str(out_dir / f"AFHeart{istep}.mp4"))
        step_video_new_audio.write_videofile(
            step_video_clips[-1],
            codec="libx264",
            audio_codec="aac",
            preset="veryfast",
            logger=None,
            threads=8,
        )

        step_video.write_videofile(
            str(out_dir / f"TmpOGAud{istep}.mp4"),
            codec="libx264",
            audio_codec="aac",
            preset="veryfast",
            logger=None,
            threads=8,
        )

        step_audio.close()
    full_video.close()
    full_video_no_audio.close()
    return step_video_clips


def speech_to_text(
    audio_path,
    text_output_path,
    language="en",
    model_name="large-v3",
    device="cpu",
    compute_type="int8",
    vad_method="silero",
    batch_size=12,
    chunk_size=30,
):
    transcription_model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
        vad_method=vad_method,
    )

    audio = whisperx.load_audio(str(audio_path))

    transcription = transcription_model.transcribe(
        audio, batch_size=batch_size, chunk_size=chunk_size
    )

    del transcription_model
    gc.collect()
    # torch.cuda().clea

    align_model, align_model_metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    aligned_segments = whisperx.align(
        transcription["segments"],
        align_model,
        align_model_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    transcription.clear()
    align_model_metadata.clear()
    del audio, transcription, align_model, align_model_metadata
    gc.collect()

    with text_output_path.open("w") as fout:
        json.dump(aligned_segments, fout, indent=4)

    aligned_segments.clear()
    del aligned_segments
    gc.collect()


def step_slicer(combined_path, out_dir):
    with combined_path.open("r") as fin:
        whisper_result = json.load(fin)

    print(f"Loaded combined transcription from {combined_path}")

    start_triggers = ("start",)
    end_triggers = ("end", "and", "finish", "stop")

    total_words = len(whisper_result["word_segments"])

    steps = []
    iword = 0
    # step_seg_idx = 0
    start_triggered = False
    # last_filled = {"segment": -1, "word": -1}
    current_seg = -1
    print("Beigning step splitting")
    for iseg, seg in enumerate(whisper_result["segments"]):
        for iword_seg, word_seg in enumerate(seg["words"]):
            word = word_seg["word"]

            if iword < total_words - 2 and (
                "step" in whisper_result["word_segments"][iword + 1]["word"].casefold()
            ):
                if not start_triggered:
                    if any(trg in word.casefold() for trg in start_triggers):
                        start_triggered = True
                        # step_seg_idx = 0
                        steps.append({"segments": [], "word_segments": []})

                else:
                    if any(trg in word.casefold() for trg in end_triggers):
                        if current_seg < iseg:
                            steps[-1]["segments"].append({"words": []})
                            current_seg = iseg

                        steps[-1]["segments"][-1]["words"].extend(
                            whisper_result["word_segments"][iword : iword + 3]
                        )
                        steps[-1]["word_segments"].extend(
                            whisper_result["word_segments"][iword : iword + 3]
                        )

                        # steps[-1]["start"] = steps[-1]["word_segments"][0]["start"]
                        # steps[-1]["end"] = steps[-1]["word_segments"][-1]["end"]
                        # steps[-1]["text"] = " ".join(s["text"] for s in steps[-1]["segments"])
                        start_triggered = False

            word_seg["word"] = word
            if start_triggered:
                steps[-1]["word_segments"].append(word_seg)
                # steps[-1]["word_segments"][-1]["seg_idx"] = step_seg_idx
                if current_seg < iseg:
                    steps[-1]["segments"].append({"words": []})
                    current_seg = iseg
                steps[-1]["segments"][-1]["words"].append(word_seg)

            iword += 1
        # step_seg_idx += 1

        # if len(steps[-1]["segments"][-1]["words"]) > 0:
        #    steps[-1]["segments"][-1]["start"] = steps[-1]["segments"][-1]["words"][0][
        #        "start"
        #    ]

        #    steps[-1]["segments"][-1]["end"] = steps[-1]["segments"][-1]["words"][-1][
        #        "end"
        #    ]
        #    steps[-1]["segments"][-1]["text"] = " ".join(
        #        w["word"] for w in steps[-1]["segments"][-1]["words"]
        #    )

    # del whisper_result
    # gc.collect()

    print(f"Step splitting done! Processing {len(steps)} steps")
    step_files = []
    for istep, step in enumerate(steps):
        print(f"Processing step {istep}")
        step["start"] = step["segments"][0]["start"]
        step["end"] = step["segments"][-1]["end"]
        print(step["start"], step["end"])
        iword = 0
        for iseg, seg in enumerate(step["segments"]):
            step["segments"][iseg]["start"] = seg["words"][0]["start"]
            step["segments"][iseg]["end"] = seg["words"][-1]["end"]
            step["segments"][iseg]["text"] = " ".join(w["word"] for w in seg["words"])
            step["word_segments"][iword]["seg_idx"] = iseg
            iword += 1

        step["text"] = " ".join(s["text"] for s in step["segments"])
        print(step["text"])
        step_out_path = out_dir / f"Step{istep}.json"
        print(f"Writing step {istep} to {step_out_path}")
        with step_out_path.open("w") as fout:
            json.dump(step, fout, indent=4)

        step_files.append(step_out_path)

    print("Step splitting done!")
    return step_out_path


def normalize_word(word):
    # Lowercase and strip punctuation for consistent alignment
    word = re.sub(r"[^\w\s]", "", word).lower()


def normalize_segment_text(seg_text):
    # 1. Remove bracketed text and replace hyphens
    seg_text = re.sub(r" ?\[.*?\]", "", seg_text).replace("-", " dash ")
    # 2. Add spaces between digits for serial number reading
    seg_text = re.sub(r"(\d)", r"\1 ", seg_text)
    # Clean up any accidental double spaces created by the regex
    seg_text = re.sub(r"\s+", " ", seg_text).strip()

    return seg_text


def map_step_with_word_segments(original_step, edited_step_text):
    # 1. Unroll using WhisperX's pre-defined word segments
    # orig_words_tracked = []
    # for seg_idx, seg in enumerate(original_step["segments"]):
    #    for word_dict in seg.get("words", []):
    #        orig_words_tracked.append({"word": word_dict["word"], "seg_idx": seg_idx})

    orig_words_tracked = original_step["word_segments"]
    orig_norm = [normalize_word(item["word"]) for item in orig_words_tracked]
    # orig_norm = [item["word"] for item in orig_words_tracked]

    edit_words = edited_step_text.split()
    edit_norm = [normalize_word(w) for w in edit_words]

    # 2. Sequence Alignment
    matcher = difflib.SequenceMatcher(None, orig_norm, edit_norm)
    opcodes = matcher.get_opcodes()

    new_segment_words = [[]] * len(original_step["segments"])

    edit_mapped_step = {"segments": [], "word_segments": []}
    # 3. Distribute edits into segment buckets
    for tag, i1, i2, j1, j2 in opcodes:
        if tag in ("equal", "replace"):
            for orig_idx, edit_idx in zip(range(i1, i2), range(j1, j2)):
                seg_id = orig_words_tracked[orig_idx]["seg_idx"]
                new_segment_words[seg_id].append(edit_words[edit_idx])

        elif tag == "insert":
            seg_id = orig_words_tracked[i1 - 1]["seg_idx"] if i1 > 0 else 0
            for edit_idx in range(j1, j2):
                new_segment_words[seg_id].append(edit_words[edit_idx])

    # 4. Re-roll the segments and apply Kokoro Regex
    for i, seg in enumerate(original_step["segments"]):
        if len(new_segment_words[i]) > 0:
            # The raw text as edited by the user
            raw_segment_text = " ".join(new_segment_words[i])

            # --- APPLY KOKORO REGEX ---
            kokoro_text = normalize_segment_text(raw_segment_text)
            # --- DETECT EDITS ---
            # Compare the new raw text with the original WhisperX text
            original_text_clean = " ".join(seg["text"].split())
            is_edited = raw_segment_text != original_text_clean

            edit_mapped_step["segments"].append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "orig_text": seg["text"],
                    "text": raw_segment_text,  # Use this for UI / Subtitles
                    "kokoro_text": kokoro_text,  # Feed this directly into Kokoro
                    "is_edited": is_edited,  # True if the user changed this segment
                    "words": new_segment_words,
                }
            )

    return edit_mapped_step


def load_spacy_model():
    dev_fallback = "en_core_web_sm"
    if hasattr(sys, "_MEIPASS"):
        # Start at the root of your model collection
        base_search_path = Path(sys._MEIPASS) / dev_fallback

        # 1. Check if the config is right here
        if (base_search_path / "config.cfg").exists():
            return base_search_path

        # 2. Check one level deeper (for that 3.8.0 folder)
        if base_search_path.exists():
            for path in base_search_path.iterdir():
                if path.is_dir() and (path / "config.cfg").exists():
                    return path

    return dev_fallback


@lru_cache(maxsize=5)
def load_kokoro_pipeline(lang_code, model):
    print(f"Loading kokoro KPipeline({lang_code}, {model})...")

    return KPipeline(lang_code, model=model)


@lru_cache(maxsize=128)
def text_to_speech(text, lang_code="a", model=None, voice="af_heart", speed=1.0):
    pipeline = load_kokoro_pipeline(lang_code, model or load_spacy_model())
    return next(pipeline(text, voice=voice, speed=speed))


def word_align_seg_kkr_to_wspx(
    seg_aligned_wspx,
    seg_kkr_pred,
    sample_rate=24000,
    init_offset=0,
    time_margin=0,
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


def crossfade_combine(a1, a2, cross_ms, sample_rate=24000):
    crossfade_samples = min(int(cross_ms / 1000.0) * sample_rate, len(a1), len(a2))
    if crossfade_samples <= 0:
        return torch.cat([a1, a2])

    fade_out = torch.linspace(1.0, 0.0, crossfade_samples)
    fade_in = torch.linspace(0.0, 1.0, crossfade_samples)

    pre_fade = a1[:-crossfade_samples]
    fade1 = a1[-crossfade_samples:]
    fade2 = a2[:crossfade_samples]
    post_fade = a2[crossfade_samples:]

    fade = fade1 * fade_out + fade2 * fade_in

    return torch.cat([pre_fade, fade, post_fade])


def step_text_to_speech(stepfile, outfile, step_text=None, sample_rate=24000, **kwargs):
    with stepfile.open("r") as fin:
        step = json.load(fin)

    step_segs = (
        map_step_with_word_segments(step, step_text)
        if step_text is not None
        else step["segments"]
    )
    seg_text_label = "kokoro_text" if step_text is not None else "text"

    total_offset = 0
    aligned_words = []

    for iseg, seg in enumerate(step_segs):
        seg_kkr_pred = text_to_speech(seg[seg_text_label], **kwargs)
        aligned_seg_words, total_offset = word_align_seg_kkr_to_wspx(
            seg, seg_kkr_pred, sample_rate, total_offset
        )

        aligned_words.extend(aligned_seg_words)

    aligned_step_crossfaded = aligned_words[0]
    for iword in range(1, len(aligned_words)):
        aligned_step_crossfaded = crossfade_combine(
            aligned_step_crossfaded, aligned_words[iword], 50
        )
    aligned_step = aligned_step_crossfaded / aligned_step_crossfaded.abs().max()

    # if outfile is not None:
    sf.write(str(outfile), aligned_step, samplerate=sample_rate)

    if step_text is None:
        return outfile
    else:
        return (outfile, step_segs)

    # return aligned_step
