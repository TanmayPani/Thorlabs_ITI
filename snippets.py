import os
import sys
import pickle
import re
import string
import difflib
import json
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
import subprocess

from pandas import read_excel
from natsort import natsorted

import torch

torch.serialization.add_safe_globals(
    ["omegaconf.listconfig.ListConfig", "omegaconf.dictconfig.DictConfig"]
)


def setup_ffmpeg():
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    # The folder where you put ffmpeg.exe, ffprobe.exe, ffplay.exe
    ffmpeg_dir = os.path.join(base_path, "ffmpeg", "bin")
    # Prefer a bundled ffmpeg if present, but never clobber the existing PATH
    # (so the system ffmpeg is still found in a normal dev environment).
    if os.path.isdir(ffmpeg_dir):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


setup_ffmpeg()

import soundfile as sf
import whisperx
import whisperx.utils
from kokoro.pipeline import KPipeline


def run_ffmpeg(args):
    """Run `ffmpeg <args>` quietly, raising with stderr captured on failure."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args],
        check=True,
        capture_output=True,
        text=True,
    )


def ffprobe_dims(path):
    """Return (width, height) of the first video stream via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    width, height = (int(x) for x in out.split("x"))
    return width, height


def ffprobe_duration(path):
    """Return the duration in seconds of the media at `path` via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return float(out)


def process_bom_data(fpath, out_dir):
    print(fpath)
    bom_file_path = list(fpath.glob("*.xlsm"))[0]
    print(bom_file_path)
    df_dict = read_excel(bom_file_path, sheet_name=None)
    # keys = list(df_dict.keys())
    print(list(df_dict.keys()))
    matches = []
    steps = [[], [], []]
    # count = 0
    for idf, (df_key, df) in enumerate(df_dict.items()):
        print(idf)
        print(df_key)
        print(df.head())
        if df_key == "Master_BOM":
            # steps[0].append(df_key)
            # steps[1].append(df[1:])
            # steps[2].append([])
            pass
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
        print("Saving BOM pickle file...")


def read_and_combine_videos(video_dir, audio_dir, thumbnail_path=None):
    combined_path = video_dir / "combined.mp4"
    if not combined_path.exists():
        print("Creating combined.mp4...")
        video_files = natsorted(video_dir.glob("*.mp4"))

        if len(video_files) == 0:
            raise FileNotFoundError(f"No video files found in {str(video_dir)}!")

        # Everything downstream uses half-resolution video, so downscale once
        # here. Target = half the first clip's dimensions, rounded down to even
        # (libx264 / yuv420p require even dimensions).
        width, height = ffprobe_dims(video_files[0])
        tw, th = (width // 2) & ~1, (height // 2) & ~1

        if len(video_files) == 1:
            print(f"Creating combined.mp4 from {video_files[0]}...")
            run_ffmpeg(
                [
                    "-i",
                    str(video_files[0]),
                    "-vf",
                    f"scale={tw}:{th}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    str(combined_path),
                ]
            )
        else:
            inputs = []
            for fvid in video_files:
                inputs += ["-i", str(fvid)]
            n = len(video_files)
            # Scale each input to the common target, then concatenate v+a.
            filtergraph = "".join(
                f"[{i}:v]scale={tw}:{th},setsar=1[v{i}];" for i in range(n)
            )
            filtergraph += "".join(f"[v{i}][{i}:a]" for i in range(n))
            filtergraph += f"concat=n={n}:v=1:a=1[v][a]"
            run_ffmpeg(
                [
                    *inputs,
                    "-filter_complex",
                    filtergraph,
                    "-map",
                    "[v]",
                    "-map",
                    "[a]",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    str(combined_path),
                ]
            )

    # Extract the audio track and a poster thumbnail from the combined video.
    run_ffmpeg(
        [
            "-i",
            str(combined_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            str(audio_dir / "combined.mp3"),
        ]
    )
    if thumbnail_path is not None:
        run_ffmpeg(
            [
                "-ss",
                "0",
                "-i",
                str(combined_path),
                "-frames:v",
                "1",
                str(thumbnail_path),
            ]
        )


def write_step_srt(step, srt_path, language="en"):
    """Write an SRT for an in-memory step dict (segments + word timings).

    `step` may be a freshly-loaded Step JSON or the edit-mapped dict returned by
    `step_text_to_speech`; both carry `segments` with per-word `start`/`end`,
    which `highlight_words=True` needs. Returns the written SRT path.
    """
    srt_path = Path(srt_path)
    step.setdefault("language", language)
    whisperx.utils.WriteSRT(srt_path.parent)(
        step,
        srt_path,
        {
            "max_line_count": None,
            "max_line_width": None,
            "highlight_words": True,
        },
    )
    return srt_path


def subtitles_vf_arg(srt_path):
    """Build the ffmpeg `subtitles=` video-filter arg, Windows-path-safe."""
    escaped = Path(srt_path).resolve().as_posix().replace(":", r"\:")
    return f"subtitles='{escaped}'"


def add_subtitles_to_step(step_path, video_path, language="en"):
    with step_path.open("r") as fin:
        step = json.load(fin)

    srt_path = write_step_srt(step, step_path.with_suffix(".srt"), language)

    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-vf",
            subtitles_vf_arg(srt_path),
            str(video_path.with_stem(f"{video_path.stem}_wSubs")),
        ]
    )


def video_step_slicer(steps, video_path, step_audio_files, out_dir):
    assert len(steps) == len(step_audio_files), (
        f"Expected one audio file per step, but got {len(step_audio_files)} audio files for {len(steps)} steps!"
    )

    # combined.mp4 is already half-resolution (see read_and_combine_videos), so
    # no scaling is needed here. `-ss` before `-i` seeks quickly to the step
    # start; because we re-encode, the cut is frame-accurate. `-t` then limits
    # the output to the step duration.
    step_video_clips = []
    for istep, step_file in enumerate(steps):
        with step_file.open("r") as fin:
            step = json.load(fin)

        start = step["start"]
        duration = step["end"] - step["start"]

        af_heart = str(out_dir / f"AFHeart{istep}.mp4")
        step_video_clips.append(af_heart)

        # Step video with the regenerated TTS audio (video from input 0, audio
        # from the TTS mp3 in input 1).
        run_ffmpeg(
            [
                "-ss",
                str(start),
                "-i",
                str(video_path),
                "-i",
                str(step_audio_files[istep]),
                "-t",
                str(duration),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                af_heart,
            ]
        )

        # Step video keeping the original audio.
        run_ffmpeg(
            [
                "-ss",
                str(start),
                "-i",
                str(video_path),
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(out_dir / f"TmpOGAud{istep}.mp4"),
            ]
        )

        # Example usage for adding subtitles:
        add_subtitles_to_step(step_file, out_dir / f"TmpOGAud{istep}.mp4")

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

    # del transcription_model
    # gc.collect()
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
    # del audio, transcription, align_model, align_model_metadata
    # gc.collect()

    with text_output_path.open("w") as fout:
        json.dump(aligned_segments, fout, indent=4)

    aligned_segments.clear()
    # del aligned_segments
    # gc.collect()


def filter_punkt(word):
    rgx = f"[{re.escape(string.punctuation)}]"
    return re.sub(rgx, "", word).casefold()


def trigger_start_logic(steps):
    steps.append({"segments": [], "word_segments": []})


def trigger_end_logic(steps, new_words, new_segment):
    if new_segment:
        steps[-1]["segments"].append({"words": []})

    steps[-1]["segments"][-1]["words"].extend(new_words)
    steps[-1]["word_segments"].extend(new_words)


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
                "step"
                in filter_punkt(whisper_result["word_segments"][iword + 1]["word"])
            ):
                filtered_word = filter_punkt(word)
                # if start_triggered:
                #    trigger_end_logic(steps, [], current_seg < iseg)
                #    if current_seg < iseg:
                #        current_seg = iseg

                if not start_triggered:
                    if any(trg in filtered_word for trg in start_triggers):
                        start_triggered = True
                        # step_seg_idx = 0
                        # steps.append({"segments": [], "word_segments": []})
                        trigger_start_logic(steps)

                elif any(trg in filtered_word for trg in end_triggers):
                    # if current_seg < iseg:
                    #    steps[-1]["segments"].append({"words": []})
                    #    current_seg = iseg

                    # steps[-1]["segments"][-1]["words"].extend(
                    #    whisper_result["word_segments"][iword : iword + 3]
                    # )
                    # steps[-1]["word_segments"].extend(
                    #    whisper_result["word_segments"][iword : iword + 3]
                    # )
                    # if not start_triggered:
                    #    trigger_start_logic(steps)
                    trigger_end_logic(
                        steps,
                        whisper_result["word_segments"][iword : iword + 3],
                        current_seg < iseg,
                    )
                    if current_seg < iseg:
                        current_seg = iseg
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

    print(f"Step splitting done! Processing {len(steps)} steps")
    step_files = []
    for istep, step in enumerate(steps):
        print(f"Processing step {istep}")

        iword = 0
        for iseg, seg in enumerate(step["segments"]):
            step["segments"][iseg]["start"] = seg["words"][0]["start"]
            step["segments"][iseg]["end"] = seg["words"][-1]["end"]
            step["segments"][iseg]["text"] = " ".join(w["word"] for w in seg["words"])
            for w in seg["words"]:
                step["word_segments"][iword]["seg_idx"] = iseg
                iword += 1

        step["start"] = step["segments"][0]["start"]
        step["end"] = step["segments"][-1]["end"]
        step["text"] = " ".join(s["text"] for s in step["segments"])
        print(step["start"], step["end"], step["text"])
        step_out_path = out_dir / f"Step{istep}.json"
        print(f"Writing step {istep} to {step_out_path}")
        with step_out_path.open("w") as fout:
            json.dump(step, fout, indent=4)

        step_files.append(step_out_path)

    print("Step splitting done!")
    return step_files


def normalize_segment_text(seg_text):
    # 1. Remove bracketed text and replace hyphens
    seg_text = re.sub(
        r"\[.*\]", lambda m: "." * len(m.group(0)[1:-1]), seg_text
    ).replace("-", " dash ")
    # 2. Add spaces between digits for serial number reading
    seg_text = re.sub(r"(\d)", r"\1 ", seg_text)
    # Clean up any accidental double spaces created by the regex
    seg_text = re.sub(r"\s+", " ", seg_text).strip()

    return seg_text


def map_step_with_word_segments(orig_step, edited_step_text):
    proc_step_edit = normalize_segment_text(edited_step_text)

    new_segment_words = [[] for _ in orig_step["segments"]]

    iword = 0

    for diff in difflib.ndiff(orig_step["text"].split(), proc_step_edit.split()):
        match diff[:2]:
            case "? ":
                continue
            case "+ ":
                seg_idx = orig_step["word_segments"][iword - 1]["seg_idx"]
                new_segment_words[seg_idx].append(
                    {"word": diff[2:], "iword_orig": iword - 1}
                )
                # print(
                #    new_word_segments[seg_idx][-1]["word"],
                #    _step["word_segments"][iword - 1]["word"],
                #    iword - 1,
                # )
            case "- ":
                # orig_word_segments.append(d[2:])
                iword += 1
            case "  ":
                # orig_word_segments.append(d[2:])
                seg_idx = orig_step["word_segments"][iword]["seg_idx"]
                new_segment_words[seg_idx].append(
                    {"word": diff[2:], "iword_orig": iword}
                )
                iword += 1

                # print(
                #    new_word_segments[seg_idx][-1]["word"],
                #    _step["word_segments"][iword - 1]["word"],
                #    iword - 1,
                # )
            case _:
                raise ValueError(
                    'diff prefix can only be one of "? ", "+ ", "- ", "  "!!'
                )
    repeats = defaultdict(list)
    for iseg, seg in enumerate(new_segment_words):
        # print(iseg)
        for iwrd, wrd in enumerate(seg):
            # print(iseg, iwrd, wrd)
            repeats[wrd["iword_orig"]].append((iseg, iwrd))

    for iwrd_orig, new_word_idx in repeats.items():
        orig_start = orig_step["word_segments"][iwrd_orig]["start"]
        orig_end = orig_step["word_segments"][iwrd_orig]["end"]
        num_words = len(new_word_idx)
        if num_words == 1:
            iseg, iwrd = new_word_idx[0]
            new_segment_words[iseg][iwrd]["start"] = orig_start
            new_segment_words[iseg][iwrd]["end"] = orig_end

        if num_words > 1:
            dur_per_wrd = (orig_end - orig_start) / float(num_words + 1)

            for i, (iseg, iwrd) in enumerate(new_word_idx):
                new_segment_words[iseg][iwrd]["start"] = orig_start + i * dur_per_wrd
                new_segment_words[iseg][iwrd]["end"] = (
                    orig_start + (i + 1) * dur_per_wrd
                )

    # 4. Re-roll the segments and apply Kokoro Regex
    edit_mapped_step = {"segments": [], "word_segments": []}
    for i, seg in enumerate(orig_step["segments"]):
        if len(new_segment_words[i]) > 0:
            # The raw text as edited by the user
            raw_segment_text = " ".join(w["word"] for w in new_segment_words[i])

            # kokoro_text = normalize_segment_text(raw_segment_text)
            # --- DETECT EDITS ---
            # Compare the new raw text with the original WhisperX text
            is_edited = raw_segment_text != seg["text"]

            edit_mapped_step["segments"].append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "orig_text": seg["text"],
                    "text": raw_segment_text,  # Use this for UI / Subtitles
                    # "kokoro_text": kokoro_text,  # Feed this directly into Kokoro
                    "is_edited": is_edited,  # True if the user changed this segment
                    "words": new_segment_words[i],
                }
            )

            edit_mapped_step["word_segments"].extend(new_segment_words[i])

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

    # avg_word_dur = (seg_aligned_wspx["end"] - seg_aligned_wspx["start"]) / float(
    #    len(seg_aligned_wspx["words"])
    # )

    # print(avg_word_dur)
    # wspx_word_start_times = tuple(
    #    word["start"]
    #    if isinstance(word, dict)
    #    else seg_aligned_wspx["start"] + iword * avg_word_dur
    #    for iword, word in enumerate(seg_aligned_wspx["words"])
    # )
    wspx_word_start_times = tuple(word["start"] for word in seg_aligned_wspx["words"])
    # print(wspx_word_start_times)

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
        map_step_with_word_segments(step, step_text) if step_text is not None else step
    )
    # seg_text_label = "kokoro_text" if step_text is not None else "text"

    total_offset = 0
    aligned_words = []

    for iseg, seg in enumerate(step_segs["segments"]):
        seg_kkr_pred = text_to_speech(seg["text"], **kwargs)
        aligned_seg_words, total_offset = word_align_seg_kkr_to_wspx(
            seg, seg_kkr_pred, sample_rate, total_offset
        )

        aligned_words.extend(aligned_seg_words)

    aligned_step_crossfaded = aligned_words[0]
    for iword in range(1, len(aligned_words)):
        aligned_step_crossfaded = crossfade_combine(
            aligned_step_crossfaded, aligned_words[iword], 100
        )
    aligned_step = aligned_step_crossfaded / aligned_step_crossfaded.abs().max()

    # if outfile is not None:
    sf.write(str(outfile), aligned_step, samplerate=sample_rate)

    if step_text is None:
        return outfile
    else:
        return (outfile, step_segs)

    # return aligned_step
