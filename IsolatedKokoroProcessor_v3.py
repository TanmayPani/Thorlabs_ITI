import itertools
import wx
import sys
import spacy
from os import path, listdir, makedirs, rename, environ, pathsep
import multiprocessing
import importlib.util
from multiprocessing import Process, set_start_method, freeze_support

# import imageio_ffmpeg
# ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()


def setup_ffmpeg():
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = path.dirname(path.abspath(__file__))

    # The folder where you put ffmpeg.exe, ffprobe.exe, ffplay.exe
    ffmpeg_dir = path.join(base_path, "ffmpeg", "bin")
    current_path = environ.get("PATH", "")
    new_path = f"{current_path}{pathsep}{ffmpeg_dir}"
    environ["PATH"] = new_path


setup_ffmpeg()


def load_spacy_model():
    if hasattr(sys, "_MEIPASS"):
        # Start at the root of your model collection
        base_search_path = path.join(sys._MEIPASS, "en_core_web_sm")

        # 1. Check if the config is right here
        if path.exists(path.join(base_search_path, "config.cfg")):
            return base_search_path

        # 2. Check one level deeper (for that 3.8.0 folder)
        if path.exists(base_search_path):
            for folder in listdir(base_search_path):
                subfolder = path.join(base_search_path, folder)
                if path.isdir(subfolder) and "config.cfg" in listdir(subfolder):
                    return subfolder

    return "en_core_web_sm"  # Dev fallback


import torch

torch.serialization.add_safe_globals(
    ["omegaconf.listconfig.ListConfig", "omegaconf.dictconfig.DictConfig"]
)

import traceback
import pickle
import shutil
from natsort import natsorted
import whisperx
import numpy as np
from re import findall, sub
import soundfile as sf
from pathlib import Path

from kokoro.pipeline import KPipeline
from threading import Thread
from pandas import read_excel, read_pickle, DataFrame, read_csv
from moviepy import concatenate_videoclips, concatenate_audioclips
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

from wxSlides import wxPresentation, wxTextBox


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def SafeLog(message):
    if wx.IsMainThread():
        wx.LogMessage(message)
    else:
        wx.CallAfter(wx.LogMessage, message)


def ComponentWriter(fpath, name, comps):
    with open(fpath + name, "wb") as f:
        pickle.dump(comps, f)


def ComponentReader(fpath, name):
    with open(fpath + name, "rb") as f:
        comps = pickle.load(f)
    return comps


def StandardizedExcelReader(fpath):
    Files = [
        i for i in listdir(fpath) if ".xlsm" in i or ".xlsx" in i if "copy Copy" in i
    ]
    # print(Files)
    FullSheet = read_excel(fpath + Files[0], sheet_name=None)
    SheetData = FullSheet.items()
    keys, dfs = [[] for i in range(2)]
    for key, df in SheetData:
        keys.append(key)
        dfs.append(df)
    matches = []
    StepKeys, StepPIDs, StepTools = [[] for i in range(3)]
    count = 0
    for i in dfs:
        if keys[count] == "Master_BOM":
            StepKeys.append(keys[count])
            tmp = i[1:]
            StepPIDs.append(tmp)
            StepTools.append([])
        # print(keys[count])
        if keys[count] != "Master_BOM" and keys[count] != "BOM_Export":
            print(keys[count])
            if len(i["Item number"]) > 0:
                mask = i["Item number"] == "Tool"
                i["grouper"] = mask.cumsum()
                ig = {
                    group_key: group_df for group_key, group_df in i.groupby("grouper")
                }
                # print(ig[1])
                ig[1].columns = ig[1].iloc[0]
                ig[1] = ig[1].loc[:, :"Quantity"]
                tmpKeys = ig[0].keys()
                matches = [j for j in tmpKeys if "Unnamed".casefold() in j.casefold()]
                matches.append("grouper")
                StepKeys.append(keys[count])
                # print(ig[0])
                StepPIDs.append(
                    # ig[0].dropna().drop(["grouper"], axis=1)
                    # ig[0].dropna().reset_index(drop=True).drop(matches, axis=1)
                    ig[0].drop(matches, axis=1).reset_index(drop=True)
                    # ig[0].dropna().reset_index(drop=True)
                )
                StepTools.append(
                    ig[1]["Tool Description"].dropna().reset_index(drop=True)[1:]
                )

        count += 1
    return StepKeys, StepPIDs, StepTools


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


def AudioWriter(
    pipeline, step_text, step_word_timings, step_timing, tag, index, sample_rate=24000
):
    # Changelines = StepSegments
    # tmp = sub(r" ?\[.*?\]", "", Changelines).replace("-", " dash ")
    tmp = sub(r" ?\[.*?\]", "", step_text).replace("-", " dash ")
    FText = [sub(r"(\d)", r"\1 ", tmp)]
    generator = pipeline(
        FText,
        voice="af_heart",  # <= change voice here
        speed=0.9,
        split_pattern=r"\n+",
    )
    # print(len(step_word_timings))
    # print(step_word_timings)
    # for w in step_word_timings:
    #    print(w)
    word_start_times = [
        word_ts["start"] for word_ts in step_word_timings if "start" in word_ts
    ]

    aligned_kkr_word_segs = []
    total_offset = 0
    time_margin = 0.0001

    # for ikkr_pred, kkr_pred in enumerate(generator):
    # for itoken, token in enumerate(itertools.chain.from_iterable(kkr_pred.tokens for kkr_pred in generator)):
    kkr_pred = next(generator)
    kkr_audio_tensor = kkr_pred.audio
    for itoken, token in enumerate(kkr_pred.tokens):
        if len(word_start_times) > itoken:
            offset = round(word_start_times[itoken] * sample_rate)
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

    aligned_kkr_step = torch.cat(aligned_kkr_word_segs)
    filename = tag + "FullStep" + str(index) + ".mp3"
    sf.write(filename, aligned_kkr_step, samplerate=sample_rate)
    return filename


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Thor(wx.Frame):
    def __init__(self, parent, id, nlp):
        super().__init__(
            parent,
            id,
            "Thorlabs Instruction Transcription Interface",
            size=(1400, 1200),
        )
        self.nlp = nlp

        # wx.Frame.__init__(self,parent,id,'Thorlabs Origin Interface',size=(750,500))

        self.panel = wx.Panel(self)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.presMaker = wxPresentation(self.panel)

        # Displaying Text
        # font = wx.Font(10, wx.DECORATIVE, wx.NORMAL, wx.BOLD)
        buttonSizer = wx.StaticBoxSizer(wx.VERTICAL, self.panel)

        self.BOMWriterCB = wx.ComboBox(
            buttonSizer.StaticBox,
            wx.ID_ANY,
            choices=["No BOM", "BOM"],
            style=wx.CB_READONLY,
        )
        self.BOMWriterCB.SetValue("No BOM")
        buttonSizer.Add(self.BOMWriterCB, wx.SizerFlags(0).Align(wx.TOP).Border(wx.ALL))

        CPS = wx.Button(buttonSizer.StaticBox, label="Core Path")
        buttonSizer.Add(CPS, wx.SizerFlags(0).Align(wx.TOP).Border(wx.ALL))

        self.VideoCombButton = wx.Button(buttonSizer.StaticBox, label="Compress Video")
        buttonSizer.Add(
            self.VideoCombButton, wx.SizerFlags(0).Align(wx.TOP).Border(wx.ALL)
        )

        self.VideoSliceButton = wx.Button(
            buttonSizer.StaticBox, label="Slice Video Steps"
        )
        buttonSizer.Add(
            self.VideoSliceButton, wx.SizerFlags(0).Align(wx.TOP).Border(wx.ALL)
        )

        self.RerenderButton = wx.Button(buttonSizer.StaticBox, label="Rerender Steps")
        buttonSizer.Add(
            self.RerenderButton, wx.SizerFlags(0).Align(wx.BOTTOM).Border(wx.ALL)
        )
        self.RerenderButton.Disable()

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        middleSizer = wx.BoxSizer(wx.HORIZONTAL)
        middleSizer.Add(buttonSizer, wx.SizerFlags(0).Expand().Border(wx.RIGHT))
        middleSizer.Add(self.presMaker, wx.SizerFlags(1).Expand())
        mainSizer.Add(middleSizer, wx.SizerFlags(1).Expand().Border())

        self.saveFileTBox = wxTextBox(self.panel, title="Save To:")
        self.saveFileTBox.Text = "example_presentation"
        self.saveFileButton = wx.Button(self.saveFileTBox.StaticBox, label="Save")
        self.saveFileButton.Disable()
        self.saveFileTBox.Add(self.saveFileButton, wx.SizerFlags(0).Border())
        mainSizer.Add(self.saveFileTBox, wx.SizerFlags(0).Expand().Border())

        self.logCtrl = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.logCtrl.Hide()
        mainSizer.Add(self.logCtrl, wx.SizerFlags(1).Expand().Border())

        self.showLogButton = wx.ToggleButton(self.panel, label="Show Log")
        mainSizer.Add(
            self.showLogButton,
            wx.SizerFlags(0).Align(wx.BOTTOM | wx.LEFT).Border(),
        )

        self.panel.SetSizer(mainSizer)

        self.logger = wx.LogTextCtrl(self.logCtrl)
        wx.Log.SetActiveTarget(self.logger)

        # self.Bind(wx.EVT_COMBOBOX, self.on_combo_selection, self.AudioWriterCB)
        # self.Bind(wx.EVT_COMBOBOX, self.BOMSelection, self.BOMWriterCB)
        self.Bind(wx.EVT_BUTTON, self.PathSelector, CPS)
        self.Bind(wx.EVT_BUTTON, self.VideoCombination, self.VideoCombButton)
        self.Bind(wx.EVT_BUTTON, self.TranscriptionModel, self.VideoSliceButton)
        self.Bind(wx.EVT_BUTTON, self.TranscriptionModel, self.RerenderButton)
        self.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggleLog, self.showLogButton)
        self.Bind(wx.EVT_BUTTON, self.OnSavePPTX, self.saveFileButton)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.CorePath = None
        self.StructurePath = None
        self.AudioPath = None
        self.VideoPath = None
        self.TextPath = None
        self.SegPath = None

        # self.WordSegments = None
        self.StepAudio = None
        self.StepVideo = None
        self.VideoClips = None
        self.ChangeLines = None
        self.VideoSlicerButton = None

        status = self.CreateStatusBar()
        status_font = self.GetStatusBar().GetFont()

        new_font_size = status_font.GetPointSize() + 2  # Increase size by 2 points
        new_font = wx.Font(
            new_font_size,
            status_font.GetFamily(),
            status_font.GetStyle(),
            status_font.GetWeight(),
            status_font.GetUnderlined(),
            status_font.GetFaceName(),
        )
        self.GetStatusBar().SetFont(new_font)
        self.Show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def OnToggleLog(self, event):
        doShow = self.showLogButton.GetValue()
        if doShow:
            self.logCtrl.Show()
            self.showLogButton.SetLabel("Hide Log")
        else:
            self.logCtrl.Hide()
            self.showLogButton.SetLabel("Show Log")
        self.panel.Layout()

    def OnSavePPTX(self, event):
        savePath = self.CorePath + self.saveFileTBox.Text + ".pptx"
        wx.LogMessage(f"Saving generated slides to {savePath}")
        self.presMaker.Save(savePath)

    def OnEnableRerender(self, event):
        if not self.RerenderButton.IsEnabled():
            self.RerenderButton.Enable()
            self.Layout()
            # self.Update()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def PathSelector(self, e):

        fileDialog = wx.DirDialog(
            frame, "Choose a directory", "", wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST
        )
        if fileDialog.ShowModal() == wx.ID_OK:
            result = fileDialog.GetPath().replace("\\", "/") + "/"
            self.CorePath = result
            Structure = listdir(result)
            self.status = self.SetStatusText(
                "Successfully loaded: " + result + " -- Structure: " + str(Structure)
            )
        else:
            self.status = self.SetStatusText(
                "Command Terminated."
            )  # Untoggle the button
        #                 print(self.filepath)
        if "BOM" not in str(Structure):
            makedirs((result + "BOM/"), exist_ok=True)
            self.StructurePath = result + "BOM/"
            self.FileMover()
            self.StructurePath = None
        else:
            self.StructurePath = result + "BOM/"
            self.FileMover()
            self.StructurePath = None
        if "Videos" not in str(Structure):
            makedirs((result + "Videos/"), exist_ok=True)
            self.StructurePath = result + "Videos/"
            self.FileMover()
            self.StructurePath = None
        else:
            self.StructurePath = result + "Videos/"
            self.FileMover()
            self.StructurePath = None
        if "StepSegs" not in str(Structure):
            makedirs((result + "StepSegs/"), exist_ok=True)
        if "StepSegsAudio" not in str(Structure):
            makedirs((result + "StepSegsAudio/"), exist_ok=True)
        if "StepSegsTxt" not in str(Structure):
            makedirs((result + "StepSegsTxt/"), exist_ok=True)

        self.TextPath = self.CorePath + "StepSegsTxt/"
        self.VideoPath = self.CorePath + "Videos/"
        self.SegPath = self.CorePath + "StepSegs/"
        self.AudioPath = self.CorePath + "StepSegsAudio/"
        # self.AudioWriterCB.Show()
        # self.LCB.Show()
        for i in range(1, self.presMaker.GetPageCount()):
            self.presMaker.DeletePage(i)
            self.panel.Layout()

    def FileMover(self):
        source_path = self.CorePath
        destination_path = self.StructurePath
        if "BOM" in self.StructurePath:
            CoreFiles = [i for i in listdir(source_path) if ".xlsm" in i]
        if "Videos" in self.StructurePath:
            CoreFiles = [
                i for i in listdir(source_path) if ".MP4".casefold() in i.casefold()
            ]
        if len(CoreFiles) > 0:
            # try:
            # Move the files
            for i in CoreFiles:
                print(str(source_path) + "/" + str(i), str(destination_path))
                shutil.move(str(source_path) + "/" + str(i), str(destination_path))
                self.status = self.SetStatusText(
                    "Moved files into -- "
                    + str(destination_path)
                    + str(CoreFiles[0])
                    + " successfully."
                )  # Untoggle the button

        else:
            self.status = self.SetStatusText(
                "Nothing to move, or Error moving file into -- "
                + str(destination_path)
                + ". Verify it exists in core directory or if moved earlier."
            )  # Untoggle the button

    def VideoCombination(self, e):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        wx.LogMessage("Begin video combination/renaming.")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # logger = (self.statusbar)
        # Corepath = self.CorePath
        # VideoPath=Corepath+'Videos/';self.VideoPath = VideoPath;self.SegPath = self.CorePath+'StepSegs/'#; self.AudioPath = Corepath+'StepSegsAudio/'
        VideoPath = self.VideoPath
        wx.CallAfter(wx.LogMessage, VideoPath)
        self.VideoCombButton.Disable()
        self.Layout()
        # self.Update()

        def work():
            try:
                if "combined.mp4" not in listdir(VideoPath):
                    VFiles = [
                        i
                        for i in listdir(VideoPath)
                        if ".MP4".casefold() in i.casefold()
                        if ".jpg" not in i
                    ]
                    # print(VFiles)
                    if len(VFiles) > 1:
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        wx.CallAfter(
                            wx.LogMessage,
                            "Multiple videos identified. Initiate sorting sequence.",
                        )
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        SegVidInds = natsorted(VFiles)
                        #                 SegVidInds = np.argsort(np.concatenate([np.double(findall(r'\d+', i.replace('.MP4'.casefold(),'').replace(Corepath+"Videos/",''))) for i in VFiles]))
                        #                 SortedVFiles = [VideoFileClip(VideoPath+VFiles[SegVidInds[i]]) for i in range(len(SegVidInds))]
                        # SortedVFiles = [
                        #     VideoFileClip(VideoPath + i) for i in SegVidInds
                        # ]
                        SortedVFiles = [
                            VideoFileClip(VideoPath + i).resized(0.5)
                            for i in SegVidInds
                        ]
                        SortedVFiles[0].save_frame(
                            self.CorePath + "StepSegs/" + "FirstFrame.jpg", t=0
                        )
                        wx.CallAfter(
                            wx.LogMessage, "Sorting complete. Initiate concatenation."
                        )

                        CombinedVid = concatenate_videoclips(SortedVFiles)
                        wx.CallAfter(
                            wx.LogMessage,
                            "Concatenation complete. Writing video to output. Please be patient.",
                        )
                        if self.get_best_codec() == "h264_nvenc":
                            preset = "p1"
                        else:
                            preset = "veryfast"
                        CombinedVid.write_videofile(
                            VideoPath + "combined.mp4",
                            temp_audiofile="temp-audio.m4a",
                            remove_temp=True,
                            audio_codec="aac",
                            codec=self.get_best_codec(),
                            # codec="libx264",
                            threads=8,
                            logger=None,
                            preset=preset,
                        )
                        # CombinedVid.write_videofile(Corepath+'/Videos/combined.mp4',temp_audiofile="temp-audio.m4a", remove_temp=True, audio_codec="aac", codec="libx264",logger=logger)
                        # wx.CallAfter(self.statusbar.SetStatusText, "Export Complete!")
                        # video = VideoFileClip(VideoPath+'combined.mp4')
                        wx.CallAfter(
                            wx.LogMessage,
                            "Video writing complete. Now writing audio to output.",
                        )

                        CombinedVid.audio.write_audiofile(
                            self.CorePath + "StepSegsAudio/" + "combined.mp3"
                        )

                        #                         CombinedVid.audio.write_audiofile(self.CorePath+'StepSegsAudio/'+"wavf_subs.wav", fps=16000, nbytes=2, codec='pcm_s16le', logger=None)
                        # self.AudioPath = self.CorePath+'StepSegsAudio/'
                        CombinedVid.audio.close()
                        CombinedVid.close()
                    if len(VFiles) == 1:
                        wx.CallAfter(
                            wx.LogMessage,
                            "Single video identified. Initiate renaming logic.",
                        )
                        rename(VideoPath + VFiles[0], VideoPath + "combined.mp4")
                        video = VideoFileClip(VideoPath + "combined.mp4")
                        video.save_frame(
                            self.CorePath + "StepSegs/" + "FirstFrame.jpg", t=0
                        )
                        wx.CallAfter(wx.LogMessage, "Audio embedding start.")
                        video.audio.write_audiofile(
                            self.CorePath + "StepSegsAudio/" + "combined.mp3"
                        )
                        #                 video.audio.write_audiofile(self.CorePath+'StepSegsAudio/'+"combined.wav", fps=16000, nbytes=2, codec='pcm_s16le', logger=None)
                        #                         CombinedVid.audio.write_audiofile(self.CorePath+'StepSegsAudio/'+"wavf_subs.wav", fps=16000, nbytes=2, codec='pcm_s16le', logger=None)
                        video.audio.close()
                        video.close()
                    if len(VFiles) == 0:
                        wx.CallAfter(wx.LogMessage, "No videos avaialable in path :(")

                        # self.status = self.SetStatusText('No videos avaiable in path :(')  # Untoggle the button
                else:
                    video = VideoFileClip(VideoPath + "combined.mp4")
                    video.save_frame(
                        self.CorePath + "StepSegs/" + "FirstFrame.jpg", t=0
                    )
                    video.close()
                    wx.CallAfter(
                        wx.LogMessage,
                        "combined.mp4 already exists. Please delete or move if another instance needs to be created.",
                    )

                    # self.status = self.SetStatusText('combined.mp4 already exists. Please delete or move if another instance needs to be created.')  # Untoggle the button
                    wx.CallAfter(
                        wx.LogMessage,
                        "Rewriting audio of existing combined.mp4 file.",
                    )
                    video = VideoFileClip(VideoPath + "combined.mp4")
                    video.audio.write_audiofile(
                        self.CorePath + "StepSegsAudio/" + "combined.mp3"
                    )
                wx.CallAfter(
                    wx.LogMessage,
                    "Audio successfully embedded. Please proceed to next step.",
                )

            except Exception as e:
                # wx.CallAfter(wx.LogMessage, f"Error: {(AudioFile+'combined.mp3')}")
                wx.CallAfter(wx.LogMessage, f"Error: {traceback.format_exc()}")
                wx.CallAfter(self.VideoSliceButton.Enable)

        Thread(target=work, daemon=True).start()
        self.VideoCombButton.Enable()
        self.Layout()
        # self.Update()

    def TranscriptionModel(self, e: wx.CommandEvent):
        self.status = self.SetStatusText("Transcription sequence initiated.")
        if self.AudioPath == "":
            raise Exception("Please enter valid filepath")
        self.VideoSliceButton.Disable()
        self.Layout()
        # self.Update()
        self.status = self.SetStatusText("Initiating thread...")  # Untoggle the button

        def work(isRerender, device, compute_type, batch_size):
            try:
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if not isRerender:
                    wx.CallAfter(
                        wx.LogMessage, f"Loading & Transcribing...  {str(device)}"
                    )
                    wx.CallAfter(
                        wx.LogMessage, "Loading large-v3 model from whisperx..."
                    )
                    wspModelLargeV3 = whisperx.load_model(
                        "large-v3", device, compute_type=compute_type
                    )
                    wx.CallAfter(wx.LogMessage, "Loading audio...")
                    loadedAudio = whisperx.load_audio(
                        Path(self.AudioPath + "combined.mp3")
                    )
                    wx.CallAfter(wx.LogMessage, "Initiating transcription...")
                    initialTranscription = wspModelLargeV3.transcribe(
                        loadedAudio, batch_size=batch_size, language="en"
                    )
                    wx.CallAfter(wx.LogMessage, "Aligining initial transcription...")
                    alignModel, modelMetadata = whisperx.load_align_model(
                        language_code="en", device=device
                    )
                    alignedTranscription = whisperx.align(
                        initialTranscription["segments"],
                        alignModel,
                        modelMetadata,
                        self.AudioPath + "combined.mp3",
                        device,
                        return_char_alignments=False,
                    )
                    wx.CallAfter(wx.LogMessage, "Begin writing.")
                    # self.WordSegments = alignedTranscription["word_segments"]
                    wx.CallAfter(
                        wx.LogMessage,
                        "Alignment sequence complete. Initiate step isolation sequence.",
                    )
                    self.FullSteps, self.StichedSteps, self.StichedTiming = TimeSlicer(
                        alignedTranscription["word_segments"]
                    )
                    wx.CallAfter(
                        wx.LogMessage,
                        "Step Isolation sequence complete. Obtaining audio slices",
                    )

                else:
                    self.StichedSteps = []
                    for i in range(1, self.presMaker.GetPageCount()):
                        self.presMaker.GetPage(i).shapes["movie"][0].movieCtrl.Stop()
                        self.presMaker.GetPage(i).shapes["movie"][0].movieCtrl.Load("")
                        self.StichedSteps.append(
                            self.presMaker.GetPage(i).shapes["textbox"][1].Text
                        )

                self.StepAudio = [
                    AudioWriter(
                        self.nlp,
                        stepText,
                        fullStep,
                        self.AudioPath + "AFHeart",
                        index,
                    )
                    for index, (stepText, fullStep) in enumerate(
                        zip(self.StichedSteps, self.FullSteps)
                    )
                ]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                wx.CallAfter(
                    wx.LogMessage, "Audio slices obtained. Initiate video rendering."
                )
                self.StepVideo = [
                    self.VideoStepWriter(self.StichedTiming[i], i)
                    for i in range(len(self.StichedTiming))
                ]
                wx.CallAfter(
                    wx.LogMessage,
                    "Videos successfully rendered. Presentation creation initiated.",
                )

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if isRerender:
                    wx.CallAfter(wx.LogMessage, "Rerendering videos.")

                    wx.CallAfter(self.ReloadVideos)
                else:
                    wx.CallAfter(wx.LogMessage, "Starting presentation.")
                    if self.BOMWriterCB.GetValue() == "BOM":
                        wx.CallAfter(wx.LogMessage, "Extracting BOM data.")
                        self.BOMWriter()
                        wx.CallAfter(
                            wx.LogMessage, "BOM data extracted. Starting presentation."
                        )
                    wx.CallAfter(self.AddSlides)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                wx.CallAfter(
                    wx.LogMessage,
                    "KokoroAI Processes Complete. Thank you for your patience!",
                )  # Thread-safe update

            except Exception as e:
                # wx.CallAfter(wx.LogMessage, f"Error: {(AudioFile+'combined.mp3')}")
                wx.CallAfter(wx.LogMessage, f"Error: {traceback.format_exc()}")
                wx.CallAfter(self.VideoSliceButton.Enable)

        # 2. Fire it off in a thread immediately
        isRerender = e.GetEventObject().GetLabel() == "Rerender Steps"
        Thread(target=work, args=(isRerender, "cpu", "int8", 1), daemon=True).start()
        # wx.LogMessage("Processing...")
        #         self.StepVideo = [self.VideoStepWriter(self.StichedTiming[i],i) for i in range(len(self.StichedTiming))]
        #         self.status = self.SetStatusText('Excel Gener.')  # Untoggle the button
        self.VideoSliceButton.Enable()
        self.Layout()
        # self.Update()

    def get_best_codec(self):
        # """Checks FFmpeg for available hardware encoders and returns the best one."""
        # try:
        #     # Get list of available encoders from FFmpeg
        #     result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
        #     encoders = result.stdout

        #     if 'h264_nvenc' in encoders:
        #         return 'h264_nvenc'         # NVIDIA GPU
        #     elif 'h264_videotoolbox' in encoders:
        #         return 'h264_videotoolbox'  # macOS (Apple Silicon/Intel)
        #     elif 'h264_qsv' in encoders:
        #         return 'h264_qsv'           # Intel QuickSync
        # except Exception:
        #     pass

        return "libx264"  # Standard CPU fallback

    def VideoStepWriter(self, e, Index):
        # print('Start Video Step Writing')
        VideoPath = self.VideoPath
        AudioPath = self.AudioPath
        Timing = self.StichedTiming
        VideoClips = []
        # NoSubVideo = VideoFileClip(VideoPath+'combined.mp4').without_audio()
        Video = VideoFileClip(VideoPath + "combined.mp4").resized(0.5).without_audio()
        # Video = VideoFileClip(VideoPath + "combined.mp4").without_audio()
        OGAudVideo = VideoFileClip(VideoPath + "combined.mp4").resized(
            0.5
        )  # .without_audio()
        #         Video = CompositeVideoClip([NoSubVideo, self.subs])

        MiniAud = AudioFileClip(self.StepAudio[Index])
        MiniOgAudVid = OGAudVideo[Timing[Index][0] : Timing[Index][1]]
        MiniVid = Video[Timing[Index][0] : Timing[Index][1]]
        MiniVid = MiniVid.with_audio(MiniAud)
        if self.get_best_codec() == "h264_nvenc":
            preset = "p1"
        else:
            preset = "veryfast"
        MiniVid.write_videofile(
            self.SegPath + "AFHeart" + str(Index) + ".mp4",
            # codec="libx264",
            codec=self.get_best_codec(),
            audio_codec="aac",
            preset=preset,
            logger=None,
            threads=8,
        )
        MiniOgAudVid.write_videofile(
            self.SegPath + "TmpOGAud" + str(Index) + ".mp4",
            # codec="libx264",
            codec=self.get_best_codec(),
            audio_codec="aac",
            preset=preset,
            logger=None,
            threads=8,
        )
        VideoClips.append(self.SegPath + "AFHeart" + str(Index) + ".mp4")
        self.VideoClips = VideoClips

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def BOMWriter(self):
        BOMPath = self.CorePath + "BOM/"
        BKeys, StepPIDs, StepTools = StandardizedExcelReader(BOMPath)

        ComponentWriter(BOMPath, "AFHeartTxt.pkl", [StepPIDs, StepTools])

    def AddSlides(self):
        SegInds = np.argsort(
            np.concatenate(
                [
                    np.double(
                        findall(
                            r"\d+", i.replace(".mp4", "").replace(self.SegPath, "")[-3:]
                        )
                    )
                    for i in listdir(self.SegPath)
                    if "TmpOGAud" in i
                ]
            )
        )

        FullStepVidPaths = [
            self.SegPath + i
            for i in listdir(self.SegPath)
            if i[-4:] != ".jpg"
            if "TmpOGAud" in i
        ]

        StepData = (
            read_pickle(self.CorePath + "BOM/AFHeartTxt.pkl")
            if self.BOMWriterCB.GetValue() == "BOM"
            else None
        )

        for i in range(len(FullStepVidPaths)):
            title = f"Step {i + 1}"
            # print(SegInds[i])
            BomTableData = (
                (StepData[0][i], DataFrame(list(StepData[1][i]), columns=["Tools"]))
                if StepData is not None
                else None
            )

            vidFileName = (
                FullStepVidPaths[SegInds[i]]
                if SegInds[i] < len(FullStepVidPaths)
                else None
            )

            self.presMaker.AddStepSlide(
                title,
                self.StichedSteps[i],
                vidFileName,
                BomTableData,
                movie_thumbnail_file_name=self.CorePath + "StepSegs/FirstFrame.jpg",
            )

            self.Bind(
                wx.EVT_TEXT,
                self.OnEnableRerender,
                self.presMaker.GetPage(i + 1).shapes["textbox"][1].textCtrl,
            )

        self.saveFileButton.Enable()
        self.RerenderButton.Enable()

    def ReloadVideos(self):

        SegInds = np.argsort(
            np.concatenate(
                [
                    np.double(
                        findall(
                            r"\d+", i.replace(".mp4", "").replace(self.SegPath, "")[-3:]
                        )
                    )
                    for i in listdir(self.SegPath)
                    if "AFHeart" in i
                ]
            )
        )
        # for i in range(len(FullStepVidPaths)):
        #     if SegInds[i]<len(FullStepVidPaths)+1:
        #         self.DeletePage(i+1)
        FullStepVidPaths = [
            self.SegPath + i
            for i in listdir(self.SegPath)
            if i[-4:] != ".jpg"
            if "TmpOGAud" not in i
        ]

        for i in range(len(FullStepVidPaths)):
            if SegInds[i] < len(FullStepVidPaths):
                self.presMaker.GetPage(i + 1).shapes["movie"][0].LoadVideo(
                    FullStepVidPaths[SegInds[i]]
                )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":
    # REQUIRED: Must set 'spawn' before creating any processes (default on Windows)
    multiprocessing.freeze_support()

    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        # model_pth = force_load_spacy_model()
        model_pth = load_spacy_model()
        pipeline = KPipeline(lang_code="a", model=model_pth)
        # print("Model and Pipeline loaded successfully!")
        # nlp_model = load_spacy_model()
        # print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        nlp_model = None
    app = wx.App()
    frame = Thor(parent=None, id=-1, nlp=pipeline)
    frame.Show()
    app.MainLoop()
