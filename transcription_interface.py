import json
import traceback
import shutil
import multiprocessing
from pathlib import Path

from threading import Thread
from pandas import read_pickle, DataFrame  # , read_csv

import wx

from wxSlides import wxPresentation, wxTextBox
from snippets import (
    ffprobe_duration,
    process_bom_data,
    read_and_combine_videos,
    run_ffmpeg,
    speech_to_text,
    step_slicer,
    step_text_to_speech,
    subtitles_vf_arg,
    video_step_slicer,
    write_step_srt,
)


class ThorLabsITI(wx.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            id=wx.ID_ANY,
            title=kwargs.pop("title", "Thorlabs Instruction Transcription Interface"),
            size=kwargs.pop("size", (1400, 1200)),
            **kwargs,
        )

        self.panel = wx.Panel(self)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.presMaker = wxPresentation(self.panel)

        buttonSizer = wx.StaticBoxSizer(wx.VERTICAL, self.panel)

        self.BOMWriterCB = wx.ComboBox(
            buttonSizer.StaticBox,
            wx.ID_ANY,
            choices=["No BOM", "BOM"],
            style=wx.CB_READONLY,
        )
        self.BOMWriterCB.SetValue("No BOM")
        buttonSizer.Add(self.BOMWriterCB, wx.SizerFlags(0).Align(wx.TOP).Border(wx.ALL))

        self.combineVideoButton = wx.Button(
            buttonSizer.StaticBox, label="Compress Video"
        )
        buttonSizer.Add(
            self.combineVideoButton, wx.SizerFlags(0).Align(wx.TOP).Border(wx.ALL)
        )

        self.transcibeStepButton = wx.Button(
            buttonSizer.StaticBox, label="Transcribe Steps"
        )
        buttonSizer.Add(
            self.transcibeStepButton, wx.SizerFlags(0).Align(wx.TOP).Border(wx.ALL)
        )
        # self.transcibeStepButton.Disable()

        self.rerenderStepAudioButton = wx.Button(
            buttonSizer.StaticBox, label="Rerender Steps"
        )
        buttonSizer.Add(
            self.rerenderStepAudioButton,
            wx.SizerFlags(0).Align(wx.BOTTOM).Border(wx.ALL),
        )
        # self.rerenderStepAudioButton.Disable()

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        corePathPicker = wx.DirPickerCtrl(
            self.panel,
            message="Select core path",
            style=wx.DIRP_DEFAULT_STYLE,  # includes wx.DIRP_DIR_MUST_EXIST
        )
        mainSizer.Add(corePathPicker, wx.SizerFlags(0).Expand().Border())

        middleSizer = wx.BoxSizer(wx.HORIZONTAL)
        middleSizer.Add(buttonSizer, wx.SizerFlags(0).Expand().Border(wx.RIGHT))
        middleSizer.Add(self.presMaker, wx.SizerFlags(1).Expand())
        mainSizer.Add(middleSizer, wx.SizerFlags(1).Expand().Border())

        self.saveFileTBox = wxTextBox(self.panel, title="Save To:")
        self.saveFileTBox.Text = "example_presentation"
        self.saveFileButton = wx.Button(self.saveFileTBox.StaticBox, label="Save")
        # self.saveFileButton.Disable()
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
        self.Bind(wx.EVT_DIRPICKER_CHANGED, self.OnCorePathPicked, corePathPicker)
        self.Bind(wx.EVT_BUTTON, self.VideoCombination, self.combineVideoButton)
        self.Bind(wx.EVT_BUTTON, self.TranscribeSteps, self.transcibeStepButton)
        self.Bind(
            wx.EVT_BUTTON, self.OnRerenderStepAudios, self.rerenderStepAudioButton
        )
        self.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggleLog, self.showLogButton)
        self.Bind(wx.EVT_BUTTON, self.OnSavePPTX, self.saveFileButton)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.CorePath = None
        self.AudioPath = None
        self.VideoPath = None
        self.TextPath = None
        self.SegPath = None

        self.Steps = []

        # self.CreateStatusBar()
        # statusFont = self.GetStatusBar().GetFont()
        # self.GetStatusBar().SetFont(
        #    wx.Font(
        #        statusFont.GetPointSize() + 2,
        #        statusFont.GetFamily(),
        #        statusFont.GetStyle(),
        #        statusFont.GetWeight(),
        #        statusFont.GetUnderlined(),
        #        statusFont.GetFaceName(),
        #    )
        # )
        # self.Show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def OnToggleLog(self, evt):
        if evt.GetEventObject().GetValue():
            self.logCtrl.Show()
            evt.GetEventObject().SetLabel("Hide Log")
        else:
            self.logCtrl.Hide()
            evt.GetEventObject().SetLabel("Show Log")
        self.panel.Layout()

    def ShowLog(self):
        """Reveal the log panel (used to surface errors)."""
        self.logCtrl.Show()
        self.showLogButton.SetValue(True)
        self.showLogButton.SetLabel("Hide Log")
        self.panel.Layout()

    def ReportError(self, msg):
        """Main-thread error handler: log it, reveal the log, and alert the user."""
        wx.LogMessage(f"Error: {msg}")
        self.ShowLog()
        wx.MessageBox(msg, "Processing error", wx.OK | wx.ICON_ERROR)

    def SetBusy(self, busy):
        """Enable/disable the action buttons while a background worker runs."""
        for btn in (
            self.combineVideoButton,
            self.transcibeStepButton,
            self.rerenderStepAudioButton,
            self.saveFileButton,
        ):
            btn.Enable(not busy)

    def OnSavePPTX(self, event):
        if self.CorePath is None:
            self.ReportError(
                "No core path is selected! Please select a path before saving."
            )
            return
        savePath = self.CorePath / f"{self.saveFileTBox.Text}.pptx"
        wx.LogMessage(f"Saving generated slides to {savePath}")
        try:
            self.presMaker.Save(savePath)
            wx.LogMessage("Save complete.")
        except Exception:
            self.ReportError(traceback.format_exc())

    def OnEnableRerender(self, event):
        if not self.rerenderStepAudioButton.IsEnabled():
            self.rerenderStepAudioButton.Enable()
            self.Layout()
            # self.Update()

    def OnCorePathPicked(self, evt: wx.FileDirPickerEvent):
        # self.CorePath = Path(evt.GetPath().replace("\\", "/"))
        self.CorePath = Path(evt.GetPath())

        (self.CorePath / "BOM").mkdir(exist_ok=True)
        for bomFile in self.CorePath.glob("*.xlsm"):
            shutil.move(bomFile, self.CorePath / "BOM" / bomFile.name)

        self.VideoPath = self.CorePath / "Videos"
        self.VideoPath.mkdir(exist_ok=True)
        for videoFile in self.CorePath.glob("*.mp4"):
            # wx.LogMessage(f"moving {videoFile} to {self.VideoPath / videoFile.name}")
            shutil.move(
                videoFile,
                self.VideoPath / videoFile.name,
            )

        self.TextPath = self.CorePath / "StepSegsTxt"
        self.SegPath = self.CorePath / "StepSegs"
        self.AudioPath = self.CorePath / "StepSegsAudio"

        self.TextPath.mkdir(exist_ok=True)
        self.SegPath.mkdir(exist_ok=True)
        self.AudioPath.mkdir(exist_ok=True)

        for page_id in range(1, self.presMaker.GetPageCount()):
            self.presMaker.DeletePage(page_id)
            # self.panel.Layout()

    def VideoCombination(self, evt):
        if self.VideoPath is None:
            self.ReportError(
                "No core path is selected! Please select a path using the browser..."
            )
            return
        wx.LogMessage("Begin video combination/renaming.")
        wx.LogMessage(f"Reading video from: {self.VideoPath}")
        self.SetBusy(True)

        def worker():
            try:
                wx.CallAfter(wx.LogMessage, "Reading videos from ...")
                read_and_combine_videos(
                    self.VideoPath,
                    self.AudioPath,
                    self.SegPath / "FirstFrame.jpg",
                )
                wx.CallAfter(
                    wx.LogMessage, f"Combined video ready in {self.VideoPath}."
                )
            except Exception:
                wx.CallAfter(self.ReportError, traceback.format_exc())
            finally:
                wx.CallAfter(self.SetBusy, False)

        Thread(target=worker, daemon=True).start()
        self.Layout()

    def TranscribeSteps(self, evt):
        if self.AudioPath is None:
            self.ReportError(
                "No core path is selected! Please select a path using the browser..."
            )
            return

        self.SetBusy(True)

        def worker():
            try:
                wx.CallAfter(
                    wx.LogMessage,
                    "Loading whisper model for transcribing speech into text...",
                )

                speech_to_text(
                    self.AudioPath / "combined.mp3",
                    self.TextPath / "combined.json",
                    model_name="large-v3",
                    device="cpu",
                    compute_type="int8",
                    vad_method="silero",
                    batch_size=12,
                    chunk_size=30,
                )

                wx.CallAfter(
                    wx.LogMessage,
                    "Speech to text transcription and alignment done! slicing the text instructions into steps...",
                )

                self.Steps[:] = step_slicer(
                    self.TextPath / "combined.json", self.SegPath
                )

                wx.CallAfter(
                    wx.LogMessage, "Using kokoro TTS to obtain audio for steps..."
                )

                stepAudioOutput = [
                    step_text_to_speech(
                        slicedStep,
                        self.SegPath / f"AFHeartFullStep{iStep}.mp3",
                    )
                    for iStep, slicedStep in enumerate(self.Steps)
                ]

                wx.CallAfter(
                    wx.LogMessage,
                    "Step-by-step audio slices obtained! Initiating video rendering...",
                )

                stepwiseVideoOutput = video_step_slicer(
                    self.Steps,
                    self.VideoPath / "combined.mp4",
                    stepAudioOutput,
                    self.SegPath,
                )

                wx.CallAfter(
                    wx.LogMessage,
                    "Kokoro audio added to step by step video...",
                )

                if self.BOMWriterCB.GetValue() == "BOM":
                    wx.CallAfter(wx.LogMessage, "Extracting BOM data.")
                    process_bom_data(self.CorePath / "BOM", self.CorePath / "BOM")
                    wx.CallAfter(wx.LogMessage, "BOM data extracted...")
                wx.CallAfter(
                    wx.LogMessage,
                    "Speech-Text-Speech Processes Complete! Creating slides...",
                )
                wx.CallAfter(self.AddSlidesSafe)

                wx.CallAfter(
                    wx.LogMessage,
                    "All done, Thank you for your patience!",
                )

            except Exception:
                wx.CallAfter(self.ReportError, traceback.format_exc())
            finally:
                wx.CallAfter(self.SetBusy, False)

        Thread(target=worker, daemon=True).start()
        self.Layout()

    def OnRerenderStepAudios(self, evt):
        self.SetBusy(True)

        def worker():
            try:
                for istep in range(self.presMaker.GetPageCount() - 1):
                    print("Rerendering step", istep)
                    movie = self.presMaker.GetPage(istep + 1).shapes["movie"][0]
                    wx.CallAfter(movie.Stop)
                    wx.CallAfter(movie.Unload)
                    newStepTextsChanged = (
                        self.presMaker.GetPage(istep + 1).shapes["textbox"][1].Text
                    )
                    self.RerenderStepAudio(istep, newStepTextsChanged)

                    wx.CallAfter(
                        movie.LoadVideo,
                        str(self.SegPath / f"AFHeartEdited{istep}.mp4"),
                    )
            except Exception:
                wx.CallAfter(self.ReportError, traceback.format_exc())
            finally:
                wx.CallAfter(self.SetBusy, False)

        Thread(target=worker, daemon=True).start()
        self.Layout()

    def RerenderStepAudio(self, istep, newStepTextsChanged):

        print(newStepTextsChanged)

        audioClipPath = self.SegPath / f"AFHeartEditedStep{istep}.mp3"

        stepAudioFile, stepSegments = step_text_to_speech(
            self.SegPath / f"Step{istep}.json",
            audioClipPath,
            newStepTextsChanged,
        )

        # self.Steps[istep]["segments"][:] = stepSegments

        # Subtitles for the edited narration: stepSegments carries the user's
        # edited text mapped onto the TTS audio's word timings. The clip starts
        # at 0, so shift the absolute cue times by the step's start.
        srtPath = self.SegPath / f"AFHeartEdited{istep}.srt"
        clipStart = (
            stepSegments["segments"][0]["start"] if stepSegments["segments"] else 0.0
        )
        write_step_srt(stepSegments, srtPath, time_offset=clipStart)

        # Swap the edited TTS audio onto the existing (already half-res) step
        # video and burn in the subtitles. The `subtitles` filter forces a video
        # re-encode, so we can't `-c:v copy` here; `-t` caps the output at the
        # video's length so the visual isn't truncated.
        srcVideo = self.SegPath / f"AFHeart{istep}.mp4"
        run_ffmpeg(
            [
                "-i",
                str(srcVideo),
                "-i",
                str(audioClipPath),
                "-t",
                str(ffprobe_duration(srcVideo)),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                subtitles_vf_arg(srtPath),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(self.SegPath / f"AFHeartEdited{istep}.mp4"),
            ]
        )

    def AddSlidesSafe(self):
        """Build slides on the main thread, surfacing any failure to the user."""
        try:
            self.AddSlides()
        except Exception:
            self.ReportError(traceback.format_exc())

    def AddSlides(self):
        # vidFiles = {}
        # print("Adding slides...")
        # for vidFile in self.SegPath.glob("TmpOGAud*.mp4"):
        #    print(vidFile)
        #    vidFileStepId = int(vidFile.stem.removeprefix("TmpOGAud"))
        #    print(vidFileStepId)
        #    vidFiles[vidFileStepId] = str(vidFile)

        print("Adding slides...")
        StepData = (
            read_pickle(self.CorePath / "BOM" / "AFHeartTxt.pkl")
            if self.BOMWriterCB.GetValue() == "BOM"
            else None
        )

        print("Adding slides...")
        # print(self.Steps)
        for stepPath in sorted(
            self.SegPath.glob("Step*.json"),
            key=lambda p: int(p.stem.removeprefix("Step")),
        ):
            istep = int(stepPath.stem.removeprefix("Step"))
            title = f"Step {istep + 1}"
            # print(SegInds[i])
            # print(title, step)
            BomTableData = (
                (
                    StepData[0][istep],
                    DataFrame(list(StepData[1][istep]), columns=["Tools"]),
                )
                if StepData is not None
                else None
            )

            # Prefer the subtitled copy produced by video_step_slicer; fall back
            # to the bare clip for older/partial renders.
            vidWithSubs = self.SegPath / f"TmpOGAud{istep}_wSubs.mp4"
            vidFilePath = (
                vidWithSubs
                if vidWithSubs.exists()
                else self.SegPath / f"TmpOGAud{istep}.mp4"
            )

            with stepPath.open("r") as fin:
                step = json.load(fin)

            # print(f"{title}, {vidFilePath}")
            self.presMaker.AddStepSlide(
                title,
                step["text"],
                str(vidFilePath),
                BomTableData,
                movie_thumbnail_file_name=str(self.SegPath / "FirstFrame.jpg"),
            )
            # print(f"{title} slide made")

            # self.Bind(
            #    wx.EVT_TEXT,
            #    self.OnEnableRerender,
            #    self.presMaker.GetPage(istep + 1).shapes["textbox"][1].textCtrl,
            # )
            # print("Text controls bound")

        # self.saveFileButton.Enable()
        # self.rerenderStepAudioButton.Enable()


if __name__ == "__main__":
    # REQUIRED: Must set 'spawn' before creating any processes (default on Windows)
    multiprocessing.freeze_support()

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    app = wx.App()
    frame = ThorLabsITI(None)
    frame.Show()
    app.MainLoop()
