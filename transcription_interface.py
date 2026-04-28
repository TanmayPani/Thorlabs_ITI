import json
import traceback
import shutil
import multiprocessing
from pathlib import Path

from threading import Thread
from pandas import read_pickle, DataFrame  # , read_csv
from moviepy import VideoFileClip, AudioFileClip  # , CompositeAudioClip

import wx

from wxSlides import wxPresentation, wxTextBox
from snippets import (
    process_bom_data,
    read_and_combine_videos,
    speech_to_text,
    step_slicer,
    step_text_to_speech,
    video_step_slicer,
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

        combineVideoButton = wx.Button(buttonSizer.StaticBox, label="Compress Video")
        buttonSizer.Add(
            combineVideoButton, wx.SizerFlags(0).Align(wx.TOP).Border(wx.ALL)
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

        showLogButton = wx.ToggleButton(self.panel, label="Show Log")
        mainSizer.Add(
            showLogButton,
            wx.SizerFlags(0).Align(wx.BOTTOM | wx.LEFT).Border(),
        )

        self.panel.SetSizer(mainSizer)

        self.logger = wx.LogTextCtrl(self.logCtrl)
        wx.Log.SetActiveTarget(self.logger)

        # self.Bind(wx.EVT_COMBOBOX, self.on_combo_selection, self.AudioWriterCB)
        # self.Bind(wx.EVT_COMBOBOX, self.BOMSelection, self.BOMWriterCB)
        self.Bind(wx.EVT_DIRPICKER_CHANGED, self.OnCorePathPicked, corePathPicker)
        self.Bind(wx.EVT_BUTTON, self.VideoCombination, combineVideoButton)
        self.Bind(wx.EVT_BUTTON, self.TranscribeSteps, self.transcibeStepButton)
        self.Bind(
            wx.EVT_BUTTON, self.OnRerenderStepAudios, self.rerenderStepAudioButton
        )
        self.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggleLog, showLogButton)
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

    def OnSavePPTX(self, event):
        savePath = self.CorePath / f"{self.saveFileTBox.Text}.pptx"
        wx.LogMessage(f"Saving generated slides to {savePath}")
        self.presMaker.Save(savePath)

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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.VideoPath is None:
            raise ValueError(
                "No core path is selected! Please select a path using the browser..."
            )
        wx.LogMessage("Begin video combination/renaming.")

        wx.LogMessage(f"Reading video form: {self.VideoPath}")
        # evt.GetEventObject().Disable()
        # self.Layout()
        # self.Update()

        def worker():
            try:
                wx.CallAfter(wx.LogMessage, "Reading videos from ...")
                read_and_combine_videos(
                    self.VideoPath,
                    self.AudioPath,
                    self.SegPath / "FirstFrame.jpg",
                )
                wx.CallAfter(wx.LogMessage, f"Moved videos into {self.VideoPath}.")

            except Exception:
                wx.CallAfter(wx.LogMessage, f"Error: {traceback.format_exc()}")
                wx.CallAfter(evt.GetEventObject().Enable)
                # wx.CallAfter(self.transcibeStepButton.Enable)

            # wx.CallAfter(evt.GetEventObject().Enable)
            # wx.CallAfter(self.transcibeStepButton.Enable)

        Thread(target=worker, daemon=True).start()
        evt.GetEventObject().Enable()
        self.Layout()
        # self.Update()

    def TranscribeSteps(self, evt):
        if self.AudioPath is None:
            raise ValueError(
                "No core path is selected! Please select a path using the browser..."
            )

        # self.SetStatusText("Transcription sequence initiated.")
        # evt.GetEventObject().Disable()
        # self.Layout()
        # self.Update()
        # self.SetStatusText("Initiating thread...")  # Untoggle the button

        def worker():
            try:
                wx.CallAfter(
                    wx.LogMessage,
                    "Loading whisper model for transcribing speech into text...",
                )

                speech_to_text(
                    self.AudioPath / "combined.mp3",
                    self.TextPath / "combined.json",
                    model_name="distil-large-v3",
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
                wx.CallAfter(self.AddSlides)

                wx.CallAfter(
                    wx.LogMessage,
                    "All done, Thank you for your patience!",
                )

            except Exception:
                # wx.CallAfter(wx.LogMessage, f"Error: {(AudioFile+'combined.mp3')}")
                wx.CallAfter(wx.LogMessage, f"Error: {traceback.format_exc()}")
                # wx.CallAfter(evt.GetEventObject().Enable)

            # wx.CallAfter(evt.GetEventObject().Enable)
            # wx.CallAfter(self.Layout)

        Thread(target=worker, daemon=True).start()
        self.Layout()

    def OnRerenderStepAudios(self, evt):
        def worker():
            try:
                for istep in range(self.presMaker.GetPageCount() - 1):
                    print("Rerendering step", istep)
                    wx.CallAfter(
                        self.presMaker.GetPage(istep + 1)
                        .shapes["movie"][0]
                        .movieCtrl.Stop
                    )
                    wx.CallAfter(
                        self.presMaker.GetPage(istep + 1)
                        .shapes["movie"][0]
                        .movieCtrl.Load,
                        "",
                    )
                    newStepTextsChanged = (
                        self.presMaker.GetPage(istep + 1).shapes["textbox"][1].Text
                    )
                    self.RerenderStepAudio(istep, newStepTextsChanged)

                    wx.CallAfter(
                        self.presMaker.GetPage(istep + 1).shapes["movie"][0].LoadVideo,
                        str(self.SegPath / f"AFHeartEdited{istep}.mp4"),
                    )
            except Exception:
                wx.CallAfter(wx.LogMessage, f"Error: {traceback.format_exc()}")

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

        videoClip = (
            VideoFileClip(self.SegPath / f"AFHeart{istep}.mp4")
            .resized(0.5)
            .without_audio()
        )

        audioClip = AudioFileClip(audioClipPath)

        videoClip = videoClip.with_audio(audioClip)

        videoClip.write_videofile(
            self.SegPath / f"AFHeartEdited{istep}.mp4",
            codec="libx264",
            audio_codec="aac",
            preset="veryfast",
            logger=None,
            threads=8,
        )

        audioClip.close()
        videoClip.close()

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
        for stepPath in self.SegPath.glob("Step*.json"):
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

            vidFilePath = self.SegPath / f"TmpOGAud{istep}.mp4"

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
