import wx
import os
from inputframe import SecondFrame
from socialfunc import detection
from social_distance_detect_v3 import detect
from social_distance_detection_v4_DEEPSORT import socialdistanceDEEP
from yolo import YOLO
import random

#wildcard = "All files (*.*)|*.*"
wildcard = "Video files(*.avi;*.mp4)|*.avi;*.mp4"


class StaticText(wx.StaticText):
    """
    A StaticText that only updates the label if it has changed, to
    help reduce potential flicker since these controls would be
    updated very frequently otherwise.
    """
    def SetLabel(self, label):

        if label != self.GetLabel():
            wx.StaticText.SetLabel(self, label)


class AnalyzeFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(AnalyzeFrame, self).__init__(*args, **kwargs)
        self.SetBackgroundColour((0, 0, 0))
        # panel = wx.Panel(self)

        # title = wx.StaticText(self, label="SOCIAL DISTANCING DETECTOR")
        # font = title.GetFont()
        # font.PointSize += 10
        # font = font.Bold()
        # title.SetFont(font)

        # sizer = wx.BoxSizer(wx.VERTICAL)
        # sizer.Add(title, 0, wx.CENTER, 0)
        # panel.SetSizer(sizer)

        self.filebut = wx.FilePickerCtrl(self, wx.ID_ANY, message='SELECT FILE TO ANALYZE', wildcard=wildcard,
                                      size=(100,30))
        self.filebut.Bind(wx.EVT_BUTTON, self.OnLoadFile)

        analyze_btn = wx.Button(self, -1, "Analyze")
        analyze_btn.Bind(wx.EVT_BUTTON, self.Analyze)

        # result_btn = wx.Button(self, -1, "SEE RESULT")
        # #sizer.Add(btn, 0, wx.ALIGN_CENTER_HORIZONTAL, 0)
        # result_btn.Bind(wx.EVT_BUTTON, self.onClicked)

        ###############
        try:
            self.mc = wx.media.MediaCtrl(self, style=wx.SIMPLE_BORDER)
        except NotImplementedError:
            self.Destroy()
            raise

        self.Bind(wx.media.EVT_MEDIA_LOADED, self.OnMediaLoaded)

        btn1 = wx.Button(self, -1, "Load File")
        self.Bind(wx.EVT_BUTTON, self.OnLoadFile, btn1)

        #play_pic = wx.Bitmap("image/but.png", wx.BITMAP_TYPE_ANY)
        btn2 = wx.Button(self, -1, "Play")
        self.Bind(wx.EVT_BUTTON, self.OnPlay, btn2)
        self.playBtn = btn2

        btn3 = wx.Button(self, -1, "Pause")
        self.Bind(wx.EVT_BUTTON, self.OnPause, btn3)

        btn4 = wx.Button(self, -1, "Stop")
        self.Bind(wx.EVT_BUTTON, self.OnStop, btn4)

        slider = wx.Slider(self, -1, 0, 0, 0)
        self.slider = slider
        slider.SetMinSize((150, -1))
        self.Bind(wx.EVT_SLIDER, self.OnSeek, slider)

        self.st_size = StaticText(self, -1, size=(100, -1))
        self.st_len = StaticText(self, -1, size=(100, -1))
        self.st_pos = StaticText(self, -1, size=(100, -1))

        self.st_size.SetForegroundColour((255, 255, 255))
        self.st_len.SetForegroundColour((255, 255, 255))
        self.st_pos.SetForegroundColour((255, 255, 255))
        # setup the layout
        sizer = wx.GridBagSizer(6, 6)
        sizer.Add(self.mc, (1, 1), span=(5, 1))  # , flag=wx.EXPAND)
        #sizer.Add(result_btn, (1, 3))
        sizer.Add(self.filebut, (1, 2))
        sizer.Add(analyze_btn, (2, 2))
        sizer.Add(btn1, (6, 1),flag=wx.EXPAND)
        sizer.Add(btn2, (6, 3),flag=wx.EXPAND)
        sizer.Add(btn3, (6, 4),flag=wx.EXPAND)
        sizer.Add(btn4, (6, 2),flag=wx.EXPAND)
        sizer.Add(slider, (7, 1), flag=wx.EXPAND)
        sizer.Add(self.st_size, (1, 4))
        sizer.Add(self.st_len, (2, 4))
        sizer.Add(self.st_pos, (3, 4))
        self.SetSizer(sizer)

        wx.CallAfter(self.DoLoadFile, os.path.abspath("/home/kuanhaochen/Documents/social_distance/output/test.mp4"))
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer)
        self.timer.Start(100)
        ##########

        self.CreateStatusBar()
        self.SetStatusText("@powered by Kuan Hao Chen")

    def onClicked(self, event):
        new_frame = SecondFrame(self, title="KUAN HAWKEYE", size=(1000, 500))
        new_frame.Show()

    def Analyze(self, evt):
        input_file = self.filebut.GetPath()
        base = os.path.basename(input_file)
        if os.path.splitext(base)[0] != "":
            output_file = "output/{}.avi".format(os.path.splitext(base)[0])
        else:
            ran = random.randint(0, 1000)
            output_file = "output/live_video{}.avi".format(ran)
        display = 1
        socialdistanceDEEP(YOLO(),input_file, output_file)
        #detect(input_file, output_file, display)

####################################
    def OnLoadFile(self, evt):

        dlg = wx.FileDialog(self, message="Choose a media file",
                            defaultDir=os.getcwd(), defaultFile="",
                            style=wx.FD_OPEN | wx.FD_CHANGE_DIR)

        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.DoLoadFile(path)

        dlg.Destroy()

    def DoLoadFile(self, path):

        self.playBtn.Disable()

        if not self.mc.Load(path):
            wx.MessageBox("Unable to load %s: Unsupported format?" % path,
                          "ERROR",
                          wx.ICON_ERROR | wx.OK)
        else:
            self.mc.SetInitialSize()
            self.GetSizer().Layout()
            self.slider.SetRange(0, self.mc.Length())

    def OnMediaLoaded(self, evt):

        self.playBtn.Enable()

    def OnPlay(self, evt):

        if not self.mc.Play():
            wx.MessageBox("Unable to Play media : Unsupported format?",
                          "ERROR",
                          wx.ICON_ERROR | wx.OK)
        else:
            self.mc.SetInitialSize()
            self.GetSizer().Layout()
            self.slider.SetRange(0, self.mc.Length())

    def OnPause(self, evt):

        self.mc.Pause()

    def OnStop(self, evt):

        self.mc.Stop()

    def OnSeek(self, evt):

        offset = self.slider.GetValue()
        self.mc.Seek(offset)

    def OnTimer(self, evt):

        offset = self.mc.Tell()
        self.slider.SetValue(offset)
        self.st_size.SetLabel('Size: %s' % self.mc.GetBestSize())
        self.st_len.SetLabel('Length: %d seconds' % (self.mc.Length() / 1000))
        self.st_pos.SetLabel('Position: %d' % offset)

    def ShutdownDemo(self):

        self.timer.Stop()
        del self.timer

