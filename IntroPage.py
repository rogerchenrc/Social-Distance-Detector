import wx
from inputframe import SecondFrame
from analyzeframe import AnalyzeFrame


#wildcard = "All files (*.*)|*.*"
wildcard = "Video files(*.avi;*.mp4)|*.avi;*.mp4"
img_file = "image/logo.png"


class MyFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(MyFrame, self).__init__(*args, **kwargs)

        # create a panel in the frame
        panel = wx.Panel(self)     # required in Window OS

        title = wx.StaticText(panel, label="SOCIAL DISTANCING ANALYZER")
        title.SetForegroundColour(wx.Colour(255,255,255))
        font = title.GetFont()
        font.PointSize += 40
        font = font.Bold()
        title.SetFont(font)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(title, 5, wx.CENTER, 0)
        panel.SetSizer(sizer)

        # play_pic = wx.Bitmap("", wx.BITMAP_TYPE_ANY)
        # btn2 = wx.BitmapButton(self, -1, play_pic, pos=(1,1))
        arrow_pic = wx.Bitmap("image/barrow.jpg", wx.BITMAP_TYPE_ANY)
        arrow_pic.SetSize((3, 3))
        btn = wx.BitmapButton(panel, -1, arrow_pic, pos=(1,1), size=(450,100))
        btn.SetBackgroundColour((0, 0, 0, 0))
        sizer.Add(btn, 0, wx.BOTTOM|wx.ALIGN_CENTER_HORIZONTAL,0)
        btn.Bind(wx.EVT_BUTTON, self.onClicked)

        self.makeMenuBar()

        self.CreateStatusBar()
        self.SetStatusText("@powered by Kuan Hao Chen")

    def onClicked(self, event):
        # self.Hide()

        new_frame = AnalyzeFrame(self, title="KUAN HAWKEYE", size=(1800,1000))
        new_frame.Show()

    def makeMenuBar(self):
        fileMenu = wx.Menu()
        helloItem = fileMenu.Append(-1, "&Hello...\tCtrl-H", "Hello welcome to HAWKEYE")
        fileMenu.AppendSeparator()
        selectItem = fileMenu.Append(-1, "Select File... \tCtrl-U", "Please select a file")
        fileMenu.AppendSeparator()
        exitItem = fileMenu.Append(wx.ID_EXIT)


        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)

        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "&Help")

        self.SetMenuBar(menuBar)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.onEraseBackground)
        self.Bind(wx.EVT_MENU, self.OnHello, helloItem)
        self.Bind(wx.EVT_MENU, self.OnSelect, selectItem)
        self.Bind(wx.EVT_MENU, self.OnExit, exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)

    def OnExit(self, event):
        self.Close(True)

    def OnHello(self, event):
        wx.MessageBox("You are now in the beta version", "Welcome to Version 1.0.0", wx.ICON_INFORMATION|wx.OK)

    def OnAbout(self, event):
        wx.MessageBox("Click the arrow to enter system", "Notification", wx.ICON_INFORMATION|wx.OK)

    def OnSelect(self, event):
        dlg = wx.FileDialog(self, message="Choose a File", defaultFile="",
                            wildcard=wildcard, style=wx.FD_OPEN | wx.FD_MULTIPLE |wx.FD_CHANGE_DIR)
        if dlg.ShowModal()==wx.ID_OK:
            paths = dlg.GetPaths()
            print("You chose the following files:")
            for path in paths:
                print(path)
        dlg.Destroy()

    def onEraseBackground(self, event):
        # Add picture to the background
        dc = event.GetDC()
        if not dc:
            dc = wx.ClientDC(self)
            rect = self.GetUpdateRegion().GetBox()
            dc.SetClippingRect(rect)
        dc.Clear()
        bmp = wx.Bitmap(img_file)
        dc.DrawBitmap(bmp, 0, 0)


if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame(None, title="KUAN HAWKEYE", size=(1800, 1000))
    frame.Show()
    app.MainLoop()

