# Python script to read the screen and return 2d array of pixels
# Will essentially "look at" the screen then feed this data to recognizer AI. (I hope)

from Xlib import display
from PIL import Image


# ASSUMES YOU'RE USING AN EMULATOR THAT USES AN X11 BASED DISPLAY
class ScreenRipper:
    def __init__(self, expectedWinTitle):
        disp = display.Display()
        self.root = disp.screen().root

        # get all window id's
        window_ids = self.root.get_full_property(disp.intern_atom('_NET_CLIENT_LIST'), 0).value

        # iterate through window ID's and find the one we want
        window_id_to_capture = None
        for window_id in window_ids:
            self.window = disp.create_resource_object('window', window_id)
            window_title = self.window.get_wm_name()
            if window_title and window_title == expectedWinTitle:
                window_id_to_capture = window_id
                break

        if self.window is None:
            Exception("BALLS")

    def get_screen(self):
        # Loop this?
        window_attributes = self.window.get_geometry()
        x = window_attributes.x
        y = window_attributes.y
        width = window_attributes.width
        height = window_attributes.height

        xwd_image = self.window.get_image(x, y, width, height, display.X.ZPixmap, 0xffffffff)
        img_dat = xwd_image.data
        # might not need?
        return Image.frombytes("RGB", (width, height), img_dat, "raw", "BGRX")
