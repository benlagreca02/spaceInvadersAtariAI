# Python script to read the screen and return 2d array of pixels
# Will essentially "look at" the screen then feed this data to recognizer AI. (I hope)

from Xlib import display
from PIL import Image
import numpy as np
import cv2
import time


# ASSUMES YOU'RE USING AN EMULATOR THAT USES AN X11 BASED DISPLAY
class ScreenRipper:
    # find the window with given title and store the reference to is
    def __init__(self, expected_window_title):
        disp = display.Display()
        self.root = disp.screen().root

        # get all window id's
        window_ids = self.root.get_full_property(disp.intern_atom('_NET_CLIENT_LIST'), 0).value

        # iterate through window ID's and find the one we want
        window_id_to_capture = None
        for window_id in window_ids:
            self.window = disp.create_resource_object('window', window_id)
            window_title = self.window.get_wm_name()
            if window_title and window_title == expected_window_title:
                window_id_to_capture = window_id
                break
        if self.window is None:
            Exception(f"Couldn't find window with name {expected_window_title}")

    # takes one screenshot and returns it as a np array of BRG pixel values
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
        return np.asarray(Image.frombytes("RGB", (width, height), img_dat, "raw", "RGBX"))


if __name__ == "__main__":
    # could make these command line args, but this should only be run once in a blue moon to collect data to create
    # training data
    WINDOW_TITLE = "simple64    build: f7bab16"
    screenshot_storage_directory = "raw_screenshots"
    first_filenum = 0
    secs_between_screenshots = 2
    frequency_of_debug_prints = 15

    i = first_filenum
    myRipper = ScreenRipper(WINDOW_TITLE)
    print(f"Found window, taking screenshots every {secs_between_screenshots} seconds...")
    while True:
        screen = myRipper.get_screen()
        cv2.imwrite(f"{screenshot_storage_directory}/{i}.png", screen)
        i += 1
        if i%frequency_of_debug_prints == 0:
            print(f"Saved picture {i}...")
        time.sleep(secs_between_screenshots)
