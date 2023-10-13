from screenRipper import ScreenRipper
import cv2
WINDOW_TITLE = "simple64    build: f7bab16"

import ImageProcessor

def save_dat_as_png(filename, data):
    data.save(f"{filename}.png", 'PNG')


if __name__ == '__main__':

    # get image from test sample pic
    img = cv2.imread("levelSample.png")
    ImageProcessor.cut_aliens_into_individual_images('test', img)

