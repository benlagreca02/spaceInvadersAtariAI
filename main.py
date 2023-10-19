import time

from utils import ColorBounds, ImageProcessor
from utils.ScreenRipper import ScreenRipper
import cv2


def small_show_and_wait(window_name, img):
    smol_bean = cv2.resize(img, (960, 540))
    cv2.imshow(window_name, smol_bean)
    cv2.waitKey(0)


# right now, just boxes around some blue aliens with a blue rectangle and shows the steps
if __name__ == '__main__':
    img = cv2.imread("testCaptures/9.png")
    small_show_and_wait("pic", img)
    img_boxes = ImageProcessor.draw_boxes_around_alien_color(ColorBounds.BLUE_ALIEN, img, [0xFF, 0xCC, 0x00])
    small_show_and_wait("boxes", img_boxes)

