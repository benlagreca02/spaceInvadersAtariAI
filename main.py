import time

from utils import ColorBounds, ImageProcessor
from utils.screenRipper import ScreenRipper
import cv2

WINDOW_TITLE = "simple64    build: f7bab16"


def small_show_and_wait(window_name, img):
    smol_bean = cv2.resize(img, (960, 540))
    cv2.imshow(window_name, smol_bean)
    cv2.waitKey(0)


# RUNS INFINITELY, takes a screenshot named "<starting_num>.png" and puts in testCaptures folder
def gather_test_pics(starting_num, secs_between_screenshots):
    myRipper = ScreenRipper(WINDOW_TITLE)
    i = starting_num
    while True:
        screen = myRipper.get_screen()
        cv2.imwrite(f"testCaptures/{i}.png", screen)
        i += 1
        print(f"Saved picture {i}...")
        time.sleep(secs_between_screenshots)


if __name__ == '__main__':
    img = cv2.imread("testCaptures/9.png")
    small_show_and_wait("pic", img)
    img_boxes = ImageProcessor.box_aliens(ColorBounds.BLUE_ALIEN, img, [0xFF, 0xCC, 0x00])
    small_show_and_wait("boxes", img_boxes)

