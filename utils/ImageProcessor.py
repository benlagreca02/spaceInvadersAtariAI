import numpy as np
import cv2

from utils import ColorBounds
import main

yMin = 275
xMin = 415

# height and width of each box
dx = 135
dy = 92

numX = 8
numY = 5

yMax = yMin + 5 * dy
xMax = xMin + 8 * dx

# note cv2's colors are in BGR
BOX_COLOR = (0xFF, 0xCC, 0x00)

INITIAL_ALIEN_GRID_TL = (xMin, yMin)
INITIAL_ALIEN_GRID_BR = (xMax, xMax)


# debugging function, draws a box around each alien as if it were the start of a level
def box_each_alien_hardcoded_grid(img):
    for y in range(yMin, yMax, dy):
        for x in range(xMin, xMax, dx):
            cv2.rectangle(img, (x, y), (x + dx, y + dy), BOX_COLOR, 1)
    return img


# cuts 1920x1080 screenshot into 40 images of where aliens would be assuming it's the start of a level
def cut_aliens_into_individual_images(basename, screen_shot):
    x_count = 0
    y_count = 0
    for y in range(yMin, yMax, dy):
        for x in range(xMin, xMax, dx):
            cropped = screen_shot[y:y + dy, x:x + dx]
            cv2.imwrite(f"screenshots/{basename}_{x_count}_{y_count}.png", cropped)
            x_count = (x_count + 1) % numX

            # for testing, just do show
        y_count += 1


# stolen from a YouTube video for now, just takes in a color and gives a upper and lower bound limit for color detection
def get_color_limits_for_detection(color_to_detect):
    c = np.uint8([[color_to_detect]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    lowerLimit = hsvC[0][0][0] - 1, 100, 100
    upperLimit = hsvC[0][0][0] + 1, 255, 255
    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)
    return lowerLimit, upperLimit


# draw rectangles around aliens of a given color bound class
# THE PURPOSE OF THIS IS TO COLLECT TRAINING DATA
# will soon modify to return the contours, then crop around them
def box_aliens(alien_color_bounds: ColorBounds, img, rect_color):
    acb = alien_color_bounds
    img_as_hsv = img.copy()
    img_as_hsv = cv2.cvtColor(img_as_hsv, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_as_hsv, acb.get_lower_bounds(), acb.get_upper_bounds())
    main.small_show_and_wait("mask", mask)
    mask = cv2.GaussianBlur(mask, (45, 45), 0)
    main.small_show_and_wait("mask blur", mask)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # yeah, I got no idea
    contours = contours[0] if len(contours) == 2 else contours[1]

    # 2 is index for width, 3 is index for height
    # filter out bounding rectangles smaller than a certain amount
    filter(lambda cont: cv2.boundingRect(cont)[2] > 50 and cv2.boundingRect(cont)[3] > 50, contours)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), rect_color, 2)
    return img
