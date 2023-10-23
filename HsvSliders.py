# HSV cv2.inRange() slider program taken from cv2 example code
# I made modifications to iterate through a folder of pictures

from __future__ import print_function

import time

import cv2
import cv2 as cv
import argparse

from utils import ColorBounds

max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
k_size_x_name = 'Kernel X Dimension'
k_size_y_name = 'Kernel Y Dimension'
sigma_x_name = 'sigmaX'
k_size_x = 1
k_size_y = 1
sigma_x = 0


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

def on_k_size_x_trackbar(val):
    global k_size_x
    # guarantee the number is odd
    k_size_x = (val + 1) % 2 + val
    cv.setTrackbarPos(k_size_x_name, window_detection_name, k_size_x)

def on_k_size_y_trackbar(val):
    global k_size_y
    # guarantee the number is odd
    k_size_y = (val + 1) % 2 + val
    cv.setTrackbarPos(k_size_y_name, window_detection_name, k_size_y)

def on_sigma_x_trackbar(val):
    global sigma_x
    # guarantee the number is odd
    sigma_x = val
    cv.setTrackbarPos(sigma_x_name, window_detection_name, sigma_x)
# Iterates through some pictures saved in "testCaptures" folder and displays sliders for playing with
# bounds for color masking

def init_windows():
    cv.namedWindow(window_capture_name)
    cv.namedWindow(window_detection_name)
    cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)
    # could change max value if I felt like it
    cv.createTrackbar(k_size_x_name, window_detection_name, k_size_x, max_value, on_k_size_x_trackbar)
    cv.createTrackbar(k_size_y_name, window_detection_name, k_size_y, max_value, on_k_size_y_trackbar)
    cv.createTrackbar(sigma_x_name, window_detection_name, sigma_x, max_value, on_sigma_x_trackbar)



if __name__ == "__main__":

    first_pic_num = 14
    skip_by = 1
    last_pic_num = 750

    folder_name = "data/images"

    init_windows()

    key = None
    while True:

        for i in range(first_pic_num, last_pic_num, skip_by):
            pic = cv.imread(f"{folder_name}/{i}.png")
            if pic is None:
                continue

            print(f"now showing picture {i}")
            # show base pic, only once per 'n'
            smol_pic = cv.resize(pic, (960, 540))
            cv.imshow(window_capture_name, smol_pic)

            frame_hsv = cv.cvtColor(pic, cv.COLOR_BGR2HSV)

            # loop and keep redrawing
            while key is None:
                # create a mask based on slider values
                mask = cv.inRange(frame_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))

                # blur the mask a little, must be odd!
                mask = cv.GaussianBlur(mask, (k_size_x, k_size_y), sigma_x)

                # figure out where rectangles should go
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # ??????
                contours = contours[0] if len(contours) == 2 else contours[1]

                # draw the rectangles on both pictures
                pic_with_boxes = pic.copy()
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # cv2.rectangle(mask, (x,y), (x+w, y+h), (0,0,255), 2)
                    # if the box is too small, it's probably not an alien
                    if w < ColorBounds.MIN_ALIEN_SIZE or h < ColorBounds.MIN_ALIEN_SIZE:
                        continue
                    cv2.rectangle(pic_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)

                smol_thresh = cv.resize(mask, (960, 540))
                cv.imshow(window_detection_name, smol_thresh)

                # redraw original pic, hopefully with rectangles
                smol_pic = cv.resize(pic_with_boxes, (960, 540))
                cv.imshow(window_capture_name, smol_pic)

                key = cv.waitKey(150)
                if key == ord('n'):
                    key = None
                    break
                elif key == ord('q') or key == 27:
                    exit(0)
                else:
                    key = None
