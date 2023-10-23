import numpy as np
import cv2

from utils import ColorBounds


def draw_boxes_around_alien_color(alien_color_bounds: ColorBounds, img, rect_color):
    acb = alien_color_bounds
    mask = make_mask(acb, img)
    contours = create_contours_over_mask(mask)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), rect_color, 2)
    return img


def make_mask(alien_color_bounds: ColorBounds, img):
    acb = alien_color_bounds
    img_as_hsv = img.copy()
    img_as_hsv = cv2.cvtColor(img_as_hsv, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_as_hsv, acb.get_lower_bounds(), acb.get_upper_bounds())
    return cv2.GaussianBlur(mask, acb.get_k_dims(), acb.get_sigma())


def create_contours_over_mask(mask):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # yeah, I got no idea
    contours = contours[0] if len(contours) == 2 else contours[1]

    # 2 is index for width, 3 is index for height
    # filter out bounding rectangles smaller than a certain amount
    filter(lambda c: cv2.boundingRect(c)[2] > ColorBounds.MIN_ALIEN_SIZE and cv2.boundingRect(c)[
        3] > ColorBounds.MIN_ALIEN_SIZE, contours)
    return contours
