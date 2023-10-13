import cv2


yMin = 275
xMin = 415

# height and width of each box
dx = 135
dy = 92

numX = 8
numY = 5

yMax = yMin + 5 * dy
xMax = xMin + 8 * dx
# In BGR for some reason...
BOX_COLOR = (0xFF, 0xCC, 0x00)

INITIAL_ALIEN_GRID_TL = (xMin, yMin)
INITIAL_ALIEN_GRID_BR = (xMax, xMax)


# debugging function
def box_each_alien(img):
    # where to start drawing boxes
    # print(f"top left: ({xMin},{yMin}) deltas: ({dx},{dy}) maxes: ({xMax},{yMax})")

    for y in range(yMin, yMax, dy):
        for x in range(xMin, xMax, dx):
            cv2.rectangle(img, (x, y), (x+dx, y+dy), BOX_COLOR, 2)

    cv2.imshow("Boxes", img)
    cv2.waitKey(0)


# cuts 1920x1080 screenshot into 40 images of where aliens would be assuming it's the start of a level
def cut_aliens_into_individual_images(basename, screen_shot):
    xcount = 0
    ycount = 0
    for y in range(yMin, yMax, dy):
        for x in range(xMin, xMax, dx):
            cropped = screen_shot[y:y + dy, x:x + dx]
            cv2.imwrite(f"screenshots/{basename}_{xcount}_{ycount}.png", cropped)
            xcount = (xcount + 1) % numX

            # for testing, just do show
        ycount += 1
