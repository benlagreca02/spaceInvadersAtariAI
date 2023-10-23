from utils import ColorBounds, ImageProcessor
import os
import cv2
import skimage as sk


# collect a metric h*ck ton of pictures of the different aliens
# given a folder with a lot of gameplay screenshots, will parse out each alien picture into its own little folder
def old_main():
    alien_pic_dir = "alien_pics"
    screenshot_dir = "data/images"

    # could auto-pull from the folder, but I don't care, or just make command line args, but I hate cl parsing
    first_sample_pic_num = 0
    last_sample_pic_num = 730
    num_to_skip_by = 1

    # could hypothetically multithread this...
    for curr_pic_num in range(first_sample_pic_num, last_sample_pic_num, num_to_skip_by):
        print(f"Reading image {curr_pic_num}:")
        img = cv2.imread(f"{screenshot_dir}/{curr_pic_num}.png")
        if img is None:
            print(f"couldn't read {curr_pic_num}.png...")
            continue
        # get all the aliens
        for color_string, color_bounds in ColorBounds.ALIENS.items():
            print(f"  Detecting {color_string} aliens:")
            mask = ImageProcessor.make_mask(color_bounds, img)
            # has "too small" filtering baked in
            contours = ImageProcessor.create_contours_over_mask(mask)
            # for each alien detected in the current image
            for contour_num, contour in enumerate(contours):
                # parse out the picture of the alien and save it
                x, y, w, h = cv2.boundingRect(contour)
                # slightly increase 'cropping' range by a smidge
                y -= int(0.3 * h)
                x -= int(0.3 * w)
                h += int(0.3 * h)
                w += int(0.3 * w)
                alien_pic = img[y:y + h, x:x + h]
                cv2.imwrite(f"{alien_pic_dir}/{color_string}/{curr_pic_num}-{contour_num}.png", alien_pic)
                print(f"      found {color_string} alien {contour_num} in current pic")


def get_relative_coordinates(x, y, w, h, file_width, file_y):
    # calculate middle of bounding box
    rel_x = (x + 0.5 * w) / file_width
    rel_y = (y + 0.5 * h) / file_y
    rel_w = w / file_width
    rel_h = h / file_y
    return rel_x, rel_y, rel_w, rel_h


# generate 'labels' for screenshots based on color mask alien detection
# file will be as per yolo standard:
# <label num> <middle of bounding box x> <m.o.b.b. y> <width of bb> <height of bb>
# all cords and size are in percent of file as float
if __name__ == '__main__':

    data_dir = os.path.join(os.getcwd(), 'data/images')
    labels_dir = os.path.join(os.getcwd(), 'data/labels')

    num_files = len(os.listdir(data_dir))
    print(f"{num_files} files to generate test data for...")

    for pic_filename in os.listdir(data_dir):
        print(f"Analyzing {pic_filename}...")
        img_path = os.path.join(data_dir, pic_filename)
        img = sk.io.imread(img_path)

        to_write_to_file = ''

        with open(os.path.join(labels_dir, pic_filename[:-3] + 'txt'), 'a') as label_file:

            # num_color_channels ignored
            file_height, file_width, num_color_channels = img.shape

            # for every alien color
            for alien_idx, alien_color in enumerate(ColorBounds.ALIENS.values()):
                # find the aliens
                mask = ImageProcessor.make_mask(alien_color, img)
                contours = ImageProcessor.create_contours_over_mask(mask)

                # for each alien of the current color
                for contour in contours:
                    # add the data about this alien to the labels folder
                    x, y, w, h = cv2.boundingRect(contour)
                    rx, ry, rw, rh = get_relative_coordinates(x, y, w, h, file_width, file_height)

                    to_write_to_file += f"{alien_idx} {rx} {ry} {rw} {rh}\n"
            label_file.write(to_write_to_file)

    print("Done!")
