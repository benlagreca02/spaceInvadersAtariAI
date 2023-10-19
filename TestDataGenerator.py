# collect a metric h*ck ton of pictures of the different aliens
# given a folder with a lot of gameplay screenshots, will parse out each alien picture into its own little folder

from utils import ColorBounds, ImageProcessor
import cv2

if __name__ == "__main__":
    alien_pic_dir = "alien_pics"
    screenshot_dir = "raw_screenshots"

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
                y -= int(0.3*h)
                x -= int(0.3*w)
                h += int(0.3*h)
                w += int(0.3*w)
                alien_pic = img[y:y+h, x:x+h]
                cv2.imwrite(f"{alien_pic_dir}/{color_string}/{curr_pic_num}-{contour_num}.png", alien_pic)
                print(f"      found {color_string} alien {contour_num} in current pic")
