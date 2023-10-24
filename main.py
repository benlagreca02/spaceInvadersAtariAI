import numpy as np
from utils import ColorBounds, ImageProcessor
import pickle
import os
import skimage as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
import ultralytics

# all the test/training data is in folders like
# 'blue' and 'yellow' for blue and yellow aliens respectively
input_dir = os.path.join(os.getcwd(), 'alien_pics')


def train(epoch_count):
    model = ultralytics.YOLO('yolov8n.yaml')
    results = model.train(data='myYoloConfig.yaml', epochs=epoch_count)


# manual test run, just iterate through a ton of pictures and try to label them
def test(model_path):
    model = ultralytics.YOLO(model_path)
    thresh = 0.3
    test_dir = 'data/test_images'
    test_results_dir = 'data/test_results'
    test_picture_names = os.listdir(test_dir)

    for test_picture_name in test_picture_names:
        print(f"looking at {test_picture_name}")
        pic = cv2.imread(os.path.join(test_dir, test_picture_name))
        results = model(pic)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > thresh:
                cv2.rectangle(pic, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(pic, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imwrite(os.path.join(test_results_dir, test_picture_name), pic)


if __name__ == "__main__":
    # very manual and 'hackish' tbh, but doesn't really matter right now
    train(10)
    # test('runs/detect/rev1epoch5/weights/last.pt')
