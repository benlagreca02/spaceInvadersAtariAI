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


def old_test():
    # actually do some AI finally
    # will iterate through keys, which are just the strings

    data = []
    labels = []

    # if trying to be 'pure' about your vocab, alien color is 'class'
    for alien_color_idx, alien_color in enumerate(ColorBounds.ALIENS):
        for picture in os.listdir(os.path.join(input_dir, alien_color)):
            img_path = os.path.join(input_dir, alien_color, picture)
            # the tutorial uses this instead of cv2... shouldn't matter
            img = sk.io.imread(img_path)
            # tutorial resizes to 15x15
            img = sk.transform.resize(img, (15, 15))
            data.append(img.flatten())
            labels.append(alien_color_idx)

    data = np.asarray(data)
    labels = np.asarray(labels)

    # training and test split
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True,
                                                        stratify=labels)

    # actually do the training
    alien_classifier = SVC()
    params = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

    # trains a few different classifiers using combos of gamma and C values above
    grid_search = GridSearchCV(alien_classifier, params)

    # actually train it!
    grid_search.fit(x_train, y_train)

    # test
    best_estimator = grid_search.best_estimator_
    y_prediction = best_estimator.predict(x_test)

    score = accuracy_score(y_prediction, y_test)

    print(f'{score * 100}% accuracy')

    pickle.dump(best_estimator, open('./model.p', 'wb'))


def train():
    model = ultralytics.YOLO('yolov8n.yaml')
    results = model.train(data='myYoloConfig.yaml', epochs=20)


# manual test run, just iterate through a ton of pictures
def test():
    model_path = './runs/detect/train2/weights/last.pt'
    model = ultralytics.YOLO(model_path)
    thresh = 0.5
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
    test()
