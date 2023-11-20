import os
import cv2
import ultralytics
# will need eventually for doing tests on the game as we play it
from utils.ScreenRipper import ScreenRipper


# auto reads from the "data/images" and "data/labels" folder based on the myYoloConfig.yaml
def train(epoch_count):
    model = ultralytics.YOLO('yolov8n.yaml')
    results = model.train(data='myYoloConfig.yaml', epochs=epoch_count)


# manual test run, just iterate through a ton of pictures and try to label them
def test(model_path, confidence_threshold):
    model = ultralytics.YOLO(model_path)
    test_dir = 'data/manual_test_images'
    test_results_dir = 'data/manual_test_results'
    test_picture_names = os.listdir(test_dir)

    for test_picture_name in test_picture_names:
        print(f"looking at {test_picture_name}")
        pic = cv2.imread(os.path.join(test_dir, test_picture_name))
        results = model(pic)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > confidence_threshold:
                cv2.rectangle(pic, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(pic, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imwrite(os.path.join(test_results_dir, test_picture_name), pic)


def test_on_video(model_path, confidence_thresh, video_file):
    # name window for 'watching' the game
    cv2.namedWindow('Game Annotated', cv2.WINDOW_NORMAL)

    video_capture = cv2.VideoCapture(video_file)

    if not video_capture.isOpened():
        exit()

    model = ultralytics.YOLO(model_path)

    label_data_flag = False
    while True:
        ret, curr_frame = video_capture.read()
        if not ret:
            break

        if label_data_flag:
            # DO THE PROCESSING
            # class overrides "call" python dunder method
            # list of length "number of images to run inference on"
            # since we want this 'live' eventually, should be
            # done this way, rather than loading a video
            results = model(curr_frame)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                if score > confidence_thresh:
                    cv2.rectangle(curr_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(curr_frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Game Annotated", curr_frame)

        # very buggy and honestly garbage
        if cv2.waitKey(1) & 0xFF == ord('o'):
            label_data_flag = True

        if cv2.waitKey(1) & 0xFF == ord('f'):
            label_data_flag = False

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # very manual and 'hackish' tbh, but doesn't really matter right now
    # train(80)
    # test('runs/detect/rev2epoch80/weights/last.pt', 0.7)
    test_on_video('runs/detect/rev2epoch80/weights/last.pt', 0.60, 'data/LongplayofSpaceInvaders.mp4')
