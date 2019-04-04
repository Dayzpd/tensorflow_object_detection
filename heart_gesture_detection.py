import tensorflow as tf
import cv2
import numpy as np
from math import ceil
from sys import exit

PATH_TO_CKPT = 'frozen_inference_graph.pb'
SCORE_THRESHOLD = .76


def get_graph():
    global PATH_TO_CKPT
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


def unpack_boxes(box, shape):
    height, width = shape
    y_min, x_min, y_max, x_max = tuple(box.tolist())
    y_min = ceil(height * y_min)
    y_max = ceil(height * y_max)
    x_min = ceil(width * x_min)
    x_max = ceil(width * x_max)
    return y_min, x_min, y_max, x_max


def resize_heart(frame, heart, boxes):
    y_min, x_min, y_max, x_max = unpack_boxes(boxes, frame.shape[:2])
    heart_size = (x_max - x_min, y_max - y_min)
    heart_draw = cv2.resize(heart, heart_size)
    alpha_heart = heart_draw[:, :, 2] / 255.0
    alpha_frame = 1.0 - alpha_heart

    for chan in range(0, 3):
        try:
            frame[y_min:y_max, x_min:x_max, chan] = (
                alpha_heart * heart_draw[:, :, chan] + alpha_frame *
                frame[y_min:y_max, x_min:x_max, chan])
        except ValueError:
            pass

    return frame


def get_available_cameras():
    camera_index = 0
    available_cameras = []

    print('Please wait while the program searches for available cameras...')
    print('프로그램이 카메라들을 찾아보고 있으면서 기다려주세요...')
    while True:
        capture_test = cv2.VideoCapture(camera_index)
        if capture_test.read()[0]:
            available_cameras.append(camera_index)
            capture_test.release()
            print('Found camera with ID: [' + str(camera_index) + '].')
            print('카메라 ID: [' + str(camera_index) + '] 찾았습니다.')
            camera_index += 1
        else:
            break

    return available_cameras


def menu():
    options = get_available_cameras()
    if len(options) == 0:
        exit('No cameras were found (카메라가 없읍니다).')

    print('####################### Main Menu (EN) #######################')
    print('Cameras found: [' + ', '.join(str(x) for x in options) + '].')
    print('Instructions: Enter the INTEGER to select your desired camera.')
    print('Example: Type [' + str(options[0]) + '] and press [Enter].')
    print('##############################################################\n')

    print('Change Locale Settings to read Korean menu below.\n')

    print('####################### Main Menu (KR) #######################')
    print('찾았는 카메라들: [' + ', '.join(str(x) for x in options) + '].')
    print('지침: 카메라를 선택하는 것을 정수를 입력합니다.')
    print('예를 들면요: [' + str(options[0]) + '] 입력하고 [Enter] 누릅니다.')
    print('##############################################################\n')

    selection = int(input('Camera selection (선택하는 카메라): '))

    return selection


def run(selection):
    global SCORE_THRESHOLD
    video_capture = cv2.VideoCapture(selection)
    heart = cv2.imread('heart.png')

    detection_graph = get_graph()

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                ret, frame = video_capture.read()

                frame_expanded = np.expand_dims(frame, axis=0)
                image_tensor = detection_graph.get_tensor_by_name(
                    'image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                classes = detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')

                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})

                scores = np.squeeze(scores)
                boxes = np.squeeze(boxes)

                for x in range(boxes.shape[0]):
                    try:
                        score = scores[x]
                    except (IndexError, NameError):
                        score = None
                    if score and score > SCORE_THRESHOLD:
                        frame = resize_heart(frame, heart, boxes[x])

                cv2.imshow('Heart Gesture Detection (Press "q" to close.)',
                           frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break


if __name__ == "__main__":
    selection = menu()

    run(selection)
