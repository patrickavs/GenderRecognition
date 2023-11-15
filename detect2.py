import cv2
import argparse
import time
import sys
from deepface import DeepFace
import threading
import queue

DeepFace.build_model("Age")
DeepFace.build_model("Gender")

def get_camera_source(camera_arg):
    if camera_arg is not None:
        return int(camera_arg)
    else:
        for camera_id in range(10):
            video = cv2.VideoCapture(camera_id)
            if video.isOpened():
                video.release()
                return camera_id
        return None

def detect_gender_and_age(face):
    face = cv2.resize(face, (224, 224))
    result = DeepFace.analyze(img_path=face, actions=['age', 'gender'], enforce_detection=False)

    if isinstance(result, list):
        result = result[0]

    age = result['age']
    gender = result['gender']

    return age, gender

def face_detection_worker(frame, detections, result_queue):
    frame_opencv_dnn = highlightFace(frame, detections)
    result_queue.put(frame_opencv_dnn)

def highlightFace(frame, detections, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frameHeight = frame_opencv_dnn.shape[0]
    frameWidth = frame_opencv_dnn.shape[1]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            face = frame[y1:y2, x1:x2]
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

            age, gender = detect_gender_and_age(face)
            info_text = f"Gender: {gender}, Age: {age}"
            cv2.putText(frame_opencv_dnn, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                        cv2.LINE_AA)

    return frame_opencv_dnn

parser = argparse.ArgumentParser()
parser.add_argument('--image')
parser.add_argument('--camera')
parser.add_argument('--ip')

args = parser.parse_args()

video_arg = None

if (args.camera or (args.ip is None and args.image is None)):
    camera_id = get_camera_source(args.camera)
    if camera_id is None:
        print(f"No camera found!")
        sys.exit()
    else:
        video_arg = camera_id
elif (args.ip):
    video_arg = args.ip
else:
    video_arg = args.image

video = cv2.VideoCapture(video_arg)
padding = 20

frame_width = int(video.get(3))
frame_height = int(video.get(4))

# Load pre-trained face detection model
prototxt_path = "opencv_face_detector.pbtxt"
weights_path = "opencv_face_detector_uint8.pb"
net = cv2.dnn.readNet(prototxt_path, weights_path)

while cv2.waitKey(1) < 0:
    start_time = time.time()
    has_frame, frame = video.read()

    if not has_frame:
        while True:
            key = cv2.waitKey(100)
            if key > 0:
                video.release()
                cv2.destroyAllWindows()
                break
            if cv2.getWindowProperty("test", cv2.WND_PROP_VISIBLE) < 1:
                break
        break

    if video.get(cv2.CAP_PROP_POS_FRAMES) % 5 == 0:
        frame_opencv_dnn = frame.copy()
        blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()

        result_queue = queue.Queue()
        face_detection_thread = threading.Thread(target=face_detection_worker, args=(frame, detections, result_queue))
        face_detection_thread.start()
        face_detection_thread.join()

        frame = result_queue.get()
        
        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"Time taken for detections: {elapsed_time * 1000} ms")

    cv2.imshow("test", frame)

video.release()
cv2.destroyAllWindows()