import cv2
import argparse
import time
import sys
import json
import numpy as np
import os

# models
FACE_MODEL = "opencv_face_detector_uint8.pb"
FACE_PROTO = "opencv_face_detector.pbtxt"
AGE_MODEL = "age_net.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"

def get_camera_or_argument(camera_arg):
        if camera_arg is not None:
            return int(camera_arg)
        else:
            for camera_id in range(10):
                video = cv2.VideoCapture(camera_id)
                if video.isOpened():
                    video.release()
                    return camera_id
            return None
class GenderDetector:
    def __init__(self):
        self.net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        self.gender_list = ['Male', 'Female']

    def predict_gender(self, face_blob):
        self.net.setInput(face_blob)
        gender_preds = self.net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]
        return gender


class AgeDetector:
    def __init__(self, age_ranges):
        self.net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        self.age_ranges = age_ranges

    def predict_age(self, face_blob):
        self.net.setInput(face_blob)
        age_preds = self.net.forward()
        age_index = age_preds[0].argmax()

        if age_index < len(self.age_ranges):
            age_range = self.age_ranges[age_index]
        else:
            age_range = "Unknown"

        return age_range


class FaceDetector:
    def __init__(self, mean_values):
        self.net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
        self.mean_values = mean_values

    def highlight_face(self, frame, conf_threshold=0.7):
        frame_opencv_dnn = frame.copy()
        frame_height, frame_width, _ = frame_opencv_dnn.shape
        blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), self.mean_values, True, False)

        self.net.setInput(blob)
        detections = self.net.forward()
        face_boxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1, y1, x2, y2 = int(detections[0, 0, i, 3] * frame_width), int(
                    detections[0, 0, i, 4] * frame_height), int(detections[0, 0, i, 5] * frame_width), int(
                    detections[0, 0, i, 6] * frame_height)
                face_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)

        return frame_opencv_dnn, face_boxes


class AgeGenderDetector:
    def __init__(self, camera_arg=None, image_arg=None, ip_arg=None):
        # self.camera_id = get_camera_or_argument(camera_arg)
        # if self.camera_id is None and self.image_arg is None and ip_arg is None:
        #     print("no source found!")
        #     sys.exit()
        # elif image_arg is not None:
        #     self.camera_id = image_arg
        # elif ip_arg is not None:
        #     self.camera_id = ip_arg
            
        age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-70)', '(71-80)', '(81-90)', '(91-100)']
        self.gender_detector = GenderDetector()
        self.age_detector = AgeDetector(age_ranges)
        self.face_highlighter = FaceDetector((78.4263377603, 87.7689143744, 114.895847746))

    def detect_age_gender(self, frame, face_boxes, padding=20):
        results = []
        for face_box in face_boxes:
            face = self.extract_face(frame, face_box, padding)
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                              swapRB=False)

            gender = self.gender_detector.predict_gender(face_blob)
            age = self.age_detector.predict_age(face_blob)

            results.append({
                'gender': gender,
                'age': age
            })

        return results

    def extract_face(self, frame, face_box, padding):
        x1 = max(0, face_box[1] - padding)
        y1 = max(0, face_box[0] - padding)
        x2 = min(face_box[3] + padding, frame.shape[0] - 1)
        y2 = min(face_box[2] + padding, frame.shape[1] - 1)

        return frame[x1:x2, y1:y2]

    # def display_results(self, frame, face_box, gender, age):
    #     color = (0, 255, 0)  # green color
    #     line_thickness = 2
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     font_scale = 0.7
    #     font_color = (255, 255, 255)  # white color

    #     cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), color, line_thickness)
    #     cv2.putText(frame, f'Gender: {gender}, Age: {age} years', (face_box[0], face_box[1] - 10), font, font_scale,
    #                 font_color, line_thickness)

    def run_on_images(self, images):
        padding = 20
        results = []

        for frame in images:
            if frame is None:
                print("Error: Unable to read an image.")
                continue

            start_time = time.time()

            result_img, face_boxes = self.face_highlighter.highlight_face(frame)
            if face_boxes is None:
                print("No face detected in the image.")
                continue

            # append results to the list
            results.extend(self.detect_age_gender(frame, face_boxes, padding))

            end_time = time.time()
            total_time = end_time - start_time
            print(f"The process took {total_time * 1000} ms")

        return {'results': results}

def handle_no_frame(self):
        while True:
            key = cv2.waitKey(100)
            if key > 0:
                cv2.destroyAllWindows()
                break
            window_status = cv2.getWindowProperty("Gender/Age Recognition", cv2.WND_PROP_VISIBLE)
            if window_status is None or window_status < 0:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', nargs='+')
    parser.add_argument('--camera')
    parser.add_argument('--ip')

    args = parser.parse_args()

    if args.image is not None:
        # Check if the provided path is a directory
        if os.path.isdir(args.image[0]):
            # If it's a directory, get all files with a supported image extension
            image_files = [os.path.join(args.image[0], f) for f in os.listdir(args.image[0]) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        else:
            # If it's not a directory, assume it's a list of image files
            image_files = args.image

        # Use absolute paths for the images and filter out non-existent files
        images = [cv2.imread(image_path) for image_path in image_files if os.path.isfile(image_path)]
        if not images:
            print("No valid images found.")
            sys.exit()
    else:
        images = []

    age_gender_detector = AgeGenderDetector(args.camera, None, args.ip)
    age_gender_detector.run_on_images(images)
    
    result_json = age_gender_detector.run_on_images(images)

    print(json.dumps(result_json, indent=2))