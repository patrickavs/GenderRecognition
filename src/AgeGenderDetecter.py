import cv2
import argparse
import time
import sys



class AgeGenderDetector:
    def __init__(self, camera_arg=None):
        self.camera_id = self.get_camera_source(camera_arg)
        if self.camera_id is None:
            print("No camera found!")
            sys.exit()

        self.faceProto = "opencv_face_detector.pbtxt"
        self.faceModel = "opencv_face_detector_uint8.pb"
        self.ageProto = "age_deploy.prototxt"
        self.ageModel = "age_net.caffemodel"
        self.genderProto = "gender_deploy.prototxt"
        self.genderModel = "gender_net.caffemodel"

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genderList = ['Male', 'Female']

        self.faceNet = cv2.dnn.readNet(self.faceModel, self.faceProto)
        self.ageNet = cv2.dnn.readNet(self.ageModel, self.ageProto)
        self.genderNet = cv2.dnn.readNet(self.genderModel, self.genderProto)

    def get_camera_source(self, camera_arg):
        if camera_arg is not None:
            return int(camera_arg)
        else:
            for camera_id in range(10):
                video = cv2.VideoCapture(camera_id)
                if video.isOpened():
                    video.release()
                    return camera_id
            return None

    def highlight_face(self, frame, conf_threshold=0.7):
        frame_opencv_dnn = frame.copy()
        frame_height = frame_opencv_dnn.shape[0]
        frame_width = frame_opencv_dnn.shape[1]
        blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), self.MODEL_MEAN_VALUES, True, False)

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        face_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                face_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
        return frame_opencv_dnn, face_boxes

    def detect_age_gender(self, frame, face_boxes, padding=20):
        for face_box in face_boxes:
            face = frame[max(0, face_box[1] - padding): min(face_box[3] + padding, frame.shape[0] - 1),
                   max(0, face_box[0] - padding): min(face_box[2] + padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

            self.genderNet.setInput(blob)
            gender_preds = self.genderNet.forward()
            gender = self.genderList[gender_preds[0].argmax()]

            self.ageNet.setInput(blob)
            age_preds = self.ageNet.forward()
            age = self.ageList[age_preds[0].argmax()][1:-1]

            color = (0, 255, 0)  # Green color
            line_thickness = 2
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.7
            font_color = (255, 255, 255)  # White color

            cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), color, line_thickness)
            cv2.putText(frame, f'Gender: {gender}, Age: {age} years', (face_box[0], face_box[1] - 10), font, font_scale,
                        font_color, line_thickness)

    def run(self, image_path=None):
        video = cv2.VideoCapture(image_path if image_path else self.camera_id)
        padding = 20
        while cv2.waitKey(1) < 0:
            has_frame, frame = video.read()
            start_time = time.time()
            if not has_frame:
                while True:
                    key = cv2.waitKey(100)
                    if key > 0:
                        cv2.destroyAllWindows()
                        break
                    if cv2.getWindowProperty("test", cv2.WND_PROP_VISIBLE) < 1:
                        break
                break

            result_img, face_boxes = self.highlight_face(frame)
            if not face_boxes:
                print("No face detected")
                continue

            self.detect_age_gender(frame, face_boxes, padding)

            end_time = time.time()
            total_time = end_time - start_time
            print(f"The process took {total_time * 1000} ms")

            cv2.imshow("test", frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    parser.add_argument('--camera')

    args = parser.parse_args()

    age_gender_detector = AgeGenderDetector(args.camera)
    age_gender_detector.run(args.image)
