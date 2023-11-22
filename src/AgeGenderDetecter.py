import cv2
import argparse
import time
import sys
import numpy as np
import json
import os
from deepface import DeepFace
from deepface.extendedmodels import Age


class AgeGenderDetector:
    # Constant variables

    __MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    # Possbile value list for models
    __ageList = [
        "(0-2)",
        "(4-6)",
        "(8-12)",
        "(15-20)",
        "(25-32)",
        "(38-43)",
        "(48-53)",
        "(60-100)",
    ]

    __genderList = ["Male", "Female"]

    def __init__(self, use_new_age=False, silent=False):
        model_location = os.path.dirname(os.path.abspath(__file__))
        model_location = os.path.join(model_location, "models")
        faceProto = os.path.join(model_location, "opencv_face_detector.pbtxt")
        faceModel = os.path.join(model_location, "opencv_face_detector_uint8.pb")
        ageProto = os.path.join(model_location, "age_deploy.prototxt")
        ageModel = os.path.join(model_location, "age_net.caffemodel")
        genderProto = os.path.join(model_location, "gender_deploy.prototxt")
        genderModel = os.path.join(model_location, "gender_net.caffemodel")

        self.faceNet = cv2.dnn.readNet(faceModel, faceProto)
        self.genderNet = cv2.dnn.readNet(genderModel, genderProto)

        self.use_new_age = use_new_age
        if self.use_new_age:
            self.newAgeModel = DeepFace.build_model("Age")
        else:
            self.ageNet = cv2.dnn.readNet(ageModel, ageProto)

        self.silent = silent

    def highlight_face(self, frame, conf_threshold=0.7):
        frame_opencv_dnn = frame.copy()
        frame_height = frame_opencv_dnn.shape[0]
        frame_width = frame_opencv_dnn.shape[1]
        blob = cv2.dnn.blobFromImage(
            frame_opencv_dnn, 1.0, (300, 300), self.__MODEL_MEAN_VALUES, True, False
        )

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
                if not self.silent:
                    cv2.rectangle(
                        frame_opencv_dnn,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        int(round(frame_height / 150)),
                        8,
                    )
        return frame_opencv_dnn, face_boxes

    def detect_age_gender(self, frame, face_boxes, padding=20):
        results = []

        for face_box in face_boxes:
            face = frame[
                max(0, face_box[1] - padding) : min(
                    face_box[3] + padding, frame.shape[0] - 1
                ),
                max(0, face_box[0] - padding) : min(
                    face_box[2] + padding, frame.shape[1] - 1
                ),
            ]

            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), self.__MODEL_MEAN_VALUES, swapRB=False
            )

            self.genderNet.setInput(blob)
            gender_preds = self.genderNet.forward()
            gender = self.__genderList[gender_preds[0].argmax()]

            age = None
            if self.use_new_age:
                face = cv2.resize(face, (224, 224))
                face = np.expand_dims(face, axis=0)

                age_predictions = self.newAgeModel.predict(face, verbose=0)[0, :]
                age = int(Age.findApparentAge(age_predictions))
            else:
                self.ageNet.setInput(blob)
                age_preds = self.ageNet.forward()
                age = self.__ageList[age_preds[0].argmax()][1:-1]

            results.append({"gender": gender, "age": age})

            if not self.silent:
                color = (0, 255, 0)
                line_thickness = 2
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.7
                # Calculate the center of the face
                center_x = (face_box[0] + face_box[2]) // 2
                center_y = (face_box[1] + face_box[3]) // 4

                # Calculate the position for displaying text at the center
                text_x = center_x - (len(f"Gender: {gender}, Age: {age} years") * 4)
                text_y = center_y - 10

                # Display age and gender information at the center
                cv2.putText(
                    frame,
                    f"Gender: {gender}, Age: {age} years",
                    (text_x, text_y),
                    font,
                    font_scale,
                    color,
                    line_thickness,
                )
        return results

    def run(self, input_data=None):
        padding = 20
        results = []

        if input_data is None or "images" not in input_data:
            print("No valid input data provided.")
            return

        images_data = input_data["images"]

        for frame in images_data:
            # Ensure the frame has the correct shape
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print("Invalid frame shape. Skipping.")
                continue

            start_time = time.time()

            result_img, face_boxes = self.highlight_face(frame)
            if not face_boxes:
                print("No face detected")
                continue

            frame_results = self.detect_age_gender(result_img, face_boxes, padding)
            results.append(frame_results)
            end_time = time.time()
            total_time = end_time - start_time
            print(f"The process took {total_time * 1000} ms")

            cv2.imshow("result", result_img)
            cv2.waitKey(1)

        # Convert the results to JSON
        json_results = json.dumps(results)
        # Printing all json results
        print(json_results)
        return json_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="src/assets/cropped_images_array.npz")
    parser.add_argument("--new_age", action="store_true")
    args = parser.parse_args()

    age_gender_detector = AgeGenderDetector(args.new_age)

    if args.image:
        if args.image.endswith(".npz"):
            # Load image data from .npz file
            with np.load(args.image) as data:
                image_data = data["images"]
        else:
            print("Invalid image file format. Please provide a .npz file.")
            sys.exit(1)

        age_gender_detector.run(input_data={"images": image_data})
    else:
        print("Please provide an image using the --image argument.")
