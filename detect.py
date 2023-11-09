#Copyright (c) 2019 Mahesh Sawant Gender-Age-Detection

import cv2
import argparse
import time
import sys
import numpy as np
from deepface import DeepFace
from deepface.commons import functions

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

# detect where the faces are and draw a rectangle around them
# def highlightFace(net, frame, conf_threshold=0.7):
#     frameOpencvDnn=frame.copy()
#     frameHeight=frameOpencvDnn.shape[0]
#     frameWidth=frameOpencvDnn.shape[1]
#     blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

#     net.setInput(blob)
#     detections=net.forward()
#     faceBoxes=[]
#     for i in range(detections.shape[2]):
#         confidence=detections[0,0,i,2]
#         if confidence>conf_threshold:
#             x1=int(detections[0,0,i,3]*frameWidth)
#             y1=int(detections[0,0,i,4]*frameHeight)
#             x2=int(detections[0,0,i,5]*frameWidth)
#             y2=int(detections[0,0,i,6]*frameHeight)
#             faceBoxes.append([x1,y1,x2,y2])
#             cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
#     return frameOpencvDnn,faceBoxes

def highlightFace(frame, size):
    faceBoxes = []
    try:
        face_objs = DeepFace.extract_faces(
            img_path=frame,
            target_size=size,
            detector_backend="opencv",
            enforce_detection=False,
        )
        print("face_obj is" + face_objs)
        for face_obj in face_objs:
            facial_area = face_obj["facial_area"]
            x = facial_area["x"],
            y = facial_area["y"],
            w = facial_area["w"],
            h = facial_area["h"]
            faceBoxes.append([x, y, w, h])
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (67, 67, 67), 1
            )
    except:
        print("ex")
        faceBoxes = []
    print(faceBoxes)
    return faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')
parser.add_argument('--camera')
parser.add_argument('--ip')

args=parser.parse_args()

video_arg = None

if (args.camera or (args.ip is None and args.image is None)):
    camera_id = get_camera_source(args.camera)
    if camera_id is None:
        print(f"no camera found!")
        sys.exit()
    else:
        video_arg = camera_id
elif (args.ip):
    video_arg = args.ip
else:
    video_arg = args.image

model_name = "VGG-Face"
detector_backend = "opencv"
target_size = functions.find_target_size(model_name)
DeepFace.build_model(model_name)
DeepFace.build_model("Age")
DeepFace.build_model("Gender")
DeepFace.find(
    img_path= np.zeros([224, 224, 3]),
    db_path="./db",
    model_name=model_name,
    detector_backend=detector_backend,
    distance_metric="cosine",
    enforce_detection=False
)

# faceProto="opencv_face_detector.pbtxt"
# faceModel="opencv_face_detector_uint8.pb"
# ageProto="age_deploy.prototxt"
# ageModel="age_net.caffemodel"
# genderProto="gender_deploy.prototxt"
# genderModel="gender_net.caffemodel"

# MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList=['Male','Female']

# faceNet=cv2.dnn.readNet(faceModel,faceProto)
# ageNet=cv2.dnn.readNet(ageModel,ageProto)
# genderNet=cv2.dnn.readNet(genderModel,genderProto)

# get the image as frame, NOT RELATED TO VIDEO CAMERAS OR ANYTHING
video=cv2.VideoCapture(video_arg)
padding=20
while cv2.waitKey(1)<0 :
    hasFrame, frame = video.read()
    startTime = time.time()

    if not hasFrame:
        # work around the infinite loop issue with the cv2 ui
        while True:
            key = cv2.waitKey(100)
            if key > 0:
                video.release()
                cv2.destroyAllWindows()
                break
            if cv2.getWindowProperty("test",cv2.WND_PROP_VISIBLE) < 1:
                break
        break

    raw_img = frame.copy()
    #resultImg,faceBoxes=highlightFace(faceNet,frame)
    try:
        face_objs = DeepFace.extract_faces(
            img_path=frame,
            target_size=target_size,
            detector_backend=detector_backend,
            enforce_detection=False
        )
        faces = []
        for face_obj in face_objs:
            facial_area = face_obj["facial_area"]
            x = facial_area["x"]
            y = facial_area["y"]
            w = facial_area["w"]
            h = facial_area["h"]
            if x and y:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 8)
    except:
        faces = []

    # if not faceBoxes:
    #     print("No face detected")

    # for all detected faces put gender and age text close to the face box
    # for faceBox in faceBoxes:
    #     face=frame[max(0,faceBox[1]-padding):
    #                min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
    #                :min(faceBox[2]+padding, frame.shape[1]-1)]
        
    #     # work around the bug where the model might detect invalid faces with dimensions 0,0
    #     if (not face.shape[0] or not face.shape[1]):
    #         continue

    #     blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    #     genderNet.setInput(blob)
    #     genderPreds=genderNet.forward()
    #     gender=genderList[genderPreds[0].argmax()]
        

    #     ageNet.setInput(blob)
    #     agePreds=ageNet.forward()
    #     age=ageList[agePreds[0].argmax()]
        
        
    #     endTime = time.time()
    #     totalTime = endTime - startTime
    #     print(f'Gender: {gender}')
    #     print(f'Age: {age[1:-1]} years')
    #     print(f"The process took {totalTime * 1000} ms")

    #     cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow("test", frame)

video.release()
cv2.destroyAllWindows()