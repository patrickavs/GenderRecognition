#Copyright (c) 2019 Mahesh Sawant Gender-Age-Detection

import cv2
import argparse
import time
import sys
from src.AgeGenderDetecter import AgeGenderDetector

# find the working webcam on the system
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

if __name__ == "__main__":
    try:
        parser=argparse.ArgumentParser()
        parser.add_argument("--image")
        parser.add_argument("--camera")
        parser.add_argument("--ip")
        parser.add_argument("--new_age", action="store_true")
        parser.add_argument("--silent", action="store_true")

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

        model = AgeGenderDetector(args.new_age, args.silent)

        video=cv2.VideoCapture(video_arg)
        padding=20
        while cv2.waitKey(1)<0 :
            hasFrame, frame = video.read()
            startTime = time.time()

            if not hasFrame and args.silent:
                break
            elif not hasFrame:
                # work around the infinite loop issue with the cv2 ui
                while True:
                    key = cv2.waitKey(100)
                    if key > 0:
                        video.release()
                        cv2.destroyAllWindows()
                        break
                    if cv2.getWindowProperty("Video Demo",cv2.WND_PROP_VISIBLE) < 1:
                        break
                break
            
            resultImg, faceBoxes = model.highlight_face(frame)

            if not faceBoxes:
                print("No face detected")
                continue
            
            model_results = model.detect_age_gender(resultImg, faceBoxes, padding)
            endTime = time.time()
            totalTime = (endTime - startTime) * 1000
            print(f"Action took {totalTime}ms")
            print(model_results)
            
            if not args.silent:
                cv2.imshow("Video Demo", resultImg)
    except KeyboardInterrupt:
        print("\nExiting by keyboard interrupt...")
    finally:
        video.release()
        cv2.destroyAllWindows()