import cv2
import os
import re
import time
import math
import numpy as np
import requests
from models.scrfd import SCRFD
from utils.pose_estimator import PoseEstimator
from multiprocessing import Process, Event

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "weights", "det_10g.onnx")

stop_event = Event()

def process_camera(camera_url, stop_event):
    try:

        ip_match = re.search(r"(\d+\.\d+\.\d+\.\d+)", camera_url) 
        ip_address = ip_match.group(1) if ip_match else None

        model = SCRFD(model_path)
        model.prepare(
            ctx_id=-1,
            det_thresh=0.45,
            input_size=(640, 640)
        )

        capture = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pose_estimator = PoseEstimator(frame_width, frame_height)

        SKIP_FRAMES = 5

        while not stop_event.is_set():

            if not capture.isOpened():
                print(f"[WARN] Failed to open {camera_url}, retrying...")
                time.sleep(5)
                continue

            window_name = f"Camera {camera_url}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 400, 300)

            while not stop_event.is_set():

                for _ in range(SKIP_FRAMES):
                    capture.grab()

                ret, frame = capture.read()
                if not ret:
                    print(f"[INFO] Stream ended: {camera_url}")
                    break

                vis = frame.copy()

                boxes, kpss = model.detect(frame)

                if boxes is not None and len(boxes) > 0:
                    for box, landmarks in zip(boxes, kpss):

                        box = pose_estimator.get_coordinates(box[:4])
                        w = box[2] - box[0]
                        h = box[3] - box[1]
                        if w < 20 or h < 20:
                            continue

                        cx_nose = landmarks[2][0] - pose_estimator.camera_center[0]
                        cy_nose = pose_estimator.camera_center[1] - landmarks[2][1]
                        face_angle = (math.degrees(math.atan2(cy_nose, cx_nose))+360)%360
             
                        P = pose_estimator.estimate_affine_matrix_3d23d(np.hstack([landmarks, np.zeros((5, 1))]))
                        R= pose_estimator.P2sRt(P)
                        yaw= pose_estimator.matrix2angle(R)
                        status = pose_estimator.get_directions(yaw,face_angle)
                        pose_estimator.visualize(vis, box, status)

                cv2.imshow(window_name, vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

            capture.release()
            cv2.destroyWindow(window_name)

    except Exception as e:
        print(f"[ERROR] {camera_url}: {e}")


if __name__ == "__main__":

    SERVER_URL="http://172.14.3.27:9000/rtsp"
    response = requests.get(SERVER_URL, timeout=10)
    response.raise_for_status()

    camera_urls= response.json()["rtsp_urls"][0:2]

    processes = [
        Process(target=process_camera, args=(url, stop_event))
        for url in camera_urls
    ]

    for p in processes:
        p.start()


