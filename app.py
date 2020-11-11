import threading
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, render_template
from imutils.video import FPS

from utils import get_video_type, get_dims
from yolo.yolo import YOLO

app = Flask(__name__)

lock1 = threading.Lock()
stream_output1 = [np.zeros((5,5)),]

lock2 = threading.Lock()
stream_output2 = [np.zeros((5,5)),]

lock3 = threading.Lock()
stream_output3 = [np.zeros((5,5)),]

# global sources for video feeds
web_cam1 = cv2.VideoCapture()  # web-cam feed
mob_cam2 = cv2.VideoCapture()  # mobile cam 1
mob_cam3 = cv2.VideoCapture()  # mobile cam 2


# Home page for displaying the camera feeds
@app.route('/')
def hello_world():
    return render_template("index.html")


def detect_objects(source, address, filename, res, yolo, stream_output, lock):
    dimensions = get_dims(source, res)
    out = cv2.VideoWriter('data/'+filename, get_video_type(filename), 12, dimensions)
    start = datetime.now()
    fps = FPS().start()
    printed = False
    threshold = 5
    count = 0
    while True:
        while not source.isOpened():
            source.open(address)
        try:
            grabbed, frame = source.read()
            # count += 1
            # if count % threshold != 0:
            #     continue
            frame = cv2.resize(frame, dimensions)
            frame = yolo.predict(frame)
            out.write(frame)
        except:
            frame = None

        current = datetime.now()
        fps.update()

        if (current - start).total_seconds() >= 10:
            fps.stop()
            print(fps.fps(), fps.elapsed())
            fps = FPS().start()
            start = datetime.now()
        with lock:
            stream_output[0] = frame.copy() if isinstance(frame, np.ndarray) else frame


def stream_video(output_frame, lock):
    while True:
        with lock:
            if output_frame[0] is None:
                yield None
                continue

            flag, encoded_image = cv2.imencode(".jpg", output_frame[0])

            if not flag:
                continue

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'


# Streaming response endpoint for camera 1
@app.route("/cam1")
def cam1():
    return Response(stream_video(stream_output1, lock1),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# Streaming response endpoint for camera 2
@app.route("/cam2")
def cam2():
    global mob_cam2
    return Response(stream_video(stream_output2, lock2),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# Streaming response endpoint for camera 3
@app.route("/cam3")
def cam3():
    return Response(stream_video(stream_output3, lock3),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# cam1_t = threading.Thread(target=detect_objects, args=(web_cam1, 0, 'web_cam1.avi', 'custom',
#                                                        YOLO(), stream_output1, lock1))  # Thread for camera 2
# cam1_t.daemon = True
# cam1_t.start()

#
# cam2_t = threading.Thread(target=detect_objects, args=(mob_cam2, 'http://10.47.27.57:8080/video', 'mob_vid2.avi', 'custom',
#                                                        YOLO(), stream_output2, lock2))  # Thread for camera 3
# cam2_t.daemon = True
# cam2_t.start()


cam3_t = threading.Thread(target=detect_objects, args=(mob_cam3, 'http://192.168.100.6:8080/video', 'mob_vid3.avi', 'custom',
                                                       YOLO(), stream_output3, lock3))  # Thread for camera 2
cam3_t.daemon = True
cam3_t.start()


if __name__ == '__main__':
    app.run()
