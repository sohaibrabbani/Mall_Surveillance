import threading
import time

import cv2
from flask import Flask, Response, render_template

from utils import get_video_type, get_dims


app = Flask(__name__)
time.sleep(2.0)

web_cam = cv2.VideoCapture(0)
mob_cam2 = None
mob_cam3 = None


def connect_cam2():
    global mob_cam2
    mob_cam2 = cv2.VideoCapture()
    while True:
        if not mob_cam2.isOpened():
            mob_cam2.open('http://172.16.3.188:8080/video')


def connect_cam3():
    global mob_cam3
    mob_cam3 = cv2.VideoCapture()
    while True:
        if not mob_cam3.isOpened():
            mob_cam3.open('http://172.16.3.188:8080/video')


@app.route('/')
def hello_world():
    return render_template("index.html")


def stream_video(source, filename, res):
    dimensions = get_dims(source, res)
    out = cv2.VideoWriter(filename, get_video_type(filename), 25, dimensions)

    while True:
        frame = cv2.resize(source.read()[1], dimensions)

        out.write(frame)

        flag, encoded_image = cv2.imencode(".jpg", frame)

        if not flag:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_image) + b'\r\n')


@app.route("/cam1")
def cam1():
    global web_cam
    return Response(stream_video(web_cam, 'web_cam.avi', 'custom'),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/cam2")
def cam2():
    global mob_cam2
    return Response(stream_video(mob_cam2, 'mob_cam2.avi', 'custom'),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/cam3")
def cam3():
    global mob_cam3
    return Response(stream_video(mob_cam3, 'mob_cam3.avi', 'custom'),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


cam1_t = threading.Thread(target=connect_cam2)
cam1_t.daemon = True
cam1_t.start()

cam2_t = threading.Thread(target=connect_cam3)
cam2_t.daemon = True
cam2_t.start()


if __name__ == '__main__':
    app.run()
