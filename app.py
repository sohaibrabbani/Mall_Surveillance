import threading
import time

import cv2
from flask import Flask, Response, render_template

from utils import get_video_type, get_dims

app = Flask(__name__)

# global sources for video feeds
web_cam = cv2.VideoCapture(0)  # web-cam feed
# mob_cam2 = cv2.VideoCapture()  # mobile cam 1
# mob_cam3 = cv2.VideoCapture()  # mobile cam 2


def connect_cam(cam, source):
    '''
    Keeps trying to open the video feed on the given IP address when the feed opens.
    :return:
    '''
    while not cam.isOpened():
        cam.open(source)


# Home page for displaying the camera feeds
@app.route('/')
def hello_world():
    return render_template("index.html")


def stream_video(source, filename, res):
    dimensions = get_dims(source, res)
    out = cv2.VideoWriter('data/'+filename, get_video_type(filename), 12, dimensions)

    while True:
        frame = cv2.resize(source.read()[1], dimensions)

        out.write(frame)

        flag, encoded_image = cv2.imencode(".jpg", frame)

        if not flag:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_image) + b'\r\n')


# Streaming response endpoint for camera 1
@app.route("/cam1")
def cam1():
    global web_cam
    return Response(stream_video(web_cam, 'web_cam.avi', 'custom'),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# Streaming response endpoint for camera 3
# @app.route("/cam2")
# def cam2():
#     global mob_cam2
#     return Response(stream_video(mob_cam2, 'mob_cam2.avi', 'custom'),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")
#
#
# # Streaming response endpoint for camera 3
# @app.route("/cam3")
# def cam3():
#     global mob_cam3
#     return Response(stream_video(mob_cam3, 'mob_cam3.avi', 'custom'),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")


# # Starting threads for trying to connect to the IP cams
# cam1_t = threading.Thread(target=connect_cam, args=(mob_cam2, 'http://172.16.3.188:8080/video'))  # Thread for camera 2
# cam1_t.daemon = True
# cam1_t.start()
#
# cam2_t = threading.Thread(target=connect_cam, args=(mob_cam3, 'http://172.16.6.67:8080/video'))  # Thread for camera 3
# cam2_t.daemon = True
# cam2_t.start()


if __name__ == '__main__':
    app.run()
