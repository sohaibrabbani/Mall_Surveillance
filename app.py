import os
import threading

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename

from utils import get_video_type, get_dims, STD_DIMENSIONS
from yolo.yolo import YOLO

app = Flask(__name__)

lock1 = threading.Lock()
stream_output1 = [np.zeros((100,100)),]

lock2 = threading.Lock()
stream_output2 = [np.zeros((100,100)),]

lock3 = threading.Lock()
stream_output3 = [np.zeros((100,100)),]

# global sources for video feeds
web_cam1 = cv2.VideoCapture()  # web-cam feed
mob_cam2 = cv2.VideoCapture()  # mobile cam 1
mob_cam3 = cv2.VideoCapture()  # mobile cam 2


# Home page for displaying the camera feeds
@app.route('/')
def index():
    return render_template("index.html")


# Home page for displaying the camera feeds
@app.route('/live')
def live():
    return render_template("live.html")


def stream_video(file_path, res, save=False):
    source = cv2.VideoCapture(file_path)
    yolo = YOLO()
    dimensions = get_dims(source, res)
    filename = file_path.split('/')[-1]
    if save:
        out = cv2.VideoWriter('data/' + filename, get_video_type(filename), 12, dimensions)

    while source.isOpened():
        try:
            grabbed, frame = source.read()
            frame = cv2.resize(frame, dimensions)
            frame = yolo.predict(frame)
            if save:
                out.write(frame)
        except:
            frame = np.zeros(dimensions)

        flag, encoded_image = cv2.imencode(".jpg", frame)

        if not flag:
            continue

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'


@app.route('/video')
def video():
    filename = request.args.get('filename')
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'static', secure_filename(filename))
    return Response(stream_video(file_path, 'custom'),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# Home page for displaying the camera feeds
@app.route('/offline', methods=['GET', 'POST'])
def offline():
    if request.method == 'GET':
        filename = request.args.get('filename')
        return render_template('offline.html', filename=filename)

    if request.method == 'POST':
        # Get the file from post request
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static', secure_filename(file.filename))
        file.save(file_path)
        return redirect(url_for('offline', filename=file.filename))

    return None


def detect_objects(source, address, filename, res, yolo, stream_output, lock):
    dimensions = get_dims(source, res)
    if filename:
        out = cv2.VideoWriter('data/'+filename, get_video_type(filename), 12, dimensions)

    while True:
        while not source.isOpened():
            source.open(address)

        try:
            grabbed, frame = source.read()
            frame = cv2.resize(frame, dimensions)
            frame = yolo.predict(frame)
            if filename:
                out.write(frame)
        except:
            frame = None

        with lock:
            stream_output[0] = frame.copy() if isinstance(frame, np.ndarray) else frame


def stream_live(output_frame, lock):
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
    return Response(stream_live(stream_output1, lock1),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# Streaming response endpoint for camera 2
@app.route("/cam2")
def cam2():
    return Response(stream_live(stream_output2, lock2),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# Streaming response endpoint for camera 3
@app.route("/cam3")
def cam3():
    return Response(stream_live(stream_output3, lock3),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


cam1_t = threading.Thread(target=detect_objects, args=(web_cam1, 0, 'web_cam1.avi', 'custom',
                                                       YOLO(), stream_output1, lock1))  # Thread for camera 2
cam1_t.daemon = True
cam1_t.start()


cam2_t = threading.Thread(target=detect_objects, args=(mob_cam2, 'http://10.47.27.57:8080/video', 'mob_vid2.avi', 'custom',
                                                       YOLO(), stream_output2, lock2))  # Thread for camera 3
cam2_t.daemon = True
cam2_t.start()


cam3_t = threading.Thread(target=detect_objects, args=(mob_cam3, 'http://192.168.100.7:8080/video', 'mob_vid3.avi', 'custom',
                                                       YOLO(), stream_output3, lock3))  # Thread for camera 2
cam3_t.daemon = True
cam3_t.start()


if __name__ == '__main__':
    app.run(debug=True)
