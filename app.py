import os
import threading
from datetime import datetime

import cv2
import numpy as np
from flask import Flask
from flask import Response, render_template, request, redirect, flash, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

from heatmap import heatmap
from utils import get_video_type, get_dims, STD_DIMENSIONS
from yolo.yolo import YOLO

from collections import deque, Counter
import cv2
import imutils
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from flask import Flask, Response, render_template
from imutils.video import VideoStream
# from keras import backend

from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
# from tools import generate_detections as gdet


app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/surveillance.db'
db = SQLAlchemy(app)
# yolo = YOLO()


def model_init_par():
    # model
    backbone = resnet50()
    classifier = BaseClassifier(nattr=6)
    model = FeatClassifier(backbone, classifier)

    # load
    checkpoint = torch.load('./exp_result/custom/custom/img_model/ckpt_max.pth')
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dicts'].items()})
    # cuda eval
    model.cuda()
    model.eval()

    # valid_transform
    height, width = 256, 192
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])
    return model, valid_transform


model_par, valid_transform = model_init_par()

lock1 = threading.Lock()
stream_output1 = [np.zeros((100, 100)), ]

lock2 = threading.Lock()
stream_output2 = [np.zeros((100, 100)), ]

lock3 = threading.Lock()
stream_output3 = [np.zeros((100, 100)), ]

# global sources for video feeds
web_cam1 = cv2.VideoCapture()  # web-cam feed
mob_cam2 = cv2.VideoCapture()  # mobile cam 1
mob_cam3 = cv2.VideoCapture()  # mobile cam 2


FPS = 2


H1 = np.array([
    [-0.4267572874811169, 0.12202433258793428, 679.1648279868555],
    [0.07855525944118393, 0.5634066050345926, 156.2850020670053],
    [-0.0007529157598914953, 0.002160243169386294, 1.0]
])

H2 = np.array([
    [2.96918550e-01, 1.69874720e+00, -1.41966972e+02],
    [-3.71748569e-01,  8.56329387e-01, 3.30452853e+02],
    [-1.92530358e-04, 2.88393061e-03, 1.00000000e+00]
])

H3 = np.array([
    [8.87417770e-01, 6.78122356e+00, -1.33642387e+03],
    [-1.10018243e+00, 3.66039041e+00, 7.94978181e+02],
    [3.13098076e-04, 1.01824975e-02, 1.00000000e+00]
])


class Object(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    x = db.Column(db.Integer)
    y = db.Column(db.Integer)
    w = db.Column(db.Integer)
    h = db.Column(db.Integer)
    name = db.Column(db.String(80))
    source = db.Column(db.String(200))
    frame_no = db.Column(db.Integer)

    def __repr__(self):
        return f'<Object x: {self.x}, y: {self.y}, w:{self.w}, h:{self.h}, class: {self.name}>'


def write_to_db(detections, filename, frame_no):
    for detection in detections:
        detection['frame_no'] = frame_no
        detection['source'] = filename
        object = Object(**detection)
        db.session.add(object)
    db.session.commit()


# Home page for displaying the camera feeds
@app.route('/')
def index():
    return render_template("index.html")


# Home page for displaying the camera feeds
@app.route('/live')
def live():
    return render_template("live.html")


# file_path3,
def plot_object(h_phary, detections, frame1):
    points = []
    for d in detections:
        point = np.array([d['x'], d['y'], 1])
        points.append(h_phary.dot(point).astype(int))
    return points



# def demo_par(model, valid_transform, img):
#     # load one image
#     img_trans = valid_transform(img)
#     imgs = torch.unsqueeze(img_trans, dim=0)
#     imgs = imgs.cuda()
#     valid_logits = model(imgs)
#     valid_probs = torch.sigmoid(valid_logits)
#     score = valid_probs.data.cpu().numpy()
#
#     # show the score in the image
#     txt_res = []
#     age_group = []
#     gender = []
#     txt = ""
#     for idx in range(len(values)):
#         if score[0, idx] >= 0.5:
#             temp = '%s: %.2f ' % (values[idx], score[0, idx])
#             if idx < 4:
#                 age_group.append(values[idx])
#             else:
#                 gender.append(values[idx])
#             # txt += temp
#             txt_res.append(temp)
#     return txt_res, age_group, gender


def stream_video(file_path1, file_path2, file_path3, res):


    vid = cv2.VideoCapture(file_path1)
    vid2 = cv2.VideoCapture(file_path2)
    vid3 = cv2.VideoCapture(file_path3)
    dimensions = get_dims(vid, res)
    top_view = cv2.cvtColor(cv2.imread('data/top_view_720p.jpg'), cv2.COLOR_BGR2RGB) / 255
    frame_no = 0
    frame_points = []
    while vid.isOpened() or vid2.isOpened() or vid3.isOpened():
        points = []
        try:
            frame_no += 1
            grabbed1, frame1 = vid.read()
            if grabbed1:
                frame1 = cv2.resize(frame1, dimensions)
                detections = yolo.predict(frame1, model_par=model_par, valid_transform=valid_transform, attribute_detect=True)
                points = plot_object(H1, detections, frame1)
                write_to_db(detections, os.path.basename(file_path1), frame_no)
                frame1_warp = cv2.warpPerspective(frame1, H1, STD_DIMENSIONS["720p"])
            # else:
                # frame1_warp = empty.copy()

            grabbed2, frame2 = vid2.read()
            if grabbed2:
                frame2 = np.rot90(frame2, 2)
                frame2 = cv2.resize(frame2, dimensions)
                detections = yolo.predict(frame2)
                points += plot_object(H2, detections, frame2)
                write_to_db(detections, os.path.basename(file_path2), frame_no)
                frame2_warp = cv2.warpPerspective(frame2, H2, STD_DIMENSIONS["720p"])
            # else:
            #     frame2_warp = empty.copy()

            grabbed3, frame3 = vid3.read()
            if grabbed3:
                frame3 = cv2.resize(frame3, dimensions)
                detections = yolo.predict(frame3)
                points += plot_object(H3, detections, frame3)
                write_to_db(detections, os.path.basename(file_path3), frame_no)
                frame3_warp = cv2.warpPerspective(frame3, H3, STD_DIMENSIONS["720p"])
            # else:
            #     frame3_warp = empty.copy()

            a = frame1_warp / 255
            b = frame2_warp / 255
            c = frame3_warp / 255

            final = np.zeros((a.shape))
            final = np.where(np.logical_and(a != 0, b != 0, c != 0), (a + b + c) / 3, final)
            final = np.where(np.logical_or(np.logical_and(a == 0, np.logical_xor(b == 0, c == 0)),
                                           np.logical_and(c == 0, np.logical_xor(a == 0, b == 0)),
                                           np.logical_and(b == 0, np.logical_xor(a == 0, c == 0))), a + b + c, final)
            final = np.where(np.logical_or(np.logical_and(a != 0, np.logical_xor(b == 0, c == 0)),
                                           np.logical_and(c != 0, np.logical_xor(a == 0, b == 0)),
                                           np.logical_and(b != 0, np.logical_xor(a == 0, c == 0))), (a + b + c) / 2, final)

            # final = top_view

            frame_points.append(points)
            some = sum(frame_points, [])
            final = heatmap(some, top_view)

        except:
            pass

        flag, encoded_image = cv2.imencode(".jpg", cv2.cvtColor((final * 255).astype(np.float32), cv2.COLOR_RGB2BGR))

        if not flag:
            continue

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'


@app.route('/video')
def video():
    filename1 = request.args.get('filename1', '')
    filename2 = request.args.get('filename2', '')
    filename3 = request.args.get('filename3', '')
    basepath = os.path.dirname(__file__)
    file_path1 = os.path.join(basepath, 'static', secure_filename(filename1))
    file_path2 = os.path.join(basepath, 'static', secure_filename(filename2))
    file_path3 = os.path.join(basepath, 'static', secure_filename(filename3))
    return Response(stream_video(file_path1, file_path2, file_path3, '720p'),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# Home page for displaying the camera feeds
@app.route('/offline', methods=['GET', 'POST'])
def offline():
    if request.method == 'GET':
        filename1 = request.args.get('filename1')
        filename2 = request.args.get('filename2')
        filename3 = request.args.get('filename3')
        return render_template('offline.html', filename1=filename1, filename2=filename2, filename3=filename3)

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
        return redirect(url_for('offline', filename1=file.filename, filename2=file.filename2))

    return None


def detect_objects(source, address, filename, res, yolo, stream_output, lock):
    dimensions = get_dims(source, res)
    if filename:
        out = cv2.VideoWriter('data/'+filename, get_video_type(filename), FPS, dimensions)
    frame_no = 0
    while True:
        while not source.isOpened():
            source.open(address)

        try:
            frame_no += 1
            grabbed, frame = source.read()
            frame = cv2.resize(frame, dimensions)
            # original_frame = frame.copy()
            write_to_db(yolo.predict(frame, model_par=model_par, valid_transform=valid_transform, attribute_detect=True), filename, frame_no)
            with lock:
                stream_output[0] = frame.copy() if isinstance(frame, np.ndarray) else frame
            if filename:
                out.write(frame)



        except:
            frame = None

        # with lock:
        #     stream_output[0] = frame.copy() if isinstance(frame, np.ndarray) else frame


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


today = datetime.now()
suffix = today.strftime('%m_%d_%Y_%H')


cam1_t = threading.Thread(target=detect_objects, args=(web_cam1,
                                                       'http://192.168.100.11:8080/video',
                                                       f'cam1_{suffix}.avi', 'custom',
                                                       YOLO(), stream_output1, lock1))  # Thread for camera 2
cam1_t.daemon = True
cam1_t.start()


cam2_t = threading.Thread(target=detect_objects, args=(mob_cam2,
                                                       'http://192.168.100.11:8080/video',
                                                       f'cam2_{suffix}.avi', 'custom',
                                                       YOLO(), stream_output2, lock2))  # Thread for camera 3
cam2_t.daemon = True
cam2_t.start()


cam3_t = threading.Thread(target=detect_objects, args=(mob_cam3,
                                                       'http://192.168.100.11:8080/video',
                                                       f'cam3_{suffix}.avi', 'custom',
                                                       YOLO(), stream_output3, lock3))  # Thread for camera 2
cam3_t.daemon = True
cam3_t.start()


if __name__ == '__main__':
    app.run(debug=True)
