import os
import threading
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from flask import Flask, Response, render_template
from flask import request, redirect, flash, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

from heatmap import heatmap
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
from utils import get_video_type, get_dims, STD_DIMENSIONS, plot_object
from yolo.yolo import YOLO

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/surveillance.db'
db = SQLAlchemy(app)

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
today = datetime.now()
suffix = today.strftime('%m_%d_%Y_%H')

H1 = np.array([
    [-0.4267572874811169, 0.12202433258793428, 679.1648279868555],
    [0.07855525944118393, 0.5634066050345926, 156.2850020670053],
    [-0.0007529157598914953, 0.002160243169386294, 1.0]
])

HL = np.array([[-0.3742945545238246, -1.091132842267712, 985.1140031814821],
               [1.2741875504115496, -0.038566613915210976, -60.77625504291608],
               [0.0006375440373105805, 0.00048385677889164683, 1.0]])

HR = np.array([[0.3017907086218721, -1.2416167716780655, 1021.5367338709084],
               [0.8382672585685141, 0.03898954281256336, 114.78088045652038],
               [-8.366779008655821e-05, -0.00014174378491588216, 1.0]])

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


lock1 = threading.Lock()
stream_output1 = [np.zeros((100, 100, 3)), []]

lock2 = threading.Lock()
stream_output2 = [np.zeros((100, 100, 3)), []]

lock3 = threading.Lock()
stream_output3 = [np.zeros((100, 100, 3)), []]

# global sources for video feeds
web_cam1 = cv2.VideoCapture()  # web-cam feed
mob_cam2 = cv2.VideoCapture()  # mobile cam 1
mob_cam3 = cv2.VideoCapture()  # mobile cam 2


FPS = 2


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


class TopPoint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    x = db.Column(db.Integer)
    y = db.Column(db.Integer)
    source1 = db.Column(db.String(200))
    source2 = db.Column(db.String(200))
    source3 = db.Column(db.String(200))
    frame_no = db.Column(db.Integer)

    def __repr__(self):
        return f'<TopPoint x: {self.x}, y: {self.y}>'


def write_to_db(detections, filename, frame_no):
    for detection in detections:
        detection['frame_no'] = frame_no
        detection['source'] = filename
        object = Object(**detection)
        db.session.add(object)
    db.session.commit()


def write_to_db_top(points, frame_no, source1=f'cam1_{suffix}.avi', source2=f'cam2_{suffix}.avi', source3=f'cam3_{suffix}.avi'):
    for point in points:
        top_point = TopPoint(x=point[0], y=point[1], frame_no=frame_no,
                             source1=source1, source2=source2, source3=source3)
        db.session.add(top_point)
    db.session.commit()


# Home page for displaying the camera feeds
@app.route('/')
def index():
    return render_template("index.html")


# Home page for displaying the camera feeds
@app.route('/live')
def live():
    return render_template("live.html")


offline_points = [[], ]


def stream_video_homography(file_path1, file_path2, file_path3, res):
    vid = cv2.VideoCapture(file_path1)
    vid2 = cv2.VideoCapture(file_path2)
    vid3 = cv2.VideoCapture(file_path3)
    yolo = YOLO()
    frame_no = 0
    frame_points = []
    while vid.isOpened() or vid2.isOpened() or vid3.isOpened():
        points = []
        frame_no += 1
        grabbed1, frame1 = vid.read()
        if grabbed1:
            frame1 = cv2.resize(frame1, STD_DIMENSIONS["720p"])
            detections = yolo.predict(frame1, model_par=model_par, valid_transform=valid_transform, attribute_detect=True)
            points = plot_object(H1, detections)
            write_to_db(detections, os.path.basename(file_path1), frame_no)
            frame1_warp = cv2.warpPerspective(frame1, H1, STD_DIMENSIONS["720p"])

        grabbed2, frame2 = vid2.read()
        if grabbed2:
            frame2 = np.rot90(frame2, 2)
            frame2 = cv2.resize(frame2, STD_DIMENSIONS["720p"])
            detections = yolo.predict(frame2, model_par=model_par, valid_transform=valid_transform, attribute_detect=True)
            points += plot_object(H2, detections)
            write_to_db(detections, os.path.basename(file_path2), frame_no)
            frame2_warp = cv2.warpPerspective(frame2, H2, STD_DIMENSIONS["720p"])

        grabbed3, frame3 = vid3.read()
        if grabbed3:
            frame3 = cv2.resize(frame3, STD_DIMENSIONS["720p"])
            detections = yolo.predict(frame3, model_par=model_par, valid_transform=valid_transform, attribute_detect=True)
            points += plot_object(H3, detections)
            write_to_db(detections, os.path.basename(file_path3), frame_no)
            frame3_warp = cv2.warpPerspective(frame3, H3, STD_DIMENSIONS["720p"])

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

        offline_points[0] = points.copy()

        write_to_db_top(points, frame_no, source1=file_path1, source2=file_path2, source3=file_path3)
        # frame_points.append(points)
        # some = sum(frame_points, [])
        # final = heatmap(some, top_view)

        flag, encoded_image = cv2.imencode(".jpg", (final * 255).astype(np.float32))

        if not flag:
            continue

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'


@app.route('/video_homo')
def video_homo():
    filename1 = request.args.get('filename1', '')
    filename2 = request.args.get('filename2', '')
    filename3 = request.args.get('filename3', '')
    basepath = os.path.dirname(__file__)
    file_path1 = os.path.join(basepath, 'static', secure_filename(filename1))
    file_path2 = os.path.join(basepath, 'static', secure_filename(filename2))
    file_path3 = os.path.join(basepath, 'static', secure_filename(filename3))
    return Response(stream_video_homography(file_path1, file_path2, file_path3, '720p'),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def stream_video_top():
    top_view = cv2.imread('data/top_view_720p.jpg')
    while True:
        final = top_view.copy()
        for point in offline_points[0]:
            cv2.circle(final, (point[0], point[1]), 6, (0, 0, 255), -1)

        flag, encoded_image = cv2.imencode(".jpg", final)
        if not flag:
            continue
        try:
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'
        except:
            pass

@app.route('/offline_top')
def offline_top():
    return Response(stream_video_top(), mimetype="multipart/x-mixed-replace; boundary=frame")


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


def detect_objects(source, address, filename, res, yolo, stream_output, lock, H):
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
            if filename:
                out.write(frame)

            detections = yolo.predict(frame, model_par=model_par, valid_transform=valid_transform, attribute_detect=True)

            with lock:
                stream_output[0] = frame.copy() if isinstance(frame, np.ndarray) else frame
                stream_output[1] = plot_object(H, detections)

            write_to_db(detections, filename, frame_no)

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


# Streaming response endpoint for camera 1
def stream_top_live():
    dimensions = STD_DIMENSIONS["720p"]
    while True:
        points = []
        if stream_output1[0] is not None:
            frame1 = stream_output1[0]
            frame1 = cv2.resize(frame1, dimensions)
            points += stream_output1[1]
            frame1_warp = cv2.warpPerspective(frame1, H1, dimensions)

        if stream_output2[0] is not None:
            frame2 = stream_output2[0]
            frame2 = cv2.resize(frame2, dimensions)
            points += stream_output2[1]
            frame2_warp = cv2.warpPerspective(frame2, H2, dimensions)

        if stream_output3[0] is not None:
            frame3 = stream_output3[0]
            frame3 = cv2.resize(frame3, dimensions)
            points += stream_output3[1]
            frame3_warp = cv2.warpPerspective(frame3, H3, dimensions)

        try:
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

            final = (final * 255).astype(np.float32)
            flag, encoded_image = cv2.imencode(".jpg", final)
        except:
            pass
        if not flag:
            continue

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'


@app.route("/top_live")
def top_live():
    return Response(stream_top_live(), mimetype="multipart/x-mixed-replace; boundary=frame")


def stream_top_static():
    top_view = cv2.imread('data/top_view_720p.jpg')
    frame_no = 0
    while True:
        frame_no += 1
        final = top_view.copy()
        points = stream_output1[1] + stream_output2[1] + stream_output3[1]
        write_to_db_top(points, frame_no)
        for point in points:
            cv2.circle(final, (point[0], point[1]), 5, (0, 0, 255), -1)

        flag, encoded_image = cv2.imencode(".jpg", final)
        if not flag:
            continue

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'


@app.route("/top_live_static")
def top_live_static():
    return Response(stream_top_static(), mimetype="multipart/x-mixed-replace; boundary=frame")



today = datetime.now()
suffix = today.strftime('%m_%d_%Y_%H')


cam1_t = threading.Thread(target=detect_objects, args=(web_cam1, 'http://192.168.137.67:8080/video',
                                                       f'cam1_{suffix}.avi', 'custom',
                                                       YOLO(), stream_output1, lock1, H1))  # Thread for camera 2
cam1_t.daemon = True
cam1_t.start()

#
# cam2_t = threading.Thread(target=detect_objects, args=(mob_cam2,
#                                                        'http://192.168.100.11:8080/video',
#                                                        f'cam2_{suffix}.avi', 'custom',
#                                                        YOLO(), stream_output2, lock2, H2))  # Thread for camera 3
# cam2_t.daemon = True
# cam2_t.start()
#
#
# cam3_t = threading.Thread(target=detect_objects, args=(mob_cam3,
#                                                        'http://192.168.100.11:8080/video',
#                                                        f'cam3_{suffix}.avi', 'custom',
#                                                        YOLO(), stream_output3, lock3, H3))  # Thread for camera 2
# cam3_t.daemon = True
# cam3_t.start()


if __name__ == '__main__':
    app.run(debug=True)
