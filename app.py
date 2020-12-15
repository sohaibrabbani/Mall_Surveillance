import os
import threading
from datetime import datetime

import json
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
from utils import get_video_type, get_dims, STD_DIMENSIONS, plot_object, draw_on_top, stiching
from yolo.yolo import YOLO
from shapely.geometry.polygon import Polygon


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


app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/surveillance.db'
db = SQLAlchemy(app)
model_par, valid_transform = model_init_par()
FPS = 2
# global sources for video feeds
web_cam1 = cv2.VideoCapture()  # web-cam feed
mob_cam2 = cv2.VideoCapture()  # mobile cam 1
mob_cam3 = cv2.VideoCapture()  # mobile cam 2
today = datetime.now()
suffix = today.strftime('%m_%d_%Y_%H')


HL = np.array([[-0.17426423209765862, -1.031394596052239, 750.412813145867],
               [1.0000328784997377, 0.14136499853820997, -10.370851456391206],
               [0.00026540226307438837, 0.0003478893717957895, 1.0]])

HR = np.array([[0.2930326492542025, -1.251081170395554, 1001.8914835416477],
               [0.804143842312905, 0.04353961962924524, 117.8190775285837],
               [-0.00011482799253370977, -0.0001927218013597335, 0.9999999999999999]])

HC = np.array([[-0.4267572874811169, 0.12202433258793428, 679.1648279868555],
               [0.07855525944118393, 0.5634066050345926, 156.2850020670053],
               [-0.0007529157598914953, 0.002160243169386294, 1.0]])

HR_offline = np.array([[0.11080560537895662, -1.0599380061894015, 965.4661420386825],
               [0.6015272798960488, -0.0192179004621332, 194.81952858686287],
               [-0.00015737848112515156, -0.00020505711257484048, 1.0]])
HC_offline = np.array([[0.11080560537895662, -1.0599380061894015, 965.4661420386825],
               [0.6015272798960488, -0.0192179004621332, 194.81952858686287],
               [-0.00015737848112515156, -0.00020505711257484048, 1.0]])
HL_offline = np.array([[0.5483783109688897, -0.158356441030431, 404.5190885191851],
                       [0.0276893098971346, 0.6240793626918955, -75.88552186278486],
                       [-0.00019537620569174031, 0.0003509974952078044, 0.9999999999999999]])

lock1 = threading.Lock()
stream_output1 = [np.zeros((480, 640, 3)), []]

lock2 = threading.Lock()
stream_output2 = [np.zeros((480, 640, 3)), []]

lock3 = threading.Lock()
stream_output3 = [np.zeros((480, 640, 3)), []]

offline_data = [[], ]

offline_heatmap_points = [[], ]
live_heatmap_points = [[], ]


class Object(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    x = db.Column(db.Integer)
    y = db.Column(db.Integer)
    w = db.Column(db.Integer)
    h = db.Column(db.Integer)
    name = db.Column(db.String(80))
    gender = db.Column(db.String(80))
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


class Perimeter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    x = db.Column(db.Integer)
    y = db.Column(db.Integer)

    def __repr__(self):
        return f'<Vertex x: {self.x}, y: {self.y}>'


top_view = cv2.imread('static/top_view_720p.jpg')
polygon = Polygon([(vertex.x, vertex.y) for vertex in Perimeter.query.all()])
poly_lock = threading.Lock()


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

# def convert_to_homography(frame, yolo, ):
#     frame1 = cv2.resize(frame1, STD_DIMENSIONS["720p"])
#     detections = yolo.predict(frame1, model_par=model_par, valid_transform=valid_transform, attribute_detect=True)
#     points = plot_object(HR, detections)
#     write_to_db(detections, os.path.basename(file_path1), frame_no)
#     frame1_warp = cv2.warpPerspective(frame1, HR, STD_DIMENSIONS["720p"])

def video_homography(file_path1, file_path2, file_path3, res):
    vid = cv2.VideoCapture(file_path1)
    vid2 = cv2.VideoCapture(file_path2)
    vid3 = cv2.VideoCapture(file_path3)
    yolo = YOLO()
    filename1 = os.path.basename(file_path1)
    filename2 = os.path.basename(file_path2)
    filename3 = os.path.basename(file_path3)
    frame_no = 0

    while vid.isOpened() or vid2.isOpened() or vid3.isOpened():
        points = []
        frame_no += 1
        grabbed1, frame1 = vid.read()
        if grabbed1:
            detections = yolo.predict(frame1, model_par=model_par, valid_transform=valid_transform, attribute_detect=True)
            points = plot_object(HR_offline, detections)
            write_to_db(detections, filename1, frame_no)
            frame1_warp = cv2.warpPerspective(frame1, HR_offline, STD_DIMENSIONS["720p"])

        grabbed2, frame2 = vid2.read()
        if grabbed2:
            detections = yolo.predict(frame2, model_par=model_par, valid_transform=valid_transform, attribute_detect=True)
            points += plot_object(HL_offline, detections)
            write_to_db(detections, filename2, frame_no)
            frame2_warp = cv2.warpPerspective(frame2, HL_offline, STD_DIMENSIONS["720p"])

        grabbed3, frame3 = vid3.read()
        if grabbed3:
            detections = yolo.predict(frame3, model_par=model_par, valid_transform=valid_transform, attribute_detect=True)
            points += plot_object(HC_offline, detections)
            write_to_db(detections, filename3, frame_no)
            frame3_warp = cv2.warpPerspective(frame3, HC_offline, STD_DIMENSIONS["720p"])

        offline_data[0] = points.copy()
        offline_heatmap_points[0].append(points)
        write_to_db_top(points, frame_no, source1=filename1, source2=filename2, source3=filename3)

        final = stiching(frame1_warp, frame2_warp, frame3_warp)
        flag, encoded_image = cv2.imencode(".jpg", final)

        if not flag:
            continue

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'


def video_tripwire():
    while True:
        final = draw_on_top(polygon, offline_data[0], top_view.copy())
        flag, encoded_image = cv2.imencode(".jpg", final)

        if not flag:
            continue

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'


def video_heatmap():
    while True:
        final = heatmap(sum(offline_heatmap_points[0][-10:], []), top_view.copy())
        flag, encoded_image = cv2.imencode(".jpg", final)
        if not flag:
            continue

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'


def detect_objects(source, address, filename, res, yolo, stream_output, lock, H):
    dimensions = get_dims(source, res)
    frame_no = 0

    if filename:
        out = cv2.VideoWriter('data/'+filename, get_video_type(filename), FPS, dimensions)

    while True:
        while not source.isOpened():
            source.open(address)

        grabbed, frame = source.read()
        if not grabbed:
            continue

        frame_no += 1

        if filename:
            out.write(frame)

        detections = yolo.predict(frame, model_par=model_par, valid_transform=valid_transform, attribute_detect=True)

        with lock:
            stream_output[0] = frame.copy()
            stream_output[1] = plot_object(H, detections)

        write_to_db(detections, filename, frame_no)


def stream_live(output_frame, lock):
    while True:
        with lock:
            flag, encoded_image = cv2.imencode(".jpg", output_frame[0])

            if not flag:
                continue

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'


# Streaming response endpoint for camera 1
def stream_homo():
    dimensions = STD_DIMENSIONS["720p"]
    while True:
        frame1 = stream_output1[0]
        frame2 = stream_output2[0]
        frame3 = stream_output3[0]

        frame1_warp = cv2.warpPerspective(frame1, HR, dimensions)
        frame2_warp = cv2.warpPerspective(frame2, HL, dimensions)
        frame3_warp = cv2.warpPerspective(frame3, HC, dimensions)

        final = stiching(frame1_warp, frame2_warp, frame3_warp)

        flag, encoded_image = cv2.imencode(".jpg", final)

        if not flag:
            continue

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'


def stream_tripwire():
    frame_no = 0

    while True:
        frame_no += 1
        final = top_view.copy()
        points_on_top = stream_output1[1] + stream_output2[1] + stream_output3[1]

        write_to_db_top(points_on_top, frame_no)
        draw_on_top(polygon, points_on_top, final)

        flag, encoded_image = cv2.imencode(".jpg", final)
        if not flag:
            continue

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'


def stream_heatmap():
    live_points = []

    while True:
        live_points += stream_output1[1] + stream_output2[1] + stream_output3[1]

        final = heatmap(live_points, top_view.copy())

        flag, encoded_image = cv2.imencode(".jpg", final)
        if not flag:
            continue

        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'


# HTML PAGES
# Home page for displaying the camera feeds
@app.route('/')
def index():
    return render_template("index.html")


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
        return redirect(url_for('offline', filename1=file.filename1, filename2=file.filename2, filename3=file.filename3))

    return None


@app.route('/static_image_heatmap')
def static_image_heatmap():
    if request.method == 'GET':
        filename1 = request.args.get('filename1')
        filename2 = request.args.get('filename2')
        filename3 = request.args.get('filename3')
        return render_template('heatmap.html', filename1=filename1, filename2=filename2, filename3=filename3)

# Home page for displaying the camera feeds
@app.route('/live')
def live():
    return render_template("live.html")


@app.route("/perimeter", methods=['GET', 'POST'])
def perimeter():
    if request.method == 'POST':
        vertices = json.loads(request.data)
        if len(vertices) <= 2:
            return Response(status=400)
        delete_count = db.session.query(Perimeter).delete()
        app.logger.info('%d vertex deleted from Perimeter table', delete_count)
        db.session.add_all([Perimeter(x=vertex[0], y=vertex[1]) for vertex in vertices])
        db.session.commit()
        global polygon, poly_lock
        with poly_lock:
            polygon = Polygon(vertices)
        return Response(status=200)

    return render_template("perimeter.html")


# LIVE VIEWS
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


@app.route("/live_homo")
def live_homo():
    return Response(stream_homo(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live_tripwire")
def live_tripwire():
    return Response(stream_tripwire(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live_heatmap")
def live_heatmap():
    return Response(stream_heatmap(), mimetype="multipart/x-mixed-replace; boundary=frame")


# OFFLINE VIEWS


@app.route('/offline_homo')
def offline_homo():
    filename1 = request.args.get('filename1', '')
    filename2 = request.args.get('filename2', '')
    filename3 = request.args.get('filename3', '')
    basepath = os.path.dirname(__file__)
    file_path1 = os.path.join(basepath, 'static', secure_filename(filename1))
    file_path2 = os.path.join(basepath, 'static', secure_filename(filename2))
    file_path3 = os.path.join(basepath, 'static', secure_filename(filename3))
    return Response(video_homography(file_path1, file_path2, file_path3, '720p'),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/static_heatmap')
def static_heatmap():
    filename1 = request.args.get('filename1', '')
    filename2 = request.args.get('filename2', '')
    filename3 = request.args.get('filename3', '')
    all_points = TopPoint.query.filter_by(source1=filename1, source2=filename2, source3=filename3).all()
    all_points = [[obj.x, obj.y] for obj in all_points]
    final = heatmap(all_points, top_view.copy())
    _, encoded_image = cv2.imencode(".jpg", final)

    return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n',
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/offline_tripwire')
def offline_tripwire():
    return Response(video_tripwire(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/offline_heatmap")
def offline_heatmap():
    return Response(video_heatmap(), mimetype="multipart/x-mixed-replace; boundary=frame")




















# #
# cam1_t = threading.Thread(target=detect_objects, args=(web_cam1, 'http://192.168.100.5:8080/video',
#                                                        f'cam1_{suffix}.avi', '480p',
#                                                        YOLO(), stream_output1, lock1, HR))  # Thread for camera 2
# cam1_t.daemon = True
# cam1_t.start()
#
#
# cam2_t = threading.Thread(target=detect_objects, args=(mob_cam2,
#                                                        0,  # 'http://10.47.27.57:8080/video',
#                                                        f'cam2_{suffix}.avi', '480p',
#                                                        YOLO(), stream_output2, lock2))  # Thread for camera 3
# cam2_t.daemon = True
# cam2_t.start()
#
#
# cam3_t = threading.Thread(target=detect_objects, args=(mob_cam3,
#                                                        0,  # 'http://192.168.100.7:8080/video',
#                                                        f'cam3_{suffix}.avi', '480p',
#                                                        YOLO(), stream_output3, lock3))  # Thread for camera 2
# cam3_t.daemon = True
# cam3_t.start()


if __name__ == '__main__':
    app.run(debug=True)
