import cv2
import imutils
from flask import Flask, Response, render_template
from imutils.video import VideoStream, WebcamVideoStream

outputFrame = None
app = Flask(__name__)

vs_mob1 = cv2.VideoCapture('http://192.168.100.11:8080/video')
vs_mob2 = cv2.VideoCapture('https://192.168.100.4:8080/video')
vs_web = VideoStream(src=0).start()


@app.route('/')
def hello_world():
    return render_template("index.html")


def stream_cam1(vs):
    while True:
        if isinstance(vs, WebcamVideoStream):
            frame = imutils.resize(vs.read(), height=400, width=400)
        elif isinstance(vs, cv2.VideoCapture):
            frame = imutils.resize(vs.read()[1], height=400, width=400)

        flag, encoded_image = cv2.imencode(".jpg", frame)
        # ensure the frame was successfully encoded
        if not flag:
            continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_image) + b'\r\n')


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/cam1")
def cam1():
    # return the response generated along with the specific media
    # type (mime type)
    # vs = VideoStream(src=0)
    # vs.start()
    return Response(stream_cam1(vs_web),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/cam2")
def cam2():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(stream_cam1(vs_mob1),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/cam3")
def cam3():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(stream_cam1(vs_mob2),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # start a thread that will perform motion detection
    # t = threading.Thread(target=stream_video)
    # t.daemon = True
    # t.start()
    app.run()

vs_web.stop()
