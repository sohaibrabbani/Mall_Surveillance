import time
from datetime import datetime
from imutils.video import FPS
import cv2

from utils import STD_DIMENSIONS


from yolo.yolo import YOLO

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('data/raw_vids/gate.mp4')

yolo = YOLO()


if __name__ == '__main__':

    count = 0
    while True:

        ret, frame = cam.read()
        # layerOutputs = yolo.predict(frame)

        cv2.imshow('frame', frame)
        cv2.imwrite("data/frames3/frame%d.jpg" % count, frame)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

