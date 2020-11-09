import time
from datetime import datetime
from imutils.video import FPS
import cv2

from yolo.yolo import YOLO

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('http://192.168.100.6:8080/video')

fps = FPS().start()
yolo = YOLO()


start = datetime.now()


frames = []

while True:
    t1 = time.time()
    ret, frame = cam.read()
    layerOutputs = yolo.predict(frame)

    cv2.imshow('frame', frame)
    current = datetime.now()
    frames.append(frame)
    fps.update()
    if (current - start).total_seconds() >= 10:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fps.stop()
print(len(frames))
print(fps.fps(), fps.elapsed())

cam.release()
cv2.destroyAllWindows()

