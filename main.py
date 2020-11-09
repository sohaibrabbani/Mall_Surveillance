import time
from datetime import datetime
from imutils.video import FPS
import cv2

from yolo.yolo import YOLO

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(0)

fps = FPS().start()
yolo = YOLO()


start = datetime.now()


frames = []
count = 0
while True:

    t1 = time.time()
    ret, frame = cam.read()
    layerOutputs = yolo.predict(frame)

    cv2.imshow('frame', frame)
    current = datetime.now()
    frames.append(frame)
    fps.update()
    cv2.imwrite("data/frames/frame%d.jpg" % count, frame)
    count += 1
    # if (current - start).total_seconds() >= 10:
    #     break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fps.stop()
print(len(frames))
print(fps.fps(), fps.elapsed())

cam.release()
cv2.destroyAllWindows()

