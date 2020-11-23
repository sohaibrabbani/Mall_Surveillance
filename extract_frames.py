import cv2

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('data/raw_vids/inside.mp4')


if __name__ == '__main__':

    count = 0
    while True:
        ret, frame = cam.read()
        cv2.imshow('frame', frame)
        cv2.imwrite("data/frames3/frame%d.jpg" % count, frame)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

