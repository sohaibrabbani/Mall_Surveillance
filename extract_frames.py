import cv2

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('http://192.168.100.13:8080/video')


if __name__ == '__main__':

    count = 0
    while True:
        ret, frame = cam.read()
        cv2.imshow('frame', frame)
        cv2.imwrite("data/frame/frameL.jpg", frame)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

