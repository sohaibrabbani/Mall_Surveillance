import cv2
import numpy as np

from utils import STD_DIMENSIONS

frame_cor = []
top_cor = []
top_view_points = np.array(top_cor)
garden_points = np.array(frame_cor)


def click_event_top(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        top_cor.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(top_view, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('top', top_view)


def click_event_garden(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        frame_cor.append([x,y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('frame', frame)


top_view = cv2.imread('data/top_view_720p.jpg')
cv2.imshow('top', top_view)
cv2.setMouseCallback('top', click_event_top)


frame = cv2.imread('data/frames3/frame8.jpg')

cv2.imshow('frame', frame)
cv2.setMouseCallback('frame', click_event_garden)


while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(frame_cor)
print(top_cor)

top_view_points = np.array(top_cor)
garden_points = np.array(frame_cor)
H = np.array([[-0.4267572874811169, 0.12202433258793428, 679.1648279868555],
              [0.07855525944118393, 0.5634066050345926, 156.2850020670053],
              [-0.0007529157598914953, 0.002160243169386294, 1.0]])

H = cv2.findHomography(garden_points, top_view_points)[0]
warp = cv2.warpPerspective(frame, H, STD_DIMENSIONS["720p"])
cv2.imwrite('wrapped_inside.jpg', warp)
print(H.tolist())
