import cv2
import numpy as np

from utils import STD_DIMENSIONS

frame_cor = [[406, 198], [793, 426], [727, 567], [777, 144], [1062, 328], [186, 410], [61, 229]]
top_cor = [[205, 235], [393, 193], [417, 219], [267, 130], [418, 124], [285, 285], [163, 305]]
top_view_points = np.array(top_cor)
garden_points = np.array(frame_cor)


def click_event_top(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        top_cor.append([x,y])
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


frame = cv2.imread('data/frame2/frame8.jpg')

cv2.imshow('frame', frame)
cv2.setMouseCallback('frame', click_event_garden)

H_outside = np.array([
    [2.96918550e-01, 1.69874720e+00, -1.41966972e+02],
    [-3.71748569e-01,  8.56329387e-01, 3.30452853e+02],
    [-1.92530358e-04, 2.88393061e-03, 1.00000000e+00]
])


while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

H = cv2.findHomography(garden_points, top_view_points)[0]
warp = cv2.warpPerspective(frame, H, STD_DIMENSIONS["720p"])
cv2.imwrite('wrapped_outside.jpg', warp)
print(H)