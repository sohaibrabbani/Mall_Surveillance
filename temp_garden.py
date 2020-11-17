import cv2
import numpy as np

from utils import STD_DIMENSIONS

frame_cor = [[683, 145], [634, 176], [819, 414], [986, 630], [771, 684], [455, 572], [488, 101]]
top_cor = [[505, 362], [478, 342], [293, 356], [208, 363], [227, 321], [290, 278], [568, 295]]
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


frame = cv2.imread('data/frames/frame8.jpg')

cv2.imshow('frame', frame)
cv2.setMouseCallback('frame', click_event_garden)


while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(frame_cor)
print(top_cor)

top_view_points = np.array(top_cor)
garden_points = np.array(frame_cor)
H = [
    [-4.27399796e-01, 1.40282390e-01, 6.87836375e+02],
    [7.21461165e-02, 6.25402709e-01, 1.58922825e+02],
    [-7.49957146e-04, 2.30730214e-03, 1.00000000e+00]
]
# H = cv2.findHomography(garden_points, top_view_points)[0]
warp = cv2.warpPerspective(frame, H, STD_DIMENSIONS["720p"])
cv2.imwrite('wrapped_inside.jpg', warp)
print(H)

