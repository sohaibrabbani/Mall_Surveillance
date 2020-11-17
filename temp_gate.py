import cv2
import numpy as np

from utils import STD_DIMENSIONS

frame_cor = [[804, 221], [760, 451], [919, 270], [770, 521], [358, 363], [985, 198], [1115, 419], [1258, 315], [368, 279], [509, 209], [493, 142], [758, 101], [536, 38], [652, 215], [194, 455], [726, 133]]
top_cor = [[371, 236], [410, 276], [427, 236], [455, 284], [297, 360], [267, 130], [479, 234], [418, 124], [220, 358], [163, 305], [113, 294], [5, 142], [56, 248], [268, 260], [334, 395], [72, 189]]
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

H = cv2.findHomography(garden_points, top_view_points)[0]
warp = cv2.warpPerspective(frame, H, STD_DIMENSIONS["720p"])
cv2.imwrite('wrapped_gate.jpg', warp)
print(H)

