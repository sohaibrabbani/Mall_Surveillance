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
        frame_cor.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('frame', frame)


top_view = cv2.imread('static/top_view_720p.jpg')
cv2.imshow('top', top_view)
cv2.setMouseCallback('top', click_event_top)


frame = cv2.imread('data/frame/frameL.jpg')

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
# HL = np.array([[-0.17426423209765862, -1.031394596052239, 750.412813145867],
#                [1.0000328784997377, 0.14136499853820997, -10.370851456391206],
#                [0.00026540226307438837, 0.0003478893717957895, 1.0]])
#
# HR = np.array([[0.3017907086218721, -1.2416167716780655, 1021.5367338709084],
#                [0.8382672585685141, 0.03898954281256336, 114.78088045652038],
#                [-8.366779008655821e-05, -0.00014174378491588216, 1.0]])

warp = cv2.warpPerspective(frame, H, STD_DIMENSIONS["720p"])
cv2.imwrite('wrapped_L.jpg', warp)
print(H.tolist())
