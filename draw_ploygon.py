import numpy as np
from shapely.geometry import Polygon
import cv2


polygon = []


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        polygon.append((x, y))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(top_view, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('top', top_view)


top_view = cv2.imread('data/top_view_720p.jpg')
cv2.imshow('top', top_view)
cv2.setMouseCallback('top', click_event)


cv2.waitKey(0)
cv2.destroyAllWindows()

polygon = Polygon(polygon)
int_coords = lambda x: np.array(x).round().astype(np.int32)
exterior = [int_coords(polygon.exterior.coords)]
alpha = 0.5
overlay = top_view.copy()
cv2.fillPoly(overlay, exterior, color=(255, 255, 0))
cv2.addWeighted(overlay, alpha, top_view, 1 - alpha, 0, top_view)
cv2.imshow("Polygon", top_view)
cv2.waitKey(0)
cv2.destroyAllWindows()
