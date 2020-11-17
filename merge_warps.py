import cv2
import numpy as np


def merge(warp1, warp2):
    result = np.zeros(warp1.shape)
    blue1 = warp1[:, :, 0]
    blue2 = warp2[:, :, 0]
    red1 = warp1[:, :, 1]
    red2 = warp2[:, :, 1]
    green1 = warp1[:, :, 2]
    green2 = warp2[:, :, 2]

    return result

warp_out = cv2.imread('wrapped1.jpg')
warp_in = cv2.imread('wrapped_inside.jpg')

final = merge(warp_out, warp_in)
cv2.imshow('final', final)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
