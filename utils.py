import os

import cv2
import numpy as np
from shapely.geometry import Point


def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
    "custom": (620, 400),
}


# Returns the new dimensions and resize the input source
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height


# Video file format to codec mapping
VIDEO_TYPE = {
    '.avi': cv2.VideoWriter_fourcc(*'XVID'),
    '.mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


# Returns the codec for given video file format
def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['.avi']


def plot_object(H, detections):
    # hp = H.dot(np.array([d['x']+d['w']/2, d['y']+d['h']/2, 1]))
    # hp = (hp/hp[2]).astype(np.int)
    if not detections:
        return []

    points = [[[d['x'] + d['w'] / 2, d['y'] + d['h'] / 2] for d in detections]]
    return cv2.perspectiveTransform(np.array(points), m=H).astype(np.int)[0].tolist()


def draw_on_top(polygon, points, top_view):
    def check_alert(points):
        return any([polygon.contains(Point(p[0], p[1])) for p in points])

    for point in points:
        cv2.circle(top_view, (point[0], point[1]), 5, (0, 0, 255), -1)

    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exterior = [int_coords(polygon.exterior.coords)]
    alpha = 0.5
    overlay = top_view.copy()
    if polygon:
        if check_alert(points):
            cv2.fillPoly(overlay, exterior, color=(0, 0, 255))
        else:
            cv2.fillPoly(overlay, exterior, color=(255, 255, 0))

    cv2.addWeighted(overlay, alpha, top_view, 1 - alpha, 0, top_view)
    return top_view

def stiching(a, b, c):
    a = a / 255
    b = b / 255
    c = c / 255

    final = np.zeros((a.shape))
    final = np.where(np.logical_and(a != 0, b != 0, c != 0), (a + b + c) / 3, final)
    final = np.where(np.logical_or(np.logical_and(a == 0, np.logical_xor(b == 0, c == 0)),
                                   np.logical_and(c == 0, np.logical_xor(a == 0, b == 0)),
                                   np.logical_and(b == 0, np.logical_xor(a == 0, c == 0))), a + b + c, final)
    final = np.where(np.logical_or(np.logical_and(a != 0, np.logical_xor(b == 0, c == 0)),
                                   np.logical_and(c != 0, np.logical_xor(a == 0, b == 0)),
                                   np.logical_and(b != 0, np.logical_xor(a == 0, c == 0))), (a + b + c) / 2, final)

    return (final * 255).astype(np.float32)
