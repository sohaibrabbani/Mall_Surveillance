import os

import cv2
import numpy as np


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
