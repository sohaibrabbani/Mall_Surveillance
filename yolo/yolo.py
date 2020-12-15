import cv2
import numpy as np
import cv2
import imutils
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from flask import Flask, Response, render_template
from imutils.video import VideoStream
# from keras import backend

from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50

np.random.seed(42)

values = ['Age1-16', 'Age17-30', 'Age31-45', 'Age46-60', 'Female', 'Male']

class YOLO:
    ACCEPTED_CLASSES = ["person"]

    def __init__(self, config_path='yolo/yolov3_big.cfg', weights_path='yolo/yolov3.weights', use_gpu=True):
        self.classes = open('yolo/coco.names').read().strip().split("\n")
        self.COLORS = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def predict(self, frame, thresh=0.1, model_par=None, valid_transform=None, draw=True, attribute_detect=False):
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.layer_names)
        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            # loop over each of the detections
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > thresh and self.classes[class_id] in self.ACCEPTED_CLASSES:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        detections = []
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                detection = {
                    'x': x, 'y': y,
                    'w': w, 'h': h,
                    'name': self.classes[class_ids[i]]
                }
                # draw a bounding box rectangle and label on the frame
                if draw:
                    color = [int(c) for c in self.COLORS[class_ids[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if attribute_detect:
                    image = Image.fromarray(frame[..., ::-1])
                    crop_img = image.crop([int(x), int(y), int(x+w), int(y+h)])
                    attributes, age_group, gender = demo_par(model_par, valid_transform, crop_img)
                    cv2.putText(frame, str(attributes), (int(x), int(y - 10)), 0, 5e-3 * 150, color, 2)
                    detection['gender'] = gender[0] if gender else None

                detections.append(detection)

        return detections


def demo_par(model, valid_transform, img):
    # load one image
    img_trans = valid_transform(img)
    imgs = torch.unsqueeze(img_trans, dim=0)
    imgs = imgs.cuda()
    valid_logits = model(imgs)
    valid_probs = torch.sigmoid(valid_logits)
    score = valid_probs.data.cpu().numpy()

    # show the score in the image
    txt_res = []
    age_group = []
    gender = []
    for idx in range(len(values)):
        if score[0, idx] >= 0.5:
            temp = '%s: %.2f ' % (values[idx], score[0, idx])
            if idx < 4:
                age_group.append(values[idx])
            else:
                gender.append(values[idx])
            txt_res.append(temp)
    return txt_res, age_group, gender
