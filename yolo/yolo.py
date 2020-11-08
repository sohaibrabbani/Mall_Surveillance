import time

import cv2
import numpy as np

np.random.seed(42)


class YOLO:
    ACCEPTED_CLASSES = ["mask", "nomask"]

    def __init__(self, config_path='yolo/yolov3.cfg', weights_path='yolo/mask-yolov3_20000.weights'):
        self.classes = open('yolo/obj.names').read().strip().split("\n")
        self.COLORS = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def predict(self, frame, thresh=0.1):
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

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # if y >= (height / 2):  # Add: For detection of a boundary in the frame
                        # if not video_started:
                        #     frame_start = frame_number
                        #     frame_end = frame_number + 80
                        #     video_started = True
                        # # update our list of bounding box coordinates,
                        # # confidences, and class IDs
                        # # video_started = frame_end != frame_number
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in self.COLORS[class_ids[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.classes[class_ids[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # if frame_skip > 2:
                #     crop_img = frame[y:y + h, x:x + w]
                #     imageCounter += 1
                #     current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                #     image_name = "object-detection-%d.jpg" % imageCounter
                #     cv2.imwrite(image_name, crop_img)
                #     data.append([current_time, video_path + image_name, video_path + video_name + video_ext])
                #     frame_skip = 0
                # else:
                #     frame_skip += 1

        return frame
