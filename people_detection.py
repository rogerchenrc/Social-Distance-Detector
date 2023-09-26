from configuration import NMS_THRESH
from configuration import MIN_CONF
import numpy as np
import time
import cv2
import os

'''
@TODO Only people detection
@param frame: Frame from video file
@param net: Pre-initialized/pretrained YOLO object detection model
@param ln: Output layer names from YOLO CNN
@param personIdx: Only people is required for detetction
@return 1) person prediction probability 2) bounding box coordinates for the detection 3) centroid of the object
'''


def people_detect(frame, net, ln, personidx=0):
    # Dimensions of frame
    (H, W) = frame.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    # start = time.time()
    layer_outputs = net.forward(ln)
    # end = time.time()

    # print("[INFO] YOLO tool {:.6f} seconds".format(end - start))

    # bounding boxes around the object
    boxes = []
    centroids = []
    # Detected object's class label
    confidences = []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the detections
        for detection in output:
            scores = detection[5:]
            # class with the maximum value
            classidx = np.argmax(scores)
            confidence = scores[classidx]

            # Two conditions needs to be met
            # 1. Confidence larger than defined value
            # 2. Only "PERSON"
            if classidx == personidx and confidence > MIN_CONF:
                # [centerX, centerY, width, height]
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype(int)
                # calculate the left upper point
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            r = (confidences[i], (x, y, x+w, y+h), centroids[i])
            results.append(r)

    return results

