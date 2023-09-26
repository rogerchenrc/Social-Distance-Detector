# python social_distance_detect_v2.py --input input/test2_oxford.mp4 --output output/v2test1.avi
import time
import sys
import cv2
import numpy as np
import os
import argparse
import configuration as config
import kutils
from configuration import NMS_THRESH, MIN_CONF
from distance import isClose

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to input video")
ap.add_argument("-o", "--output", type=str, default="", help="path to output video")
ap.add_argument("-d", "--display", type=int, default=1, help="base path to YOLO directory")
args = vars(ap.parse_args())
np.random.seed(42)

# txt_filename = kutils.search_document("/home/kuanhaochen/Documents/social_distance/output", args["output"])
# sys.stdout = open("Data1_{}.txt".format(txt_filename), "w")

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")


# derive the paths to the YOLO weight and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov4.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov4.cfg"])

print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# determine only the output layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#fps
prev_fps = 0
new_fps = 0

vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
(W, H) = (None, None)

q = 0

while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        q = W

    frame = frame[0:H, 200:q]
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()
    layer_outputs = net.forward(ln)
    end = time.time()

    print("[INFO] YOLO tool {:.6f} seconds".format(end - start))

    # bounding boxes around the object
    boxes = []
    confidences = []
    # Detected object's class label
    classIDs = []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the detections
        for detection in output:
            scores = detection[5:]
            # class with the maximum value
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Two conditions needs to be met
            # 1. Confidence larger than defined value
            # 2. Only "PERSON"
            if LABELS[classID] == "person" and confidence > MIN_CONF:
                # [centerX, centerY, width, height]
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype(int)
                # calculate the left upper point
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                classIDs.append(classID)
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    if len(idxs) > 0:

        status = list()
        idf = idxs.flatten()
        close_pair = list()
        s_close_pair = list()
        center = list()
        dist = list()

        for i in idf:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(0)
        for i in range(len(center)):
            for j in range(len(center)):
                g = isClose(center[i], center[j])

                if g == 1:
                    close_pair.append([center[i], center[j]])
                    status[i] = 1
                    status[j] = 1

                elif g == 2:
                    s_close_pair.append([center[i], center[j]])
                    if status[i] != 1:
                        status[i] = 2
                    if status[j] != 1:
                        status[j] == 2

        total_p = len(center)
        low_risk_p = status.count(2)
        high_risk_p = status.count(1)
        safe_p = status.count(0)
        kk = 0

        for i in idf:
            # Sub Frame
            sub_img = frame[10:170, 10:W - 10] # change later
            black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
            # blending two images
            res = cv2.addWeighted(sub_img, 0.77, black_rect, 0.23, 1.0)

            frame[10:170, 10:W-10] = res
            # First SubFrame
            cv2.putText(frame, "KHC SOCIAL DISTANCING ANALYZER", (210, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (20, 60), (510, 160), (170,170, 170), 2)
            cv2.putText(frame, "Connecting lines shows closeness among people. ", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, "--YELLOW: CLOSE ", (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, "--RED   : TOO CLOSE ", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Second subFrame
            cv2.rectangle(frame, (535, 60), (W-20, 160), (170, 170, 170), 2)
            cv2.putText(frame, "Bounding box shows the level of risk to the person ", (545, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, "[]DARK RED: HIGH RISK ", (565, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
            cv2.putText(frame, "[]ORANGE   : LOW RISK ", (565, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)
            cv2.putText(frame, "[]GREEN    : SAFE", (565, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Third subFrame below
            tot_str = "TOTAL COUNT:" + str(total_p)
            high_str = "HIGH RISK COUNT:" + str(high_risk_p)
            low_str = "LOW RISK COUNT:" + str(low_risk_p)
            safe_str = "SAFE COUNT:" + str(safe_p)

            sub_img = frame[H-120:H, 0:210]
            black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
            res = cv2.addWeighted(sub_img, 0.8, black_rect, 0.2, 1.0)

            frame[H-120:H, 0:210] = res
            cv2.putText(frame, tot_str, (10, H-90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, high_str, (10, H-65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 1)
            cv2.putText(frame, low_str, (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 1)
            cv2.putText(frame, safe_str, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            if status[kk] == 1:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 150), 2)

            elif status[kk] == 0:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 120, 255), 2)
            kk += 1
        for h in close_pair:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)

        for b in s_close_pair:
            cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)

    # new_fps = time.time()
    # fps = 1 / (new_fps - prev_fps)
    # prev_fps = new_fps
    # fps = float(fps)
    # fps = str(fps)
    fps = vs.get(cv2.CAP_PROP_FPS)
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame, "FPS:{}".format(fps), (W-150, H-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)

    if args["display"] > 0:
        # if check videocapture
        cv2.imshow('Social Distancing Analyzer', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)

# sys.stdout.close()
print("Process finished:{}".format(args["output"]))
writer.release()
vs.release()
