from __future__ import division, print_function, absolute_import
import os
import sys
import datetime
import math
import warnings
import cv2
import numpy as np
import argparse

from PIL import Image
from yolo import YOLO
from timeit import time
from distance import calculateDistance
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from collections import deque
import tensorflow.compat.v1 as tf


warnings.filterwarnings('ignore', message= r"Passing", category=FutureWarning)
# Allow run on GPU
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True # Distribute small proportion, add when needed
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.InteractiveSession(config=sess_config) #Default


# Only allow 35 people recorded for memory relocation
pts = [deque(maxlen=35) for _ in range(9999)]

# initialize a list of colors to represent each possible class label
np.random.seed(69)

# Just to look fancy hehe
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


def violationRate(violation, total):
    # Little latency in detection cause total to be 0 at start
    try:
        violated = violation / total
    except ZeroDivisionError:
        violated = 0
    violated_percentage = round(violated * 100, 2)
    return violated_percentage


def socialdistanceDEEP(yolo, input_filename, output_filename):
    fps = 0.0
    start = time.time()
    #input
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.3

    counter = []
    # deep_sort
    model = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model, batch_size=1)

    # find_objects = ['person']

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    video_capture = cv2.VideoCapture(input_filename if input_filename else 0)

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_filename, fourcc, 15, (w, h))
        # statistics output
        list_file = open('detection_result.txt', 'w')
        list_file.write(str("Frame") + ',' + str("ID") + ',' + str("Boxes") + ',' + str("Violation") + ','
                        + str("Total People"))
        list_file.write('\n')
        frame_index = -1

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if not ret:
            break

        t1 = time.time()

        image = Image.fromarray(frame[..., ::-1])  # reverse the list bgr to rgb
        boxs, confidence, class_names = yolo.detect_image(image) # imitate the process of detection
        features = encoder(frame, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections]) #refer section 2.3 YOLO box
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []
        center2 = []
        co_info = []
        s_close_pair = []
        violated = set()

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)  # white

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            # print(frame_index)
            list_file.write(str(frame_index) + ',')
            list_file.write(str(track.track_id) + ',')
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 3)
            b0 = str(bbox[0])
            b1 = str(bbox[1])
            b2 = str(bbox[2] - bbox[0])
            b3 = str(bbox[3] - bbox[1])
            # boxes
            list_file.write(str(b0) + ',' + str(b1) + ',' + str(b2) + ',' + str(b3))

            # print(str(track.track_id))
            list_file.write('\n')
            # list_file.write(str(track.track_id)+',')
            cv2.putText(frame, "ID:" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150, color,
                        2)
            if len(class_names) > 0:
                class_name = class_names[0]
                cv2.putText(frame, str(class_names[0]), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, color, 2)

            i += 1
            # bbox_center_point(x,y)
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            # track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            # draw distance line
            (w, h) = (bbox[2], bbox[3])
            center2.append(center)
            co_info.append([w, h, center2])


            # calculateDistance
            if len(center2) > 2:
                for i in range(len(center2)):
                    for j in range(len(center2)):
                        # g = isclose(co_info[i],co_info[j])
                        # D = dist.euclidean((center2[i]), (center2[j]))
                        x1 = center2[i][0]
                        y1 = center2[i][1]
                        x2 = center2[j][0]
                        y2 = center2[j][1]
                        dis = calculateDistance(x1, y1, x2, y2)

                        if dis < 200:
                            # print(dis)
                            cv2.line(frame, (center2[i]), (center2[j]), (0, 255, 0), 2)
                            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 3)

                        if dis < 50:

                            cv2.line(frame, (center2[i]), (center2[j]), (0, 0, 255), 5)
                            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 3)
                            violated.add(violationRate(len(center2[i] + center2[j]), count))
                            cv2.putText(frame, "Violated Percentage: " + str(max(violated)) + "%", (int(20), int(160)), 0,
                                        5e-3 * 200, (0, 255, 0),
                                        2)
            else:
                pass

            # center point
            cv2.circle(frame, (center), 1, color, thickness)

            # draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)

        count = len(set(counter))
        list_file.write(str(violated) + ',')
        list_file.write(str(count))
        list_file.write('\n')
        cv2.putText(frame, "FPS: %f" % (fps * 2), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
        cv2.putText(frame, "Total Pedestrian Counter: " + str(count), (int(20), int(80)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "Current Pedestrian Counter: " + str(i), (int(20), int(120)), 0, 5e-3 * 200, (0, 255, 0), 2)
        # cv2.putText(frame, "Violation Counter: " + str(i), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 2)
        # cv2.putText(frame, "Violation Portion: " + str(i), (int(20), int(80)), 0, 5e-3 * 200, (0, 255, 0), 2)

        cv2.namedWindow("Analyzing...Press Q to quit", 0)
        cv2.resizeWindow('Analyzing...Press Q to quit', 1024, 768)
        cv2.imshow('Analyzing...Press Q to quit', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps = (fps + (1. / (time.time() - t1))) / 2
        out.write(frame)
        frame_index = frame_index + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Process Finished]")
    end = time.time()

    if len(pts[track.track_id]) is not None:
        print(input_filename[43:] + " There are in total " + str(count) + " " + str(class_name) + ' Found')

    else:
        print("[No Person Found]")
    video_capture.release()

    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


# input_file = 'input/test2_oxford.mp4'
# output_file = 'output/deeptest.avi'
# if __name__ == '__main__':
#     socialdistanceDEEP(YOLO(), input_file, output_file)

