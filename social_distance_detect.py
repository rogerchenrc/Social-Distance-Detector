# Main Code of social distancing detection
# time python social_distance_detect.py --input test2_oxford.mp4 --output output/output2.avi --display 0
import configuration as config
from people_detection import people_detect
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to input video")
ap.add_argument("-o", "--output", type=str, default="", help="path to output video")
ap.add_argument("-d", "--display", type=int, default=1, help="base path to YOLO directory")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")


# derive the paths to the YOLO weight and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov4.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov4.cfg"])

print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# GPU usage
if config.GPU_USAGE:
    print("[INFO] Setting preferable backend and target to CUDA")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# determine only the output layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

print("[INFO] Accessing Video Stream.....")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None


while True:
    (grabbed, frame) = vs.read()
    # error detection
    if not grabbed:
        break

    frame = imutils.resize(frame, width=700)
    results = people_detect(frame, net, ln, personidx=LABELS.index("person"))

    # Initialize the set of indexes that violate the minimum social distance
    # Random
    violate = set()

    # more than (at least) two object needs to be needed for distancing calculation
    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                if D[i, j] < config.MIN_DISTANCE:
                    violate.add(i)
                    violate.add(j)

    for(i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0) # green

        if i in violate:
            color = (0, 0, 255)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)


    text = "Roger's Demo, Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    cv2.putText(frame, "Social Distancing Detector", (210, 45), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255),2)
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)

