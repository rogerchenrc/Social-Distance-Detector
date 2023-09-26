import numpy as np
import math


def euclideanDistance(x1,y1,x2,y2):
    distance = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return distance


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def dist(c1, c2, calibrate=True):
    if calibrate:
        distance = ((c1[0] - c2[0]) ** 2 + 550 / ((c1[1] + c2[1]) / 2) * (c1[1] - c2[1]) ** 2) ** 0.5
    else:
        distance = np.sqrt(np.square(c1[0] - c2[0]) + np.square(c1[1] - c2[1]))

    return distance


def isClose(p1, p2):
    c_d = dist(p1, p2)
    calibration = (p1[1] + p2[1]) / 2
    a = 0.15 * calibration
    b = 0.2 * calibration
    if 0 < c_d < a:
        return 1
    elif 0 < c_d < b:
        return 2
    else:
        return 0

