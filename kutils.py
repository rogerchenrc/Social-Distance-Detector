import os
from os import path
import cv2


def crop_image(filename, output_filename, size=(10, 10)):
    img = cv2.imread(filename)
    scale_percent = 90

    # width = int(img.shape[1]*scale_percent/100)
    # height = int(img.shape[0]*scale_percent/100)
    # dsize = (width, height)

    output = cv2.resize(img, size)
    result = cv2.imwrite(output_filename, output)
    return result


def search_document(directory, output_file):
    output_name, output_extension = os.path.splitext(output_file)
    output_name = output_name.split("/")
    filename = []

    for f in os.listdir(directory):
        name, extension = path.splitext(f)
        filename.append(name)

    if output_name[-1] in filename:
        return output_name[-1]

    return None

#print(search_document("/home/kuanhaochen/Documents/social_distance/output", "output/v2test1.avi"))

