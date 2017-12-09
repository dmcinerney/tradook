import cv2
import numpy as np
from Boxes import *
from edge_detection_testing import get_interesting_areas

def find_box_clicked(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global all_words
        for box in all_words:
            if point_is_in_box(x, y, box):
                print x,y
                cv2.rectangle(img, (box.minX, box.highY), (box.maxX, box.lowY), (0, 255, 0), 3)
                cv2.imshow('original', img)

def display_interactive_image(image_filename, words):
    global all_words
    all_words = words
    img = cv2.imread(image_filename)
    cv2.namedWindow('original')
    cv2.setMouseCallback('original', find_box_clicked)
    while(1):
        cv2.imshow('original',img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break
    cv2.destroyAllWindows()
