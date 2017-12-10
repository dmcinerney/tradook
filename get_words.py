from Word import Word
import numpy as np
import matplotlib.pyplot as plt
import cv2
from edge_detection_testing import grouper

def box_diff(box1, box2):
    return box1.minX - box2.maxX
    # return box1.getCenter()[0]-box2.getCenter()[0]

def get_words(lines):
    cuttoff = 3
    words = []
    for line in lines:
        boxes = sorted(line, key=lambda x:x.getCenter()[0])
        if len(boxes) == 1:
            words.append(Word([boxes[0]]))
            continue
        avg_width = np.mean([box.getWidth() for box in boxes])
        print(avg_width)
        diff = [boxes[i+1].getCenter()[0]-boxes[i].getCenter()[0] for i in range(len(boxes)-1)]
        diff = [box_diff(boxes[i+1],boxes[i]) for i in range(len(boxes)-1)]
        diff_groups = list(grouper(diff, avg_width/4, lambda x:x))
        print(diff_groups)
        diff_group = min(diff_groups, key=lambda x:np.abs(np.mean(x)))

        # plt.hist(diff, 20)
        # plt.show()
        mean = min(avg_width/2, np.mean(diff_group))
        if len(diff_group) > 2:
            std = np.std(diff_group)
        else:
            std = avg_width/6
        # std = avg_width/2
        print(mean, std)

        print(len(boxes))
        for i,box in enumerate(boxes):
            if i > 0:
                print(box.letter, box_diff(box,words[-1].boxes[-1]))
            if i == 0 or box_diff(box,words[-1].boxes[-1]) > mean+cuttoff*std:
                words.append(Word([box]))
                # print("starting new word")
            elif box_diff(box,words[-1].boxes[-1]) > mean-cuttoff*std:
                words[-1].append(box)
                # print("appending box to word")

    return words

def draw_letters(boxes, img):
    # print(len(boxes),boxes)
    for box in boxes:
        first_point = (int(box.minX), int(box.highY))
        second_point = (int(box.maxX), int(box.lowY))
        cv2.rectangle(img, first_point, second_point, (0, 255, 0), 2)

def draw_words(words, img):
    for word in words:
        draw_letters(word.boxes, img)
        first_point = (word.minx, word.miny)
        second_point = (word.maxx, word.maxy)
        cv2.rectangle(img, first_point, second_point, (255, 0, 0), 2)


if __name__ == '__main__':
    from Boxes import Box
    lines = [[Box((0,0),(20,0)),Box((20,0),(40,0)),Box((40,0),(60,0)),Box((60,0),(80,0)),Box((100,0),(120,0)),Box((120,0),(140,0)),Box((140,0),(160,0)),Box((160,0),(180,0))]]
    words = get_words(lines)
    print(words)
