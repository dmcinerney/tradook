import matplotlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from Boxes import *
from sklearn.cluster import KMeans
from Word import Word

def get_interesting_areas(image):
    mser = cv2.MSER_create()
    img = cv2.imread(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
    # vis = img_gray.copy()



    # regions = mser.detectRegions(vis, None)
    regions = mser.detectRegions(img_gray, None)
    boxes = list()
    # i = 1
    for i,region in enumerate(regions):

        region = np.array(region)
        minX = np.min(region[:, 0])
        maxX = np.max(region[:, 0])
        highestY = np.min(region[:, 1])
        lowestY = np.max(region[:, 1])
        box = Box((minX, highestY), (maxX, lowestY))
        boxes.append(box)
        # cv2.circle(vis, (int(box.centerX), int(box.centerY)), 2, (0, 0, 255), -1)
        cv2.rectangle(img, (maxX, lowestY), (minX, highestY), (150, (i + 10) % 255, 0), 1)
    plt.imshow(img)
    plt.show()
    return boxes



def cleanup_boxes(boxes, img):
    img = cv2.imread(img)
    boxes = sort_boxes(boxes)
    #remove duplicate boxes
    i = 1
    print len(boxes)
    boxes_size = len(boxes)
    while i < boxes_size:
        if get_center_distance(boxes[i], boxes[i - 1]) <= 5 and np.abs(boxes[i].getArea() - boxes[i-1].getArea()) <= 100: #should tailor this value based on the sizes of boxes in the image
            boxes.pop(i - 1)
            boxes_size -= 1
        else:
            i += 1
    print len(boxes)
    for box in boxes:
        cv2.rectangle(img, (box.maxX, box.lowY), (box.minX, box.highY), (0, 255, 0), 1)
    plt.imshow(img)
    plt.show()
    return boxes
    # print boxes

def find_lines(boxes):
    #determine the lines
    lines = list()
    line = list()
    centers = list()
    newLine = 1
    lowY = boxes[0].lowY
    highY = boxes[0].highY
    for letter in boxes:
        new_range = box_is_within_range(letter, highY, lowY)
        if len(new_range) != 0:
            lowY, highY = new_range[0], new_range[1]
            line.append(letter)
            centers.append([letter.centerX, letter.centerY])
        else:
            lines.append(line)
            line = list()
            line.append(letter)
            centers.append([letter.centerX, letter.centerY])
            lowY = letter.lowY
            highY = letter.highY
            newLine = 0

    lines.append(line)
    # cv2.line(vis, (0, lowY), (vis.shape[1], lowY),(0,255, 0), 2)
    # cv2.line(vis, (0, highY), (vis.shape[1], highY),(0,255, 0), 2)
    # plt.imshow(vis)
    # plt.show()
    return lines

def draw_lines_for_testing(lines, img):
    for line in lines:
        for i in range(0, len(line) - 1):
            cv2.line(img, (line[i].centerX, line[i].centerY), (line[i + 1].centerX, line[i + 1].centerY), 2)

def new_find_lines(boxes):
    boxes = sorted(boxes, cmp=compare_xs)
    lines = list()
    while (len(boxes) != 0):
        line = list()
        i = 0
        slope = 0
        intercept = boxes[0].centerY
        first_x = boxes[0].centerX
        first_y = boxes[0].centerY
        # print boxes[0].getHeight()
        thres_divider = 3
        threshold = float(boxes[0].getHeight()) / float(thres_divider) #might have to change this value (3)
        # print '$$$$',threshold
        num_boxes_in_line = 0
        while i < len(boxes):
            # print len(boxes)
            next_x, next_y = get_lines_point_at_x(boxes[i].centerX, slope, intercept)
            # print next_x, next_y
            if point_is_in_box(next_x, next_y, boxes[i]) and point_is_within_threshold(next_y, boxes[i].centerY, threshold):
                line.append(boxes[i])
                slope, intercept, threshold = recalculate_values(slope, threshold, boxes[i], first_x, first_y, thres_divider)
                # print '****', slope, intercept, threshold
                boxes.pop(i)
            else:
                # lines.append(line)
                i+=1
        lines.append(line)
    # lines.append(line)
    return lines

#cluster per line
def split_words(lines, img):
    img = cv2.imread(img)
    words = list()
    for line in lines:
        # centers = [[box.centerX, box.centerY] for box in line]
        centers = [[box.centerX] for box in line]
        num_boxes = len(line)
        errors = list()
        for i in np.arange(1, num_boxes):
            kmeans = KMeans(n_clusters=i, random_state=0).fit(centers)
            errors.append(kmeans.inertia_)

        if len(errors) == 0:
            continue
        if len(errors) == 1:
            num_clusters = 1
        else:
            derivatives = np.ediff1d(errors)
            num_clusters = np.argmin(derivatives) + 1
        # print num_clusters

        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(centers)

        letters_in_word = list()
        label = kmeans.labels_[0]
        first_point = (line[0].minX, line[0].highY)
        # words.append(Word([line[0]]))
        for i in np.arange(0, len(line)):
            if i == len(line) - 1:
                second_point = (line[i].maxX, line[i].lowY)
                letters_in_word.append(line[i])
                print letters_in_word
                words.append(Word(letters_in_word))
                cv2.rectangle(img, first_point, second_point, (0, 0, 255), 2)
                break
            if kmeans.labels_[i] != kmeans.labels_[i+1]:
                second_point = (line[i].maxX, line[i].lowY)
                letters_in_word.append(line[i])
                words.append(Word(letters_in_word))
                cv2.rectangle(img, first_point, second_point, (0, 0, 255), 2)
                first_point = (line[i + 1].minX, line[i + 1].highY)

            # if kmeans.labels_[i-1] != kmeans.labels_[i]:
            #     words.append(Word([line[i]]))
    plt.imshow(img)
    plt.show()
    return words


def remove_non_letters(boxes, results):
    i = 0
    boxes_size = len(boxes)
    while i < boxes_size:
        if results[i] == '-': #should tailor this value based on the sizes of boxes in the image
            boxes.pop(i)
            results.pop(i)
            boxes_size -= 1
        else:
            i += 1

    return boxes, results


def grouper(iterable, threshold, function):
    iterable = sorted(iterable, key=function)
    prev = None
    group = []
    for item in iterable:
        if prev is None or function(item) - function(prev) <= threshold:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def sort_boxes(boxes, group_by_line=False):
    # print(boxes)
    new_boxes = []
    for line in grouper(boxes, 10, lambda box:box.getCenter()[1]):
        if group_by_line:
            new_boxes.append([])
        for box in sorted(line, key=lambda box:box.getCenter()[0]):
            if group_by_line:
                new_boxes[-1].append(box)
            else:
                new_boxes.append(box)
    return new_boxes


boxes = [Box((0, 4), (2, 6)), Box((3, 3.5), (5, 6))]
boxes = [Box((2, 0), (4, 2)), Box((5, 0), (7, 2)), Box((8, 0), (10, 2)), Box((0, 4), (2, 6)), Box((3, 3.5), (5, 6))]
print boxes
boxes = sorted(boxes, cmp=compare_xs)
print boxes
lines = new_find_lines(boxes)
print lines
