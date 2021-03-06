import matplotlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy.random as random
from Boxes import *
from sklearn.cluster import KMeans
import sys

# a = np.array([[8, 4], [2, 0], [4, 8], [10,6]])

# a = [Point(2,2), Point(3,3), Point(2,3), Point(1,1), Point(1,4)]
# print a
# a.sort(compare_points)
# print a
#
# b = list()
# b.append(1)
# b.append(2)
# b.append(3)
# b.append(None)
# print b

# boxes = [Box((280, 1500), (500, 1000)), Box((17, 31), (24, 42)), Box((30, 28), (35, 40)), Box((25, 100), (30, 130)), Box((32, 105), (39, 127))]
# def cluster():
#     boxes = [Box((250, 400), (500, 800)), Box((550, 450), (800, 820)), Box((850, 450), (1200, 820)) , Box((1350, 400), (1600, 800)), Box((1650, 400), (1900, 800)) , Box((1350, 1400), (1600, 1800)), Box((1650, 1400), (1900, 1800))]
#     # , Box((20, 2000), (400, 2500)), Box((450, 1800), (600, 2450))]
#     img = cv2.imread('clean_image.png')
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     vis = img_gray.copy()
#     # print vis.shape
#
#     boxes.sort(compare_boxes)
#
#     #remove duplicate boxes
#     i = 1
#     # print len(boxes)
#     boxes_size = len(boxes)
#     # cv2.rectangle(vis,(280, 1500) , (500, 1000), (0, 255, 255), 30)
#     # while i < boxes_size:
#     #     print 'ing'
#     #     print boxes[i]
#     #     cv2.rectangle(vis, (boxes[i].minX, boxes[i].highY), (boxes[i].maxX, boxes[i].lowY), (255, 255, 0), 30)
#     #     if get_center_distance(boxes[i], boxes[i - 1]) <= 5: #should tailor this value based on the sizes of boxes in the image
#     #         boxes.pop(i - 1)
#     #         boxes_size -= 1
#     #     else:
#     #         cv2.rectangle(vis, (boxes[i].lowY, boxes[i].maxX), (boxes[i].highY, boxes[i].minX), (255, 255, 0), 1)
#     #         i += 1
#     # print boxes
#     for letter in boxes:
#         cv2.rectangle(vis, (letter.minX, letter.highY), (letter.maxX, letter.lowY), (255, 255, 0), 30)
#     # plt.imshow(vis)
#     # plt.show()
#     # print boxes
#     lines = list()
#     line = list()
#     centers = list()
#     newLine = 1
#     lowY = boxes[0].lowY
#     highY = boxes[0].highY
#     for letter in boxes:
#         cv2.rectangle(vis, (letter.minX, letter.highY), (letter.maxX, letter.lowY), (255, 255, 0), 30)
#         # cv2.rectangle(vis, (letter.maxX, letter.lowY), (letter.minX, letter.highY), (255, 255, 0), 1)
#         new_range = box_is_within_range(letter, highY, lowY)
#         # print new_range
#         if len(new_range) != 0:
#             lowY, highY = new_range[0], new_range[1]
#             # print lowY
#             line.append(letter)
#             centers.append([letter.centerX, letter.centerY])
#         else:
#             # line.append(None)
#             lines.append(line)
#             cv2.line(vis, (0, lowY), (vis.shape[1], lowY),(0,255, 0), 20)
#             # cv2.line(vis, (0, highY), (vis.shape[1], highY),(0,255, 0), 20)
#             line = list()
#             line.append(letter)
#             centers.append([letter.centerX, letter.centerY])
#             lowY = letter.lowY
#             highY = letter.highY
#             newLine = 0
#
#     # line.append(None)
#     lines.append(line)
#     cv2.line(vis, (0, lowY), (vis.shape[1], lowY),(0,255, 0), 20)
#     plt.imshow(vis)
#     plt.show()
#     # cv2.line(vis, (0, highY), (vis.shape[1], highY),(0,255, 0), 20)
#
#     print lines
#     for line in lines:
#         num_boxes = len(line)
#         errors = list()
#         for i in np.arange(1, num_boxes):
#             kmeans = KMeans(n_clusters=i, random_state=0).fit(centers)
#             errors.append(kmeans.inertia_)
#
#         derivatives = np.ediff1d(errors)
#         if len(errors) == 1:
#             num_clusters = 1
#         else:
#             num_clusters = np.argmin(derivatives) + 2
#         print num_clusters
#
#         kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(centers)
#         print kmeans.labels_
#
#         words = list()
#         first_point = (line[0].minX, line[0].highY)
#         print len(line)
#         for i in np.arange(0, len(line)):
#             if i == len(line) - 1:
#                 second_point = (line[i].maxX, line[i].lowY)
#                 cv2.rectangle(vis, first_point, second_point, (0, 0, 255), 30)
#                 break
#             if kmeans.labels_[i] != kmeans.labels_[i+1]:
#                 second_point = (line[i].maxX, line[i].lowY)
#                 cv2.rectangle(vis, first_point, second_point, (0, 0, 255), 30)
#                 first_point = (line[i + 1].minX, line[i + 1].highY)
#
#     plt.imshow(vis)
#     plt.show()

# print sys.maxsize
# print (sys.maxsize + 100000000000000000000) / 2
# cluster()
# img = cv2.imread('image4.png')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
# z, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
# components = cv2.connectedComponentsWithStats(img_thresh, 3)
#
# print len(components[1])
# stats = components[2]
# centroids = components[3]
#
# # for i in rang(len(centroids)):
# for stat in stats:
#     minX = stat[0]
#     maxX = minX + stat[2]
#     minY = stat[1]
#     maxY = minY + stat[3]
#
#     cv2.rectangle(img_gray, (int(minX), int(minY)), (int(maxX), int(maxY)), (255, 255, 0), 1)
#
#
# plt.imshow(img_gray)
# plt.show()
# from autocorrect import spell
# print spell('helo')
# print spell('hlleo')
# print spell('banaxana')
# boxes = [Box((250, 400), (500, 800)), Box((550, 450), (800, 820)), Box((850, 450), (1200, 820)) , Box((1350, 400), (1600, 800)), Box((1650, 400), (1900, 800)) , Box((1350, 1400), (1600, 1800)), Box((1650, 1400), (1900, 1800))]
boxes =[Box((250, 400), (500, 800)), Box((550, 450), (800, 820)), Box((850, 450), (1200, 820)), Box((1350, 400), (1600, 800)), Box((1650, 400), (1900, 800)), Box((1350, 1400), (1600, 1800))]

def find_box_clicked(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
        boxes =[Box((550, 450), (800, 820)), Box((850, 450), (1200, 820)), Box((1350, 400), (1600, 800)), Box((1650, 400), (1900, 800)), Box((1350, 1400), (1600, 1800))]
        for box in boxes:
            if point_is_in_box(x, y, box):
                cv2.rect(img, (box.minX, box.highY), (box.maxX, box.lowY), (0, 255, 0), 3)
                return box

image = cv2.imread('image1.png')
cv2.namedWindow("image")
cv2.setMouseCallback("image", find_box_clicked)
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
