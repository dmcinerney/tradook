import matplotlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from Boxes import *
from sklearn.cluster import KMeans

# def auto_canny(img, sigma=0.33):
#     median = np.median(img)
#     lower = int(max(0, (1 - sigma)* sigma))
#     upper = int(min(255, (1 + sigma)* sigma))
#
#     return cv2.Canny(img, lower, upper)


mser = cv2.MSER_create()
# for i in range(1,7):
img = cv2.imread('clean_image.png')
print img.shape
print len(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = img_gray.copy()
# vis = cv2.Canny(img_gray, 30, 200)
# plt.imshow(vis)
# plt.show()
img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)


# regions = mser.detectRegions(vis, None)
regions = mser.detectRegions(img_gray, None)
boxes = list()
# i = 1
for region in regions:

    region = np.array(region)
    minX = np.min(region[:, 0])
    maxX = np.max(region[:, 0])
    highestY = np.min(region[:, 1])
    lowestY = np.max(region[:, 1])
    box = Box((minX, highestY), (maxX, lowestY))
    boxes.append(box)
    cv2.circle(vis, (int(box.centerX), int(box.centerY)), 2, (0, 0, 255), -1)
    cv2.rectangle(vis, (maxX, lowestY), (minX, highestY), (255, 0, 0), 2)

boxes.sort(compare_boxes)
#remove duplicate boxes
i = 1
print len(boxes)
boxes_size = len(boxes)
while i < boxes_size:
    if get_center_distance(boxes[i], boxes[i - 1]) <= 5: #should tailor this value based on the sizes of boxes in the image
        boxes.pop(i - 1)
        boxes_size -= 1
    else:
        i += 1
print boxes

#determine the lines
lines = list()
line = list()
centers = list()
newLine = 1
lowY = boxes[0].lowY
highY = boxes[0].highY
for letter in boxes:
    cv2.rectangle(vis, (letter.minX, letter.highY), (letter.maxX, letter.lowY), (255, 255, 0), 30)
    new_range = box_is_within_range(letter, highY, lowY)
    print new_range
    if len(new_range) != 0:
        lowY, highY = new_range[0], new_range[1]
        line.append(letter)
        centers.append([letter.centerX, letter.centerY])
    else:
        lines.append(line)
        cv2.line(vis, (0, lowY), (vis.shape[1], lowY),(0,255, 0), 20)
        cv2.line(vis, (0, highY), (vis.shape[1], highY),(0,255, 0), 20)
        line = list()
        line.append(letter)
        centers.append([letter.centerX, letter.centerY])
        lowY = letter.lowY
        highY = letter.highY
        newLine = 0

lines.append(line)
cv2.line(vis, (0, lowY), (vis.shape[1], lowY),(0,255, 0), 20)
cv2.line(vis, (0, highY), (vis.shape[1], highY),(0,255, 0), 20)

#cluster per line
for line in lines:
    num_boxes = len(line)
    errors = list()
    for i in np.arange(1, num_boxes):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(centers)
        errors.append(kmeans.inertia_)

    if len(errors) == 1:
        num_clusters = 1
    else:
        derivatives = np.ediff1d(errors)
        num_clusters = np.argmin(derivatives) + 2
    print num_clusters

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(centers)
    print kmeans.labels_

    words = list()
    label = kmeans.labels_[0]
    first_point = (line[0].minX, line[0].highY)
    for i in np.arange(0, len(line)):
        if i == len(line) - 1:
            second_point = (line[i].maxX, line[i].lowY)
            cv2.rectangle(vis, first_point, second_point, (0, 0, 255), 30)
            break
        if kmeans.labels_[i] != kmeans.labels_[i+1]:
            second_point = (line[i].maxX, line[i].lowY)
            cv2.rectangle(vis, first_point, second_point, (0, 0, 255), 30)
            first_point = (line[i + 1].minX, line[i + 1].highY)

plt.imshow(vis)
plt.show()
