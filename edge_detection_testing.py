import matplotlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def auto_canny(img, sigma=0.33):
    median = np.median(img)
    lower = int(max(0, (1 - sigma)* sigma))
    upper = int(min(255, (1 + sigma)* sigma))

    return cv2.Canny(img, lower, upper)


mser = cv2.MSER_create()
image_letters = []
for i in range(1,7):
    image_letters.append([])
    img = cv2.imread('image' + str(i) + '.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img_gray.copy()
    # vis = cv2.Canny(img_gray, 30, 200)
    # plt.imshow(vis)
    # plt.show()
    img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)
    # regions = mser.detectRegions(vis)
    regions, _ = mser.detectRegions(img_gray)
    # i = 1
    for region in regions:

        region = np.array(region)
        minX = np.min(region[:, 0])
        maxX = np.max(region[:, 0])
        highestY = np.min(region[:, 1])
        lowestY = np.max(region[:, 1])

        image_letters[-1].append((None, minX, lowestY, maxX, highestY))

        cv2.rectangle(vis, (maxX, lowestY), (minX, highestY), (255, 0, 0), 2)

    # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    plt.imshow(vis)
    plt.show()

for i,letters in enumerate(image_letters, 1):
    img = cv2.imread('image' + str(i) + '.png')
    image_folder = 'image' + str(i)
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    for j,letter in enumerate(letters):
        cv2.imwrite(os.path.join(image_folder,'image'+str(j)+'.png'), img[letter[1]:letter[3],letter[2]:letter[4]])
    
with open('image_letter_positions.txt', 'w') as f:
    for letters in image_letters:
        f.write(str(letters)+'\n')