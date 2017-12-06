from PIL import Image
from edge_detection_testing import *
import numpy as np

info = get_interesting_areas('image1.jpg')
print(info[10].minX)


idx = 24
x1 = info[idx].minX
x2 = info[idx].maxX
y2 = info[idx].lowY
y1 = info[idx].highY


#print(x1,x2,y1,y2)
print("length", len(info))


# (centerX, centerY)(minX, highY)(maxX, lowY)

image = Image.open('image1.jpg')
image = image.convert('RGB')

new_image = image.crop((x1, y1, x2, y2))
#print(new_image)
#new_image = image
new_image = np.asarray(new_image)
max_value = np.amax(new_image)
print("max", max_value)
(vertical,horizontal,colors) = np.shape(new_image)
#next_image = np.zeros((vertical,horizontal, colors), dtype = "float32")
#new_image = new_image * int(255.0/max_value)
'''
for i in range(0, len(info)):
    idx = i
    x1 = info[idx].minX
    x2 = info[idx].maxX
    y2 = info[idx].lowY
    y1 = info[idx].highY
    cv2.rectangle(new_image,(x1,y1),(x2,y2),(0,255,0),3)
'''

#print(new_image)
cv2.imshow('image', new_image)
cv2.waitKey()
