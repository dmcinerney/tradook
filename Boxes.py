import numpy as np

class Box():
    def __init__(self,a, b):
        self.letter = None
        self.minX = a[0]
        self.highY = a[1]
        self.maxX = b[0]
        self.lowY = b[1]
        self.centerX, self.centerY = self.getCenter()

    def getCenter(self):
        return (self.minX + self.maxX) / float(2),  (self.lowY + self.highY) / float(2)

    def getArea(self):
        return (self.maxX - self.minX) * (self.lowY - self.highY)

    def getHeight(self):
        return self.lowY - self.highY

    def setLetter(self, l):
        self.letter = l

    def __str__(self):
        return "box: "+str(self.letter)
        # return '(' + str(self.centerX) + ', ' + str(self.centerY) + ')' + '(' + str(self.minX) + ', ' + str(self.highY) + ')' + '(' + str(self.maxX) + ', ' + str(self.lowY) + ')\n'

    def __repr__(self):
        return self.__str__()

# def compare_boxes(a, b):
#     if (a.centerX == b.centerX and a.centerY == b.centerY):
#         return 0
#     if (a.centerY < b.centerY or (a.centerY == b.centerY and  a.centerX < b.centerY)):
#         return -1
#     return 1

def compare_boxes(a, b):
    distA = np.sqrt(np.square(a.centerX) + np.square(a.centerY))
    distB = np.sqrt(np.square(b.centerX) + np.square(b.centerY))
    if (distA < distB):
        return -1
    if (distA == distB):
        if (a.centerY == b.centerY):
            return 0
        if (a.centerY < b.centerY):
            return -1
        else:
            return 1
    return 1

def get_center_distance(a, b):
    return np.sqrt(np.power(a.centerX - b.centerX, 2) + np.power(a.centerY - b.centerY, 2))

#return new range
def box_is_within_range(a, highY, lowY):
    if a.highY >= highY and a.highY <= lowY:
        if a.lowY > lowY:
            return [a.lowY, highY]
        else:
            return [lowY, highY]
    if a.lowY >= highY and a.lowY <= lowY:
        if a.highY < highY:
            return [lowY, a.highY]
        else:
            return [lowY, highY]
    else:
        return []

def compare_xs(a,b):
    if a.centerX == b.centerX:
        return 0
    if a.centerX < b.centerX:
        return -1
    return 1

def compare_areas(a, b): #returns -1 if a is smaller and 1 if b is smaller
    if a.getArea() <= b.getArea:
        return -1
    return 1
def point_is_in_box(x, y, box):
    if x >= box.minX and x <= box.maxX and y >= box.highY and y <= box.lowY:
        return True
    return False

def get_lines_point_at_x(x, slope, intercept):
    y = slope * x + intercept
    return x, y

def point_is_within_threshold(next_y, y, threshold):
    if np.abs(next_y - y) <= threshold:
        return True
    return False

def recalculate_values(slope, threshold, box, first_x, first_y, thres_divider):
    x = box.centerX
    y = box.centerY
    old_x = first_x
    old_y= first_y
    if (x - old_x) == 0:
        new_slope = 0
    else:
        new_slope = (y - old_y) / (x - old_x)
    avg_slope = (new_slope + slope) / 2
    new_intercept = first_y - avg_slope * first_x
    new_threshold = (threshold + (box.getHeight() / float(thres_divider))) / 2
    return avg_slope, new_intercept, new_threshold
