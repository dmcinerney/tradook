import numpy as np

class Box():
    def __init__(self,a, b):
        self.minX = a[0]
        self.highY = a[1]
        self.maxX = b[0]
        self.lowY = b[1]
        self.centerX, self.centerY = self.getCenter()

    def getCenter(self):
        return (self.minX + self.maxX) / float(2),  (self.lowY + self.highY) / float(2)

    def __str__(self):
        return '(' + str(self.centerX) + ', ' + str(self.centerY) + ')' + '(' + str(self.minX) + ', ' + str(self.highY) + ')' + '(' + str(self.maxX) + ', ' + str(self.lowY) + ')\n'

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
