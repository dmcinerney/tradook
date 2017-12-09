class Word:
    def __init__(self, boxes):
        self.boxes = []
        self.minx = None
        self.miny = None
        self.maxx = None
        self.maxy = None
        for box in boxes:
            self.boxes.append(box)
        self.update_box()
        
    def append(self, box):
        self.boxes.append(box)
        self.update_box()

        
    def update_box(self):

        self.minx = min([min(box.minX,box.maxX) for box in self.boxes])
        self.miny = min([min(box.highY,box.lowY) for box in self.boxes])
        self.maxx = max([max(box.minX,box.maxX) for box in self.boxes])
        self.maxy = max([max(box.highY,box.lowY) for box in self.boxes])
        
    def position(self):
        self.update_box()
        return [(self.minx + self.maxx)/2, (self.miny+self.maxy)/2]

    def __len__(self):
        return len(self.boxes)

    def __str__(self):
        return_str = "word: "
        for box in self.boxes:
            return_str += str(box.letter)
        return return_str

from Boxes import Box
if __name__ == '__main__':
    w = Word([Box((1,1),(1,1)), Box((2,2),(2,2)), Box((3,1),(1,1)), Box((4,1),(1,1)), Box((5,1),(1,1))])
    print(w.minx, w.miny, w.maxx, w.maxy, w.boxes)
