class Word:
    def __init__(self):
        self.boxes = []
        self.minx = None
        self.miny = None
        self.maxx = None
        self.maxy = None
        
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