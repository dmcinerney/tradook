class Word:
    def __init__(self, boxes):
        self.boxes = []
        self.letters = []
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
        self.letters.append(None)

    def set_letters(self, letters):
        for i,letter in enumerate(letters):
            self.set_letter(i, letter)

    def set_letter(self, i, letter):
        if i < len(self.letters):
            self.letters[i] = letter
        else:
            print(self, i, letter)

        
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
        for l in self.letters:
            return_str += l
        return return_str