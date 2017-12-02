import numpy as np
import matplotlib.pyplot as plt

class Letter:
	def __init__(self, letter, p1x, p1y, p2x, p2y):
		self.letter = letter
		self.x1 = p1x
		self.y1 = p1y
		self.x2 = p2x
		self.y2 = p2y

	def size(self):
		return np.abs(self.x1-self.x2)*np.abs(self.y1-self.y2)

	def position(self):
		return ((self.x1+self.x2)/2, (self.y1+self.y2)/2)

def grouper(iterable, threshold):
    prev = None
    group = []
    for item in iterable:
        if not prev or item - prev <= threshold:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group

def main(letters_file):
	image_letters = []
	with open(letters_file, 'r') as f:
		for line in f:
			image_letters.append([Letter(*letter) for letter in eval(line)])
	for i,letters in enumerate(image_letters):
		sizes = np.array([l.size() for l in letters])
		# groups = dict(enumerate(grouper(sizes, 15)))
		x_distances = np.array([l.position()[0] for l in letters])
		print(x_distances)
		y_distances = np.array([l.position()[1] for l in letters])
		print(groups)
		if i == 0: break


if __name__ == '__main__':
	main('image_letter_positions.txt')