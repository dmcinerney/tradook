import edge_detection_testing as ed
from letter_recognition_network import *

def main(image_filename):
	# take in an image filename

	# step 1
	# call jack's function to get possible letter regions and pass in image_filename
	# ouput: an array of boxes ordered
	boxes = ed.get_interesting_areas(image_filename)

	# step 2
	# input: array of boxes ordered
	# clean up boxes using centroid and area
	# output: an array of boxes ordered
	boxes = ed.cleanup_boxes(boxes)

	# step 3
	# input: array of boxes ordered
	# load boxes into dataloader
	# output: Dataloader object ordered


	# step 4
	# input: DataLoader object
	# pao-net computes results and outputs predictions for each box in order
	# output: array of labels
	results = letter_recognition_network(boxes, image_filename)


	# step 5
	# delete examples that are not letters
	boxes = ed.remove_non_letters(boxes, results)
	lines = ed.sort_boxes(boxes, group_by_line=True)

	# step 6
	# input: array of boxes ordered
	# call function to form words from boxes
	# output: array of words with letters set to None

	# step 7
	# attach letters to words and do any possible letter label correction using spellcheck library


if __name__ == '__main__':
	image_filename = ""
	main(image_filename)
