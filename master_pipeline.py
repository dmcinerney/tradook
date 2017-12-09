import edge_detection_testing as ed
from letter_recognition_network import *

def main(image_filename):
	# take in an image filename
	i = 1

	# call jack's function to get possible letter regions and pass in image_filename
	# ouput: an array of boxes ordered
	print("step "+str(i)+": get interesting areas")
	boxes = ed.get_interesting_areas(image_filename)
	i += 1


	# input: array of boxes ordered
	# clean up boxes using centroid and area
	# output: an array of boxes ordered
	print("step "+str(i)+": clean up boxes")
	boxes = ed.cleanup_boxes(boxes, image_filename)
	i += 1


	# input: boxes, image filename
	# pao-net computes results and outputs predictions for each box in order
	# output: array of labels
	print("step "+str(i)+": predict letters for each box using pao-net")
	results = letter_recognition_network(boxes, image_filename)
	i += 1


	# delete examples that are not letters
	print("step "+str(i)+": delete non-letter boxes and group by line (and sort)")
	boxes, results = ed.remove_non_letters(boxes, results)
	for j,box in enumerate(boxes):
		box.setLetter(results[j])
	lines = ed.sort_boxes(boxes, group_by_line=True)
	i += 1


	# input: array of boxes ordered
	# call function to form words from boxes
	# output: array of words with letters set to None
	print("step "+str(i)+": group boxes together into words")
	words = ed.split_words(lines, image_filename)
	i += 1

	print([str(word) for word in words])
	print("done")


if __name__ == '__main__':
	image_filename = "image"+sys.argv[1]+".png"
	main(image_filename)
