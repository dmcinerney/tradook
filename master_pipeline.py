import edge_detection_testing as ed

def main(image_filenamme):
	# take in an image filename

	# step 1
	# call jack's function to get possilbe letter regions and pass in image_filename
	# ouput: an array of boxes ordered
	boxes = ed.get_interesting_areas(image_filenamme)

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

	# step 5
	# zip boxes with labels and delete examples that are not letters

	# step 6
	# input: array of boxes ordered
	# call function to form words from boxes
	# output: array of words with letters set to None

	# step 7
	# attach letters to words and do any possible letter label correction using spellcheck library


if __name__ == '__main__':
	image_filenamme = ""
	main(image_filenamme)