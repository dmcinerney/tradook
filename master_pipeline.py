import edge_detection_testing as ed

def main(image_filename):
	# take in an image filename

	# step 1
	# call jack's function to get possible letter regions and pass in image_filename
	# ouput: an array of boxes ordered
	print("step 1: get interesting areas")
	boxes = ed.get_interesting_areas(image_filename)

	# step 2
	# input: array of boxes ordered
	# clean up boxes using centroid and area
	# output: an array of boxes ordered
	print("step 2: clean up boxes")
	boxes = ed.cleanup_boxes(boxes)

	# step 3
	# input: array of boxes ordered
	# load boxes into dataloader
	# output: Dataloader object ordered
	print("step 3: load into dataloader object")


	# step 4
	# input: DataLoader object
	# pao-net computes results and outputs predictions for each box in order
	# output: array of labels
	print("step 4: predict letters for each box using pao-net")


	# step 5
	# delete examples that are not letters
	print("step 5: delete non-letter boxes and group by line (and sort)")
	boxes = ed.remove_non_letters(boxes, results)
	lines = ed.sort_boxes(boxes, group_by_line=True)

	# step 6
	# input: array of boxes ordered
	# call function to form words from boxes
	# output: array of words with letters set to None
	print("step 6: group boxes together into words")

	# step 7
	# attach letters to words and do any possible letter label correction using spellcheck library
	print("step 7: attatch letters to words")

	print("done")


if __name__ == '__main__':
	image_filename = "image1.png"
	main(image_filename)
