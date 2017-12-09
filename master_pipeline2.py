import edge_detection_testing as ed
from letter_recognition_network import *
import interactive_image
import autocorrecter
from get_words import get_words, draw_words, draw_letters

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
    lines = ed.new_find_lines(list(boxes))
    lines = sorted(lines, key=lambda x:x[0].getCenter()[1])
    ed.draw_lines_for_testing(lines, image_filename)
    i += 1


    # input: array of boxes ordered
    # call function to form words from boxes
    # output: array of words with letters set to None
    print("step "+str(i)+": group boxes together into words")
    # words = ed.split_words(lines, image_filename)
    words = get_words(lines)

<<<<<<< HEAD
	# autocorrect all the words in words
	words = auto_correct_of_all_words(words)

	#display interactive image
	display_interactive_image(image_filename, words)

	print([str(word) for word in words])
	print("done")
=======
    img = cv2.imread(image_filename)
    draw_letters(boxes, img)
    draw_words(words, img)
    plt.imshow(img)
    plt.show()
    i += 1

    print([str(word) for word in words])
    # for word in words:
    #     print(str(word))
    #     for letter in word.boxes:
    #         print(str(letter),letter.minX,letter.maxX,letter.lowY,letter.highY)
    print("done")
>>>>>>> f2113e839f3feb492565f0b75a721667f84d0487


if __name__ == '__main__':
    image_filename = "image"+sys.argv[1]+".png"
    main(image_filename)
