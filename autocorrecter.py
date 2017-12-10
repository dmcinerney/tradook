from autocorrect import spell
import Word
import enchant

def auto_correct_of_all_words(words):
    d = enchant.Dict("en_US")
    # potentials = []
    for i,word in enumerate(words):
        potentials = []
        potentials.append(spell(str(word)))
        if i+1 < len(words):
            potentials.append(spell(str(word)+str(words[i+1])))

        for j,potential in enumerate(potentials):
            if d.check(potential) and (j == 0 or potentials[j-1] != potential):
                word.content.append(potential)

    # final_words = []
    # for i,potential in enumerate(potentials):
    #     if d.check(potential) and (i == 0 or potentials[i-1] != potential):
    #         final_words.append(potential)
    # return final_words