from autocorrect import spell
import Word

def auto_correct_of_all_words(words):
    for word in words:
        word.content = spell(str(word)[6:])
