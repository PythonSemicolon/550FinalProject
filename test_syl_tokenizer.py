import rusyllab
import pylabeador

syllables_es = [pylabeador.syllabify(word) for word in "traer un vestido amarillo".split()]
print(syllables_es)

syllables_ru = [rusyllab.split_word(word) for word in "У меня есть красивая собака".split()]
print(syllables_ru)

syllables_ko = [[c for c in word] for word in "예쁜 개가 있어요".split()]
print(syllables_ko)