# inspired from https://github.com/Hassaan-Elahi/Writing-Styles-Classification-Using-Stylometric-Analysis

import collections as coll
import math
import re
import time
import scipy as sc
import numpy as np
from matplotlib import style
import spacy
import rusyllab
import pylabeador
from transformers import pipeline
import esupar


class StylometricAnalyzer:
    def __init__(self, excerpt, language) -> None:
        start = time.time()
        self.count_syllable_fail = 0 # syllabification fail counts
        self.excerpt = excerpt.replace("\n", " ").replace("\0", "").replace('\r', '')  # caused issues with regex
        self.excerpt = re.sub(r'\b-\b', ' ', self.excerpt)
        self.excerpt = re.sub(r'\b.\b', ' ', self.excerpt)
        self.excerpt = re.sub(r'\b;\b', ' ', self.excerpt)
        self.excerpt = re.sub(r'\b\[\b', ' ', self.excerpt)

        self.language = language

        # storing so no need to run multiple parses
        self.lemmas_with_punc = self.parse_tokens()

        # strip punctuation
        punc = r',.\'!¡";\?¿:;—()-'

        if self.language == 'ru':
            # strip russian punctuation
            punc += r'«»'
            # only keep characters in Russian alphabet
            self.lemmas = list(filter(None, [lemma.strip(punc) for lemma in self.lemmas_with_punc if re.match(r'[а-яА-Я]', lemma)]))
            # Exclude punctuation from words
            self.words = list(filter(None, [word.replace(punc, '') for word in self.parse_words() if re.match(r'[а-яА-Я]', word)]))

        if self.language == 'es':
            # only keep characters in Spanish alphabet
            self.lemmas = list(filter(None, [lemma.strip(punc) for lemma in self.lemmas_with_punc if re.match(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]', lemma)]))
            self.words = list(filter(None, [word.replace(punc, '') for word in self.parse_words() if re.match(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]', word)]))

        if self.language == 'ko':
            # only keep characters in Korean alphabet
            self.lemmas = list(filter(None, [lemma.strip(punc) for lemma in self.lemmas_with_punc if re.match(r'[ㄱ-ㅎㅏ-ㅣ가-힣]', lemma)]))
            self.words = list(filter(None, [word.replace(punc, '') for word in self.parse_words() if re.match(r'[ㄱ-ㅎㅏ-ㅣ가-힣]', word)]))

        # TODO : Print lemmas and words to see if they are correct

        self.sentences = self.parse_sentences()
        self.pos_tags = self.generate_pos_tags()
        
        # print(f'constructor run in {time.time() - start} s.')

    
    # returns a list of sentences of the excerpt, each of which is a list of individual words
    def parse_sentences(self) -> list[list[str]]:
        # split sentences
        sentences = re.split(r'[.!?]', self.excerpt)

        # parse words in sentences, strip leading/trailing non-alphanumeric characters
        sentences = [list(filter(None, [word.strip(r',.\'!¡";\?¿:;—-') for word in sentence.split(' ')])) for sentence in sentences]

        if sentences[-1] == []: # remove empty sentence at end
            sentences.pop()
        
        return sentences
    
    # returns a list of words of the excerpt
    def parse_words(self) -> list[str]:
        lemmatize_model = spacy.load(f"{self.language}_core_news_sm")

        document = lemmatize_model(self.excerpt)
        lemmatized_tokens = [token.text for token in document if token.text != ' ']
        
        return lemmatized_tokens
    
    # returns a list of lemmatized tokens of the excerpt
    def parse_tokens(self) -> list[str]:
        lemmatize_model = spacy.load(f"{self.language}_core_news_sm")

        document = lemmatize_model(self.excerpt)
        lemmatized_tokens = [token.lemma_ for token in document if token.lemma_ != ' ']
        
        return lemmatized_tokens
    
    # generates a list of POS tag count for 4 categories: nouns, verbs, adjectives, adverbs
    def generate_pos_tags(self):
        start = time.time()
        if self.language == 'ru':
            # https://huggingface.co/KoichiYasuoka/bert-base-russian-upos
            model = esupar.load("KoichiYasuoka/bert-base-russian-upos")
        
        if self.language == 'es':
            # https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-capitel-pos
            model = pipeline("token-classification", model="PlanTL-GOB-ES/roberta-large-bne-capitel-pos")
        
        if self.language == 'ko':
            # https://github.com/KoichiYasuoka/esupar
            model = esupar.load("ko")
        
        pos_results = model(self.excerpt)

        if self.language == 'es':
            pos_tags = {
                'nouns': len([word for word in pos_results if word['entity'] == 'NOUN']),
                'verbs': len([word for word in pos_results if word['entity'] == 'VERB']),
                'adjectives': len([word for word in pos_results if word['entity'] == 'ADJ']),
                'adverbs': len([word for word in pos_results if word['entity'] == 'ADV']),
            }
        
        else:
            tag_values = pos_results.values[3]
            pos_tags = {
                'nouns': len([tag for tag in tag_values if tag == 'NOUN']),
                'verbs': len([tag for tag in tag_values if tag == 'VERB']),
                'adjectives': len([tag for tag in tag_values if tag == 'ADJ']),
                'adverbs': len([tag for tag in tag_values if tag == 'ADV']),
            }

        # print(f'POS tags generated in {time.time() - start} s.')
        return pos_tags


    # Type 1: Basic sentence structure features

    # Feature 1: sentence length (in words) (sentence structure)
    def get_average_sentence_length_words(self) -> float:
        return np.average([len(sent) for sent in self.sentences])
    
    # Feature 2: sentence length (in characters) (sentence structure)
    def get_average_sentence_length_chars(self) -> float:
        return np.average([sum([len(word) for word in sent]) for sent in self.sentences])
    
    # Feature 3: punctuation ratio (% of chars that are punctuation) (sentence structure)
    def get_punctuation_ratio(self) -> float:
        punc = [",", ".", "'", "!", '"', ";", "?", "¿", ":", ";", "—"]
        count = len([c for c in self.lemmas_with_punc if c in punc])
        return count / len(self.excerpt)
    

    # Type 2: Lexical vocabulary richness features

    # Feature 4: hapax legemena ratio (% of unique words that appear only once)
    def get_hapax_legemena_ratio(self) -> float:
        words = self.lemmas
        freqs = coll.Counter()
        freqs.update(words)
        V1 = 0 # number of words that appear only once
        for word in freqs:
            if freqs[word] == 1:
                V1 += 1
        V = len(set(words)) # number of unique words (vocab size)
        return V1 / V
    
    # Feature 5: Honore's statistic (lexical richness)
    def get_honore_statistic(self) -> float:
        hl_ratio = self.get_hapax_legemena_ratio()
        N = len(self.lemmas) # total word count
        honore = 100 * math.log(N / (1 - hl_ratio + 0.0001))
        return honore
    
    # Feature 6: Average word frequency class (connected to Zipf's law)
    def get_average_word_frequency_class(self) -> float:
        words = self.lemmas
        freqs = coll.Counter()
        freqs.update(words)
        maximum = float(max(list(freqs.values())))
        return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])
    
    # Feature 7: Type-token ratio (unique words / total words)
    def get_type_token_ratio(self) -> float:
        words = self.lemmas
        return len(set(words)) / len(words)
    
    # Feature 8: Brunet's index (lexical richness)
    def get_brunet_index(self) -> float:
        words = self.lemmas
        a = -0.165
        V = float(len(set(words)))
        N = len(words)
        brunet = N ** (V ** a)
        return brunet
    
    # Feature 9: Yule's characteristic K (text constancy measure)
    def get_yules_characteristic_k(self) -> float: 
        words = self.lemmas
        N = len(words)
        freqs = coll.Counter()
        freqs.update(words)
        vi = coll.Counter()
        vi.update(freqs.values())
        M = sum([(value * value) * vi[value] for key, value in freqs.items()])
        yule = 10000 * (M - N) / math.pow(N, 2)
        return yule
    
    # Feature 10: Simpson's index (lexical diversity)
    def get_simpsons_index(self) -> float:
        words = self.lemmas
        N = len(words)
        freqs = coll.Counter()
        freqs.update(words)
        n = sum([i * (i - 1) for i in freqs.values()])
        simpsons = 1 - (n / (N * (N - 1)))
        return simpsons
    
    # Feature 11: Shannon's entropy (lexical diversity)
    def get_shannons_entropy(self) -> float:
        words = self.lemmas
        N = len(words)
        freqs = coll.Counter()
        freqs.update(words)
        arr = np.array(list(freqs.values()))
        distribution = 1. * arr
        distribution /= max(1, N)
        shannon = sc.stats.entropy(distribution, base=2)
        return shannon
    

    # Type 3: Readability features

    def count_syllables(self, word) -> int:
        try:
            if self.language == 'ru':
                return len(rusyllab.split_word(word))
            if self.language == 'es':
                return len(pylabeador.syllabify(word))
            if self.language == 'ko':
                return len(word) # korean characters represent syllables
        except:
            self.count_syllable_fail += 1 # increase by 1 if the syllable count fails
            return 1 # if syllable count fails, return 1 as a default
    
    # Feature 12: Average number of syllables per word (readability)
    def get_average_syllables_per_word(self) -> float:
        words = self.words
        print(f"self.words: {words}")
        return np.average([self.count_syllables(word) for word in words])
    
    # Feature 13: Flesch reading ease (readability)
    def get_flesch_reading_ease(self) -> float:
        words = self.words
        sentence_count = len(self.sentences)
        syllable_count = 0
        for word in words:
            syllable_count += self.count_syllables(word)
        l = len(words)
        return 206.835 - 1.015 * (l / float(sentence_count)) - 84.6 * (syllable_count / float(l))
    
    # Feature 14: Flesch-Kincaid grade level (readability)
    def get_flesch_kincaid_grade_level(self) -> float:
        words = self.words
        sentence_count = len(self.sentences)
        syllable_count = 0
        for word in words:
            syllable_count += self.count_syllables(word)
        l = len(words)
        return 0.39 * (l / sentence_count) + 11.8 * (syllable_count / float(l)) - 15.59
    
    # Feature 15: Gunning Fog Index (readability)
    def get_gunning_fog_index(self) -> float:
        words = self.words
        sentence_count = len(self.sentences)
        word_count = float(len(words))
        complex_words = 0
        for word in words:
            if (self.count_syllables(word) > 2):
                complex_words += 1
        return 0.4 * ((word_count / sentence_count) + 100 * (complex_words / word_count))
    

    # Type 4: Syntactic/grammar features

    # Feature 16: Noun-to-verb ratio (part of speech/syntax)
    def get_noun_verb_ratio(self) -> float:
        noun_count = self.pos_tags['nouns']
        verb_count = self.pos_tags['verbs']
        return noun_count / verb_count
    
    # Feature 17: Verb density (part of speech/syntax)
    def get_verb_density(self) -> float:
        word_count = len(self.lemmas)
        verb_count = self.pos_tags['verbs']
        return verb_count / word_count
    
    # Feature 18: Noun density (part of speech/syntax)
    def get_noun_density(self) -> float:
        word_count = len(self.lemmas)
        noun_count = self.pos_tags['nouns']
        return noun_count / word_count
    
    # Feature 19: Adjective density (part of speech/syntax)
    def get_adjective_density(self) -> float:
        word_count = len(self.lemmas)
        adjective_count = self.pos_tags['adjectives']
        return adjective_count / word_count
    
    # Feature 20: Adverb density (part of speech/syntax)
    def get_adverb_density(self) -> float:
        word_count = len(self.lemmas)
        adverb_count = self.pos_tags['adverbs']
        return adverb_count / word_count
    

    # Return entire feature vector
    def get_feature_vector(self) -> list[float]:
        start = time.time()
        fv = [
            self.get_average_sentence_length_words(),
            self.get_average_sentence_length_chars(),
            self.get_punctuation_ratio(),
            self.get_hapax_legemena_ratio(),
            self.get_honore_statistic(),
            self.get_average_word_frequency_class(),
            self.get_type_token_ratio(),
            self.get_brunet_index(),
            self.get_yules_characteristic_k(),
            self.get_simpsons_index(),
            self.get_shannons_entropy(),
            self.get_average_syllables_per_word(),
            self.get_flesch_reading_ease(),
            self.get_flesch_kincaid_grade_level(),
            self.get_gunning_fog_index(),
            self.get_noun_verb_ratio(),
            self.get_verb_density(),
            self.get_noun_density(),
            self.get_adjective_density(),
            self.get_adverb_density()
        ]

        # print(f'Feature vector generated in {time.time() - start} s.')
        return fv
