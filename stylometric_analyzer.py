# taken from https://github.com/Hassaan-Elahi/Writing-Styles-Classification-Using-Stylometric-Analysis

import collections as coll
import math
import pickle
import string
import scipy as sc

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import nltk
import spacy
from konlpy.tag import Mecab
import rusyllab
import pylabeador

nltk.download('cmudict')
nltk.download('stopwords')

style.use("ggplot")
cmuDictionary = None

class StylometricAnalyzer:
    def __init__(self, excerpt, language, pos_tags) -> None:
        self.excerpt = excerpt
        self.language = language
        self.pos_tags = pos_tags
    
    # returns a list of sentences of the excerpt, each of which is a list of individual words
    def parse_sentences(self) -> list[list[str]]:
        sentences = self.excerpt.split('.!?')

        # rough parse, based purely on spaces
        sentences = [sentence.split() for sentence in sentences]

        if sentences[-1] == ['']: # remove empty sentence at end
            sentences.pop()
        
        return sentences
    
    # returns a list of lemmatized tokens of the excerpt
    def parse_tokens(self) -> list[str]:

        if self.language == 'ko':
            mecab = Mecab()
            tokens = mecab.pos(text)
            
            lemmatized_tokens = []
            for token, pos in tokens:
                lemmatized_tokens.append(token)

        else:
            if self.language == 'es':
                lemmatize_model = spacy.load("es_core_news_sm")

            if self.language == 'ru':
                lemmatize_model = spacy.load("ru_core_news_sm")
            
            document = lemmatize_model(self.excerpt)
            lemmatized_tokens = [token.lemma_ for token in document]
        
        return lemmatized_tokens
    
    # Feature 1: sentence length (in words) (sentence structure)
    def get_average_sentence_length_words(self) -> float:
        print(self.parse_sentences())
        return np.average([len(sent) for sent in self.parse_sentences()])
    
    # Feature 2: sentence length (in characters) (sentence structure)
    def get_average_sentence_length_chars(self) -> float:
        return np.average([sum([len(word) for word in sent]) for sent in self.parse_sentences()])
    
    # Feature 3: punctuation ratio (% of chars that are punctuation) (sentence structure)
    def get_punctuation_ratio(self) -> float:
        punc = [",", ".", "'", "!", '"', ";", "?", ":", ";"]
        count = 0
        for c in self.excerpt:
            if c in punc:
                count += 1
        return count / len(self.excerpt)
    
    # Feature 4: hapax legemena ratio (% of unique words that appear only once)
    def get_hapax_legemena_ratio(self) -> float:
        words = self.parse_tokens()
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
        N = len(self.parse_tokens()) # total word count
        honore = 100 * math.log(N / (1 - hl_ratio + 0.0001))
        return honore
    
    # Feature 6: Average word frequency class (connected to Zipf's law)
    def get_average_word_frequency_class(self) -> float:
        words = self.parse_tokens()
        freqs = coll.Counter()
        freqs.update(words)
        maximum = float(max(list(freqs.values())))
        return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])
    
    # Feature 7: Type-token ratio (unique words / total words)
    def get_type_token_ratio(self) -> float:
        words = self.parse_tokens()
        return len(set(words)) / len(words)
    
    # Feature 8: Brunet's index (lexical richness)
    def get_brunet_index(self) -> float:
        words = self.parse_tokens()
        a = -0.165
        V = float(len(set(words)))
        N = len(words)
        brunet = N ** (V ** a)
        return brunet
    
    # Feature 9: Yule's characteristic K (text constancy measure)
    def get_yules_characteristic_k(self) -> float: 
        words = self.parse_tokens()
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
        words = self.parse_tokens()
        N = len(words)
        freqs = coll.Counter()
        freqs.update(words)
        n = sum([i * (i - 1) for i in freqs.values()])
        simpsons = 1 - (n / (N * (N - 1)))
        return simpsons
    
    # Feature 11: Shannon's entropy (lexical diversity)
    def get_shannons_entropy(self) -> float:
        words = self.parse_tokens()
        N = len(words)
        freqs = coll.Counter()
        freqs.update(words)
        arr = np.array(list(freqs.values()))
        distribution = 1. * arr
        distribution /= max(1, N)
        shannon = sc.stats.entropy(distribution, base=2)
        return shannon
    
    def count_syllables(self, word) -> int:
        if self.language == 'ru':
            return len(rusyllab.split_word(word))
        if self.language == 'es':
            return len(pylabeador.syllabify(word))
        if self.language == 'ko':
            return len(word) # korean characters represent syllables
    
    # Feature 12: Average number of syllables per word (readability)
    def get_average_syllables_per_word(self) -> float:
        words = self.parse_tokens()
        return np.average([self.count_syllables(word) for word in words])
    
    # Feature 13: Flesch reading ease (readability)
    def get_flesch_reading_ease(self) -> float:
        words = self.parse_tokens()
        sentence_count = len(self.parse_sentences())
        syllable_count = 0
        for word in words:
            syllable_count += self.count_syllables(word)
        l = len(words)
        return 206.835 - 1.015 * (l / float(sentence_count)) - 84.6 * (syllable_count / float(l))
    
    # Feature 14: Flesch-Kincaid grade level (readability)
    def get_flesch_kincaid_grade_level(self) -> float:
        words = self.parse_tokens()
        sentence_count = len(self.parse_sentences())
        syllable_count = 0
        for word in words:
            syllable_count += self.count_syllables(word)
        l = len(words)
        return 0.39 * (l / sentence_count) + 11.8 * (syllable_count / float(l)) - 15.59
    
    # Feature 15: Gunning Fog Index (readability)
    def get_gunning_fog_index(self) -> float:
        words = self.parse_tokens()
        sentence_count = len(self.parse_sentences())
        word_count = float(len(words))
        complex_words = 0
        for word in words:
            if (self.count_syllables(word) > 2):
                complex_words += 1
        return 0.4 * ((word_count / sentence_count) + 100 * (complex_words / word_count))
    
    # Feature 16: Noun-to-verb ratio (part of speech/syntax)
    def get_noun_verb_ratio(self) -> float:
        noun_count = self.pos_tags['nouns']
        verb_count = self.pos_tags['verbs']
        return noun_count / verb_count
    
    # Feature 17: Verb density (part of speech/syntax)
    def get_verb_density(self) -> float:
        word_count = len(self.parse_tokens())
        verb_count = self.pos_tags['verbs']
        return verb_count / word_count
    
    # Feature 18: Noun density (part of speech/syntax)
    def get_noun_density(self) -> float:
        word_count = len(self.parse_tokens())
        noun_count = self.pos_tags['nouns']
        return noun_count / word_count
    
    # Feature 19: Adjective density (part of speech/syntax)
    def get_adjective_density(self) -> float:
        word_count = len(self.parse_tokens())
        adjective_count = self.pos_tags['adjectives']
        return adjective_count / word_count
    
    # Feature 20: Adverb density (part of speech/syntax)
    def get_adverb_density(self) -> float:
        word_count = len(self.parse_tokens())
        adverb_count = self.pos_tags['adverbs']
        return adverb_count / word_count
    
    def get_feature_vector(self) -> list[float]:
        return [
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