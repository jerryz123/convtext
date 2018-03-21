import numpy as np
import nltk
from xml.etree import ElementTree
from nltk.corpus import shakespeare
import tensorflow as tf

class Corpus(object):
    def __init__(self):
        self.text = ""
        for f in shakespeare.fileids():
            self.text += shakespeare.raw(f)
        self.text = self.text.replace('\r', '')

    def one_hot_raw(self):
        alphabet = set()
        for char in self.text:
            if char == '\r':
                continue
            alphabet.add(char)

        alphabet = sorted(alphabet)
        char_vals = [alphabet.index(c) for c in self.text]
        sub = char_vals[:4]
        def id_to_char(i):
            return alphabet[i]
        return id_to_char, sub

