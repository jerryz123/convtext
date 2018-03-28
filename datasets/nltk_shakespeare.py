import numpy as np
import nltk
from xml.etree import ElementTree
from nltk.corpus import shakespeare
import tensorflow as tf
from utils.encodings import to_one_hot


class Corpus(object):
    def __init__(self):
        self.text = ""
        for f in shakespeare.fileids():
            self.text += shakespeare.raw(f)
        self.text = self.text.replace('\r', '')


        # Generate
        self.id2char, self.char2id, self.chars = self.generate_chars()
        self.vocab_size = len(self.char2id)
        self.one_hot = to_one_hot(self.char2id, self.chars)
        print("Unique characters: ", len(self.char2id))
        print("Total length in characters: ", len(self.chars))


    def generate_chars(self):
        alphabet = set()
        for char in self.text:
            if char == '\r':
                continue
            alphabet.add(char)

        alphabet = sorted(alphabet)
        char_vals = np.array([alphabet.index(c) for c in self.text], dtype=int)
        vocab = {c:i for i,c in enumerate(alphabet)}
        return alphabet, vocab, char_vals

    def create_batches_one_hot(self, batch_size=1, seq_length=1000, vocab_size=0):
        one_hot = self.one_hot
        assert vocab_size == self.vocab_size

        num_batches = one_hot.shape[0] // (batch_size * seq_length)
        new_chars = one_hot[:num_batches*batch_size*seq_length].reshape((batch_size, seq_length*num_batches, -1))
        print(new_chars.shape)
        batches = np.split(new_chars,
                           num_batches,
                           1)

        return batches
