import numpy as np

def to_one_hot(char2id, chars):
    n_chars = len(char2id)
    one_hot = np.eye(n_chars)[chars.reshape(-1)]
    return one_hot

def one_hot_to_char(id2char, one_hot):
    return [id2char[np.argmax(j)] for j in one_hot]
