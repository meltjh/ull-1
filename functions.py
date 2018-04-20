import numpy as np

# Getting the pre-trained word vectors.
def get_word_vectors(filename):
    word_vectors = dict()
    with open(filename) as f:
        for line in f:
            content = line.split()
            word_vectors[content[0]] = np.array(content[1:len(content)]).astype(np.float)
    return word_vectors