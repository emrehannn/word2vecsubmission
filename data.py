# data.py
import numpy as np
from hyperparams import CONTEXT_SIZE, EPOCHS

with open("text8", "r") as f:
    text = f.read()


vocabulary = set(text.split())

vocabularysize = len(vocabulary)
# 253854

word_to_idx = {word: i for i, word in enumerate(vocabulary)}
idx_to_word = {i: word for i, word in enumerate(vocabulary)}

text_idx = [word_to_idx[word] for word in text.split()]

text_idx = np.array(text_idx)

# training pairs
def generate_training_pairs():
    for i in range(CONTEXT_SIZE, len(text_idx) - CONTEXT_SIZE):
        context = np.concatenate((text_idx[i - CONTEXT_SIZE: i], text_idx[i + 1 : i + CONTEXT_SIZE + 1]))
        target = text_idx[i]
        yield target, context

    
