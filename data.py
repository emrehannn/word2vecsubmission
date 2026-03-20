# data.py
import numpy as np
from hyperparams import CONTEXT_SIZE

with open("text8", "r") as f:
    text = f.read(7000000) # read the first 7 million characters for faster training, the full text8 is 100 million characters


words = text.split()

vocabulary = sorted(set(words))

vocabularysize = len(vocabulary)
# 253854

word_to_idx = {word: i for i, word in enumerate(vocabulary)}
idx_to_word = {i: word for i, word in enumerate(vocabulary)}

text_idx = [word_to_idx[word] for word in words]

text_idx = np.array(text_idx)
 

## positive samples, sliding context over text and yielding target and context pairs. 
# for example, if CONTEXT_SIZE = 2, then for the sentence "the cat sat on the mat", we would yield:
# target: "sat", context: ["the", "cat", "on", "the"]
def get_positive_samples():
    for i in range(CONTEXT_SIZE, len(text_idx) - CONTEXT_SIZE):
        context = np.concatenate((text_idx[i - CONTEXT_SIZE: i], text_idx[i + 1 : i + CONTEXT_SIZE + 1]))
        target = text_idx[i]
        yield target, context

    
## negative sampling

# a negative sample is a random word from the vocabulary that is not the target word.
# we will use these negative samples to train our model to distinguish between true context words and random words.
# for example, if the target word is "sat", and the context words are ["the", "cat", "on", "the"],
# we might sample negative words like ["dog", "house", "tree"] that are not in the context of "sat".

# no. of words occurences
word_counts = np.bincount(text_idx)

noise_dist = word_counts.astype(float) # convert to float for power scaling, otherwise they are integers

noise_dist **= 0.75 # power scaling, as per the word2vec paper to give more weight to less frequent
                    # words and less weight to more frequent words.

noise_dist /= np.sum(noise_dist) # normalize to get probabilities. this is the distribution
                                # we will sample from when generating negative samples.


# negative training pairs
def get_negative_samples(target, k):
    candidates = np.random.choice(vocabularysize, size=k * 3, p=noise_dist)
    candidates = candidates[candidates != target]
    return candidates[:k]
# I draw 3x more candidates than I need so I have room to filter out the target word, then take the first k
# paper does this by picking random words in a loop one word at a time, but they use c. i prefer to batch it

# the paper, on skip-gram, draw k negative samples per context word. 
# In skip-gram, each (target, context_word) pair is a separate training example. So for a window of 2,
# you have 4 individual pairs per target position, and you draw k negatives for each one. Total negatives per position = 4 * k


# on the other hand, with cbow, we only draw k negative samples per target word,
# regardless of the context size, so if k = 4, we would draw 4 negative samples per target word.


# paper also discards some very frequent words, 
# P(discard) = 1 - sqrt(t / f(w)) f(w) is the frequency of the word, and t is a chosen threshold, typically around 1e-5.
# this is not implemented here.