# eval.py
import numpy as np
from data import word_to_idx, idx_to_word

# latest results
# king, queen: 0.902  
# king, dog:   0.270  
# Should try Google's question words benchmark

E = np.load("embeddings.npy")
E = E / np.linalg.norm(E, axis=1, keepdims=True)  # normalize so dot product = cosine similarity

def sim(a, b):
    return E[word_to_idx[a]] @ E[word_to_idx[b]]

def nearest_neighbors(word, n=8):
    scores = E @ E[word_to_idx[word]]
    top_n = np.argpartition(scores, -(n + 1))[-(n + 1):]  # grab top n+1 unsorted
    top_n = top_n[np.argsort(scores[top_n])[::-1]]        # sort just those
    return [(idx_to_word[i], scores[i]) for i in top_n if idx_to_word[i] != word][:n]

## analogy: "a is to b as c is to ?"
# e.g. man:king :: woman:? → king - man + woman ≈ queen
# works because relationships are encoded as directions in embedding space
def analogy(a, b, c, n=5):
    query = E[word_to_idx[b]] - E[word_to_idx[a]] + E[word_to_idx[c]]
    query /= np.linalg.norm(query)
    scores = E @ query
    ranked = np.argsort(scores)[::-1]
    return [(idx_to_word[i], scores[i]) for i in ranked if idx_to_word[i] not in {a, b, c}][:n]


print("== similarity ==")
print(f"king, queen: {sim('king', 'queen'):.3f}  (should be high)")
print(f"king, dog:   {sim('king', 'dog'):.3f}  (should be low)")
