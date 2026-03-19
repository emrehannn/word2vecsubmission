#train.py
from model import Word2Vec
from data import generate_training_pairs
from hyperparams import EPOCHS
import numpy as np


model = Word2Vec()


for epoch in range(EPOCHS):
    epoch_loss = 0
    pairs = 0
    for target, context in generate_training_pairs():
        probability, hidden = model.forward_pass(context)
        epoch_loss += model.compute_loss(probability, target)
        model.backward_pass(probability, target, hidden, context)
        pairs += 1
    average_loss = epoch_loss / pairs
    
    print(f"Epoch {epoch} loss: {average_loss}")
    
    

np.save("embeddings.npy", model.W_in)

# STEPS FOR IMPROVEMENT

# 1. negative sampling 
# 2. batching