
# OUTDATED! this is the old implementation that uses softmax.
# check model.py for negative sampling implementation. 

import numpy as np
from hyperparams import EMBEDDING_DIM, LEARNING_RATE
from data import vocabularysize


# this is virtually impossible to train with text8 dataset because of the softmax layer, but i coded it anyway.

class Word2VecSoftmax:
    
    def __init__(self):
        self.W_in = np.random.randn(vocabularysize, EMBEDDING_DIM)    # embedding matrices
        self.W_out = np.random.randn(vocabularysize, EMBEDDING_DIM)

    def softmax(self, x):
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def forward_pass(self, context):
        hidden = np.mean(self.W_in[context], axis=0)
        # dot product of hidden layer and target embedding
        score = np.dot(self.W_out, hidden)
        # softmax score of every word in the vocabulary
        probability = self.softmax(score)
        return probability, hidden

    def compute_loss(self, probability, target):
        loss = -np.log(probability[target])
        return loss

    def backward_pass(self, probability, target, hidden, context):
        # step 1  w.r.t scores dL/dS
        one_hot_target = np.zeros(vocabularysize)
        one_hot_target[target] = 1
        d_scores = probability - one_hot_target # how wrong was each probability

        #step 2 w.r.t output layer dL/dW_out = dS x h[j]
        d_W_out = np.outer(d_scores, hidden) 
    
        #step 3 derivatife of loss w.r.t. hidden layer same as above,
        #  but we blame hidden instead of output layer for the error. dL/dh = W_out.T @ d_scores
        d_hidden = np.dot(self.W_out.T , d_scores)

        #step 4 derivatife of loss w.r.t. input layer loss averaged over to hidden/size dL/(dW_in[context])
        self.W_in[context] -= LEARNING_RATE * (d_hidden / len(context))
        
        self.W_out -= d_W_out * LEARNING_RATE




