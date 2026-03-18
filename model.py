# model.py
import numpy as np
from hyperparams import EMBEDDING_DIM, LEARNING_RATE, CONTEXT_SIZE
from data import vocabularysize


class Word2Vec:
    
    def __init__(self):
        self.W_in = np.random.randn(vocabularysize, EMBEDDING_DIM)    # embedding matrices
        self.W_out = np.random.randn(vocabularysize, EMBEDDING_DIM)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x)) # overflow risk


    # forward pass
    # hidden layer: h = mean of context picked from W_in
    # scores: W_out @ h
    # probabilities: softmax of scores

    def forward_pass(self, context):
        hidden_layer = np.mean(self.W_in[context], axis=0)
        # dot product of hidden layer and target embedding
        score = np.dot(self.W_out, hidden_layer)
        # softmax score of every word in the vocabulary
        probability = self.softmax(score)
        return probability, hidden_layer

    # compute loss

    def compute_loss(self, probability, target):
        loss = -np.log(probability[target])
        return loss


    # backward pass

    #  backwards from forward pass, chain rule. To see how much a weight (w) affected the total error,
    # look at how much the weights affected the score, and then how much that score affected the error

    # dL/dW = dL/dS x dS/dW


    def backward_pass(self, probability, target, hidden_layer, context):
        # step 1  w.r.t scores dL/dS
        one_hot_target = np.zeros(vocabularysize)
        one_hot_target[target] = 1
        d_scores = probability - one_hot_target # how wrong was each probability


        #step 2 w.r.t output layer dL/dW_out = dS x h[j]
        d_W_out = np.outer(d_scores, hidden_layer) 
    

        #step 3 derivatife of loss w.r.t. hidden layer same as above,
        #  but we blame hidden instead of output layer for the error. dL/dh = W_out.T @ d_scores
        d_hidden = np.dot(self.W_out.T , d_scores)

        #step 4 derivatife of loss w.r.t. input layer loss averaged over to hidden/size dL/(dW_in[context])
        self.W_in[context] -= LEARNING_RATE * (d_hidden / CONTEXT_SIZE)
        
        self.W_out -= d_W_out * LEARNING_RATE


# forward notes
# hidden   = mean(W_in[context])
# scores   = W_out @ hidden
# probs    = softmax(scores)
# loss     = -log(probs[target])

# backwards pass going reverse chain rule notes

# step 1 loss to scores
# how wrong was each score?
# d_scores = probs - onehot(target)

# scores = W_out @ hidden        ← one forward line, two things to blame

# first blame W_out:    how much did W_out cause the wrong scores?
# second — blame hidden:   how much did hidden cause the wrong scores?


# step 2 scores to W_out
# how much did each output weight cause that wrongness?
# d_W_out = outer(d_scores, hidden)

# step 3 scores to hidden
# how much did the hidden layer cause that wrongness?
# d_hidden = W_out.T @ d_scores

# step 4 hidden to W_in
# how much did each input word embedding cause that hidden error
# d_W_in = d_hidden / context_size   (divide because forward has a mean of context size)