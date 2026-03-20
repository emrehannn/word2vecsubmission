# model.py
import numpy as np
from hyperparams import EMBEDDING_DIM
from data import vocabularysize


# this is a cbow implementation, where we take the mean
# of the context word vectors to predict the target word.

class Word2Vec:
    def __init__(self):
        # input embeddings (also the final word vectors we care about)
        self.W_in = (np.random.rand(vocabularysize, EMBEDDING_DIM) - 0.5) / EMBEDDING_DIM #  as per the word2vec paper initialized with substracting 0,5 to get the values between -0,5 and 0,5 
        # (good for sigmoid and training in the first epochs) also dividing with embedding dim to make the numbers small enough. 
        
        # output embeddings 
        self.W_out = np.zeros((vocabularysize, EMBEDDING_DIM))   # paper initializes them as zero, otherwise they are beyond sigmoids saturation


    def sigmoid(self, x):
        x = np.clip(x, -10, 10) # clip to avoid overflow in exp, otherwise we get infs and nans that break training. this is a common trick when implementing sigmoid.
                                # this was a big issue, embeddings were just bad before this fix.
        return 1 / (1 + np.exp(-x))


    def forward_pass(self, context, target, negatives):

        # mean the context vectors into one in the Y axis. (2*context , embeddingdim) to (embeddingdim, )
        self.hidden = np.mean(self.W_in[context], axis = 0)

        # dot output matrix with hidden dimension values to get the score, both are (100, )
        target_score = self.sigmoid(np.dot(self.W_out[target], self.hidden))

        # its the same, but there are k negative scores and shape is thus (k, )
        negative_scores = self.sigmoid(self.W_out[negatives] @ self.hidden)

        return target_score, negative_scores


    def compute_loss(self, target_score, negative_scores):
        # L = -log(target_score) - sum(log(1 - negative_scores))
        return -np.log(target_score) - np.sum(np.log(1 - negative_scores))
        # -positive signal - negative signal



    def backward_pass(self, target, target_score, negative_scores, negatives, context, lr):

        
        # positive sample gradients
        target_error = target_score - 1 # dL/d_score for positive sample (gradient of -log(sigmoid(score)))
        d_W_out = target_error * self.hidden # dL/dW_out = dL/dscore * dscore/dW_out = target_error * hidden
        
        # negative sample gradients
        negative_error = negative_scores 
        d_W_out_negatives = np.outer(negative_error, self.hidden) # only outer product gives (k, EMBEDDING_DIM) dL/dW_out for negatives = dL/dscore * dscore/dW_out

        # hidden layer gradients
        d_hidden = target_error * self.W_out[target] + negative_error @ self.W_out[negatives]

        # update weights
        self.W_out[target] -= lr * d_W_out # update the target word's output vector
        self.W_out[negatives] -= lr  * d_W_out_negatives
        self.W_in[context] -= lr * (d_hidden / len(context))


