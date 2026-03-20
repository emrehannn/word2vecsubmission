# word2vec-numpy

Word2vec is an embedding model developed by Google. This is a base level implementation of CBOW only using numpy and text8 dataset by Matt Mahoney.

The model.py file implements strategies used in the 2nd word2vec paper, like negative sampling, power scaling, and others. It is worth noting that the 2nd Word2Vec paper actually uses Skip-gram (as it is actually a better embedding architecture), but I wanted to implement CBOW with negative sampling so it is different enough, and also combines information from both papers.

The model_softmax.py file implements just word2vec with barebones forward and backward pass using softmax. I coded this before reading through the paper to see how a blind approach would perform. current data.py and train.py is not fitting for it anymore, but it is left here as a comparison for the derivatives. If you are wondering, its practically impossible to train word2vec like this, because softmax requires a 253k dimensional dot product and then the sum on every step. First paper actually tries to implement a variation of softmax called hierarchical softmax, but it is much more complex to implement than the negative sampling introduced (and implemented in this repo) in the 2nd paper.

---

## Data Pipeline

### Vocabulary
1. Data.py reads the text8 dataset.
2. Define the vocabulary; sort, remove duplicates, separate by words (sorting is important for consistency, we get same indices every run),
3. Define word to index and index to word mappings for models input and output mappings

### Positive Sampling
1. Slide over the entire data with a window,
2. Capture context words by concatenating words in range len(context_size) around the target word in "context",
3. Capture every word that has enough context words around it as a possible target word
4. Ensure checks for start of the data and end of the data is off by (context_size) to let every target have enough context.
5. Yield context and target

### Negative Sampling
1. Count number of occurences of each word in our vocabulary in the dataset in word_count (np.bincount)
2. Create a probability distribution, turn each integer into a float,
3. Scale each float to power of 3/4, closing the gap a little between rare and very common words,
4. Divide each float to sum of the scaled counts,
5. Draw k * 3 candidates in one vectorized np.random.choice call using the scaled distribution, instead of sampling one by one in a loop. We draw 3x more than we need so we have room to filter. The original paper does this one at a time in a plain loop, but that is fast in C — in Python the loop overhead is too costly.
6. Filter out the target word using boolean indexing, take the first k remaining.
7. Randomly choosing words based on their distribution makes sure there is no bias, they are as randomly picked as possible. Thats the negative signal that we want to avoid.

---

## Model Architecture

### Core
1. Initialize input and output weights, output weights are initialized as zero to avoid sigmoids vanishing gradient.
2. Standard Sigmoid

### Forward Pass
1. mean of all context words is our hidden vector, np.mean(W_in[context]) = hidden
2. target_score = W_out[target] x hidden
3. target_probabilities = sigmoid(scores)
4. negative_score = W_out[negatives] x hidden // the shape will be (k, ) because we have k elements in negatives
5. return target_score, negative_score

### Compute Loss
1. basically the same with k-dimensional cross entropy loss in pytorch.
2. How well model evaluates positive samples, how well model evaluates negative samples?

### Backward Pass
Wnew = Wold - learning rate * gradient

1. target_error = target_score - 1, gradient of -log(sigmoid(x)), when model is perfect target_score = 1 and error is 0 so how wrong was the positive prediction? perfect prediction means no update
2. negative_error = negative_scores, gradient of -log(1 - sigmoid(x)), when model is perfect negative_scores = 0 and error is 0, how wrong were the negative predictions? perfect prediction means no update again
3. update W_out[target] by target_error * hidden // W_out[target] needs to align more with hidden when we are wrong
4. update W_out[negatives] with outer product of negative_error and hidden, giving (k, EMBEDDING_DIM). needs to push away from hidden, one row per negative word since each has its own gradient
5. d_hidden accumulates error from both positive and negative output vectors back into the hidden state, individually calculate and add them up. Dont forget that there are k negatives
6. distribute d_hidden equally across all context vectors in W_in[context], dividing by len(context) to mirror the mean pooling in the forward pass
7. Nudge weights to the opposite of gradient

Positive output weight backprop:
dL/dW+ = dL/dSigmoid+ x dSigmoid+/dS+ x dS+/dW+ = ((sigmoid+) - 1) x h

Negative output weight backprop:
dL/dW- = dL/dSigmoid- x dSigmoid-/dS- x dS-/dW- = (sigmoid-) x h

Hidden error backprop:
dL/dH = (positive blame for hidden, similar to positive output weight backprop) + sum(negative blame for hidden, similar to negative output weight backprop)

Input weights:
dL/dW_in = dL/dH x 1/n, because hidden is made up of a mean of n vectors.

---

## Training Pipeline

### Model
1. Initialize model with chosen hyperparams.
2. Calculate the total steps the training will take. This is useful because the paper utilizes a linear learning rate with decay. epoch count * length of the data - 2x context size (clipping the beginning and the end)
3. for the chosen epoch count, loop over the text;

### Loop
1. Update learning rate,
2. Get positive and negative samples,
3. Forward pass, Backward Pass
4. Compute per epoch loss just for tracking progress.

And finally, save the model for evals. The barebones evals implementation gives these scores: 

king, queen: 0.902
king, dog:   0.270

---

## Notes
1. Subsampling frequent words (P(discard) = 1 - sqrt(t / f(w))) is not implemented.
2. Batching could improve performance 10x to 100x, but it requires a massive refactor.
3. Should try Google's question words benchmark.

## Requirements

This implementation only uses numpy.

## Data
Download the text8 dataset from Matt Mahoney's site:
http://mattmahoney.net/dc/text8.zip

Unzip it and place `text8` in the project root before training.

## References

- Mikolov et al. (2013) — *Efficient Estimation of Word Representations in Vector Space* https://arxiv.org/abs/1301.3781
- Mikolov et al. (2013) — *Distributed Representations of Words and Phrases and their Compositionality* https://arxiv.org/abs/1310.4546
