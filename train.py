# train.py
from model import Word2Vec
from data import get_positive_samples, get_negative_samples, text_idx
from hyperparams import CONTEXT_SIZE, EPOCHS, k, LEARNING_RATE
import numpy as np

model = Word2Vec()

# we count steps so that we can do learning rate decay as training progresses, as per the word2vec paper.
# total steps is the number of weight updates we will do.
total_steps = EPOCHS * (len(text_idx) - 2 * CONTEXT_SIZE) # 2 x context size because we skip the first and last few words that don't have a full context.
step = 0

LOG_EVERY = 100_000

for epoch in range(EPOCHS):
    epoch_loss = 0  # reset loss accumulator for this epoch
    running_loss = 0  # for logging every LOG_EVERY steps
    
    pairs = 0  # reset pair counter for this epoch

    for target, context in get_positive_samples():  # slide window across the text, yielding one (target, context) pair per position


        lr = LEARNING_RATE * (1 - step / total_steps)  # decay lr linearly — starts at LEARNING_RATE, approaches 0 by the last step
        negatives = get_negative_samples(target, k)  # sample k random words from the vocabulary that are not the target

        target_score, negative_scores = model.forward_pass(context, target, negatives)  # compute sigmoid scores for the target and all negatives
        model.backward_pass(target, target_score, negative_scores, negatives, context, lr)  # compute gradients and update W_in and W_out

        loss = model.compute_loss(target_score, negative_scores)  # accumulate the loss for this pair (for logging only, doesn't affect training)
        epoch_loss += loss
        running_loss += loss


        step += 1  # global step counter, used for lr decay
        pairs += 1  # epoch-local counter, used for averaging the loss  

        if step % LOG_EVERY == 0:
            print(f"step {step}/{total_steps} | epoch {epoch} | lr {lr:.6f} | loss {running_loss / LOG_EVERY:.4f}")
            running_loss = 0
    np.save("embeddings.npy", model.W_in) # only save W_in for inference
    print(f"Epoch {epoch} loss: {epoch_loss / pairs}")  # average loss per training pair this epoch


