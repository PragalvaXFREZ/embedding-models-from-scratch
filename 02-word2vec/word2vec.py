"""Skip-gram word2vec with hierarchical softmax, from scratch in NumPy.

Follows Mikolov et al. (2013), "Efficient Estimation of Word Representations
in Vector Space". Trains on the first 2M words of text8 and writes the learned
input vectors to data/C.npy (one row per word in data/vocab.txt).
"""
import collections
import heapq
import os
import random
import time
import urllib.request
import zipfile

import numpy as np

# All data lives next to this script, so it runs the same from any directory.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEXT8 = os.path.join(DATA_DIR, "text8")

# ---------------------------------------------------------------------------
# 1. Corpus: download text8 (first run only), count, drop rare words
# ---------------------------------------------------------------------------
if not os.path.exists(TEXT8):
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "text8.zip")
    urllib.request.urlretrieve("http://mattmahoney.net/dc/text8.zip", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)

with open(TEXT8) as f:
    words = f.read().split()

words = words[:2_000_000]           # 2M words keeps training fast on a laptop

MIN_COUNT = 5
counts = collections.Counter(words)
vocab = [w for w, c in counts.most_common() if c >= MIN_COUNT]  # sorted by frequency, id 0 = most common

word2id = {w: i for i, w in enumerate(vocab)}
id2word = vocab                     # id2word[i] gives the word back
freq = [counts[w] for w in vocab]   # freq[i] = count of word i

# Corpus as IDs. Rare words are removed *before* windowing, so words on either
# side of a dropped word become neighbours (the original C code does the same).
corpus = [word2id[w] for w in words if w in word2id]

# ---------------------------------------------------------------------------
# 2. Training pairs (center, context)
# ---------------------------------------------------------------------------
WINDOW = 5
random.seed(0)

def make_pairs(corpus, window):
    pairs = []
    n = len(corpus)
    for i, center in enumerate(corpus):
        # Tricky bit: the window size is resampled per word from 1..window.
        # Close neighbours fall inside almost every sampled window, far ones
        # only in the wide ones, so nearby words are weighted more heavily.
        R = random.randint(1, window)
        start = max(0, i - R)
        end = min(n, i + R + 1)
        for j in range(start, end):
            if j != i:
                pairs.append((center, corpus[j]))
    return pairs

pairs = make_pairs(corpus, WINDOW)

# ---------------------------------------------------------------------------
# 3. Parameters
# ---------------------------------------------------------------------------
D = 100
rng = np.random.default_rng(0)

# C = input vectors (the embeddings we keep); W_out = output vectors for the
# full-softmax baseline below. Small random C, zero W_out, as in the C code.
# NOTE: C is re-initialised before the hierarchical-softmax run in section 5.
# This first draw still matters: it advances rng, so removing it would change
# the final numbers.
C = (rng.random((len(vocab), D)) - 0.5) / D
W_out = np.zeros((len(vocab), D))

# ---------------------------------------------------------------------------
# Baseline (kept for reference, not run): plain SGD with a full softmax.
# Every step scores all V words, so it is slow (~178 s for 20k pairs).
# Hierarchical softmax below replaces this.
# ---------------------------------------------------------------------------
lr = 0.025
N = 20000
# start = time.time()
# running_loss = 0.0
#
# for i in range(N):
#     center, context = pairs[i]
#
#     h = C[center]                                   # input word's vector
#     scores = W_out @ h                              # score every word as a neighbour
#     exp_scores = np.exp(scores - scores.max())      # subtract max for numerical stability
#     probs = exp_scores / exp_scores.sum()           # full softmax over all V words
#
#     running_loss += -np.log(probs[context])
#
#     error = probs.copy()
#     error[context] -= 1                             # dL/dscores = p - y
#
#     grad_h = W_out.T @ error                        # compute BEFORE touching W_out
#     W_out -= lr * np.outer(error, h)
#     C[center] -= lr * grad_h
#
#     if (i + 1) % 2000 == 0:
#         print(i + 1, "avg loss:", running_loss / 2000)
#         running_loss = 0.0
#
# print("seconds:", time.time() - start)

# ---------------------------------------------------------------------------
# 4. Huffman tree over the vocabulary
# ---------------------------------------------------------------------------
# Frequent words end up near the root (short paths), rare ones deep down, so the
# average number of forks per prediction is ~log2(V) or less instead of V.

# Shadows the Counter above: from here on counts[i] = count of vocab[i]
# (same values as freq).
counts = [counts[w] for w in vocab]

V = len(vocab)

# A heap always hands you the smallest item first.
# Each entry is (count, node_id). Leaves are words: ids 0..V-1.
heap = [(counts[i], i) for i in range(V)]
heapq.heapify(heap)

parent = {}      # parent[node] = the fork above it
went_right = {}  # went_right[node] = 1 if it's the right child of its parent, else 0

next_id = V      # forks get new ids starting after the last word
while len(heap) > 1:
    c1, a = heapq.heappop(heap)   # rarest thing left
    c2, b = heapq.heappop(heap)   # second rarest
    parent[a], went_right[a] = next_id, 0
    parent[b], went_right[b] = next_id, 1
    heapq.heappush(heap, (c1 + c2, next_id))   # the new fork counts as their combined frequency
    next_id += 1

# A binary tree with V leaves has exactly V - 1 internal nodes (forks),
# so fork ids run V .. 2V-2 and the last one created is the root.
root = next_id - 1
print("forks:", next_id - V)

def path_to(word_id):
    # Walk UP from the word to the root, recording each fork and which side we came from.
    nodes, turns = [], []
    node = word_id
    while node != root:
        nodes.append(parent[node])        # the fork above us
        turns.append(went_right[node])    # 0 = we were its left child, 1 = right
        node = parent[node]
    return nodes[::-1], turns[::-1]       # flip so it reads root -> word

# ---------------------------------------------------------------------------
# 5. Train with hierarchical softmax
# ---------------------------------------------------------------------------
# One vector per fork, starting at zero like W_out did.
# Fork ids run from V up to root, so fork id f lives in row f - V.
W_fork = np.zeros((V - 1, D))

def sigmoid(x):
    # squashes any number into 0..1, used as "probability of going right"
    return 1 / (1 + np.exp(-x))

# paths[w] = (fork rows along the path, turns along the path), precomputed for every word
paths = []
for w in range(V):
    nodes, turns = path_to(w)
    paths.append((np.array(nodes) - V, np.array(turns)))   # "- V" turns fork ids into W_fork row numbers

C = (rng.random((V, D)) - 0.5) / D   # fresh start, so the comparison with the 178 s baseline is fair
W_fork = np.zeros((V - 1, D))

lr = 0.025
N = len(pairs)
REPORT = 500_000
start = time.time()
running_loss = 0.0

for i in range(N):
    # Linear decay: lr goes from 0.025 to ~0 over the run. The floor
    # (0.025 * 1e-4) keeps it from hitting exactly zero, as in the C code.
    lr = max(0.025 * (1 - i / N), 0.025 * 0.0001)
    center, context = pairs[i]
    rows, turns = paths[context]         # the forks to walk to reach the real neighbour

    h = C[center]                        # a view, not a copy: safe only because C[center] is updated last
    p = sigmoid(W_fork[rows] @ h)        # one "go right?" probability per fork on the path

    # Loss: binary cross-entropy of the correct turn at each fork, summed along the path.
    # P(context | center) is the product of these turn probabilities.
    running_loss += -np.sum(turns * np.log(p) + (1 - turns) * np.log(1 - p))

    error = p - turns                    # same p - t idea, one number per fork

    # Order matters: grad_h must use W_fork *before* it is updated.
    # W_fork[rows] with an index array is a copy, so `-=` writes back correctly;
    # rows never repeat within one path, so no updates are lost.
    grad_h = error @ W_fork[rows]
    W_fork[rows] -= lr * np.outer(error, h)
    C[center] -= lr * grad_h

    if (i + 1) % REPORT == 0:
        print(i + 1, "avg loss:", running_loss / REPORT, "lr:", round(lr, 5))
        running_loss = 0.0

print("seconds:", time.time() - start)

# ---------------------------------------------------------------------------
# 6. Save embeddings for eval.py
# ---------------------------------------------------------------------------
np.save(os.path.join(DATA_DIR, "C.npy"), C)
with open(os.path.join(DATA_DIR, "vocab.txt"), "w") as f:
    f.write("\n".join(vocab))
