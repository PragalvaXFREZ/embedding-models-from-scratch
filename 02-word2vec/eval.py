"""Sanity-check the word2vec embeddings: nearest neighbours and analogies.

Run word2vec.py first; it writes data/C.npy and data/vocab.txt.
"""
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

C = np.load(os.path.join(DATA_DIR, "C.npy"))
with open(os.path.join(DATA_DIR, "vocab.txt")) as f:
    vocab = f.read().split("\n")

def nearest(word, k=8):
    C_unit = C / np.linalg.norm(C, axis=1, keepdims=True)   # every row to length 1, so dot = cosine
    sims = C_unit @ C_unit[vocab.index(word)]                 # cosine with every word
    best = np.argsort(-sims)[1:k + 1]                          # top k, skipping the word itself
    return [(vocab[j], round(float(sims[j]), 3)) for j in best]

for w in ["france", "king", "three", "computer"]:
    print(w, "->", nearest(w))


def analogy(a, b, c, k=5):
    # "a is to b as c is to ?"  ->  b - a + c
    C_unit = C / np.linalg.norm(C, axis=1, keepdims=True)
    idx = [vocab.index(w) for w in (a, b, c)]
    x = C_unit[idx[1]] - C_unit[idx[0]] + C_unit[idx[2]]

    # Inputs are normalised first so no single word dominates by vector length.
    sims = C_unit @ (x / np.linalg.norm(x))   # cosine of x with every word
    sims[idx] = -1                            # rule out the three input words, as the paper does
    best = np.argsort(-sims)[:k]
    return [(vocab[j], round(float(sims[j]), 3)) for j in best]

tests = [
    ("man", "king", "woman"),        # expect queen
    ("france", "paris", "italy"),    # expect rome
    ("big", "bigger", "small"),      # expect smaller (a syntactic one, Table 1)
    ("two", "four", "three"),        # fun one: what does it do with numbers?
]
for a, b, c in tests:
    try:
        print(f"{a} : {b} :: {c} : ?", analogy(a, b, c))
    except ValueError:                # vocab.index raises this for unknown words
        print(f"{a} : {b} :: {c} : ? -> one of these words isn't in the vocab")
