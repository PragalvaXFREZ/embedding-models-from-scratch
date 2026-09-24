# Experiment 2 for the "Observing the Probabilities" blog.
# Does a bigger corpus fix the embeddings? Compares repetition (more text, nothing new)
# against a corpus with more words and more combinations, with and without weight decay.
import itertools
import math
import torch
import torch.nn.functional as F

N = 3  # context of n-1 = 2 words


def examples(sentences, w2i):
    # One long token stream, like the original corpus, with "." ending each sentence.
    tokens = [w for s in sentences for w in s.split()]
    X = [[w2i[w] for w in tokens[i - (N - 1):i]] for i in range(N - 1, len(tokens))]
    Y = [w2i[w] for w in tokens[N - 1:]]
    return torch.tensor(X), torch.tensor(Y)


def train(X, Y, V, seed, steps=3000, lr=0.1, weight_decay=0.0, m=5, h=8):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn(V, m, generator=g, requires_grad=True)
    H = torch.randn((N - 1) * m, h, generator=g, requires_grad=True)
    d = torch.randn(h, generator=g, requires_grad=True)
    U = torch.randn(h, V, generator=g, requires_grad=True)
    W = torch.randn((N - 1) * m, V, generator=g, requires_grad=True)
    b = torch.randn(V, generator=g, requires_grad=True)
    params = [C, H, d, U, W, b]

    def model(X):
        x = C[X].view(X.shape[0], -1)
        return b + x @ W + torch.tanh(x @ H + d) @ U

    for _ in range(steps):
        # like the paper, the penalty is on the weights and C, not on the biases d and b
        loss = F.cross_entropy(model(X), Y) + weight_decay * sum((p ** 2).sum() for p in (C, H, U, W))
        for p in params:
            p.grad = None
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad
    return model, C.detach(), sum(p.numel() for p in params)


def cos(E, w2i, a, c):
    return F.cosine_similarity(E[w2i[a]], E[w2i[c]], dim=0).item()


def mean(xs):
    return sum(xs) / len(xs)


SEEDS = range(10)

# ---------------------------------------------------------------- A. repetition
print("=" * 78)
print("A. REPETITION: the original corpus, repeated 50 times")
print("=" * 78)
orig = ["the cat sat on the mat", "the dog sat on the rug"]
vocab = sorted({w for s in orig for w in s.split()})
w2i = {w: i for i, w in enumerate(vocab)}

X1, Y1 = examples(orig, w2i)
X50, Y50 = X1.repeat(50, 1), Y1.repeat(50)
print(f"1x : {len(Y1)} examples, {len(set(zip(map(tuple, X1.tolist()), Y1.tolist())))} distinct")
print(f"50x: {len(Y50)} examples, {len(set(zip(map(tuple, X50.tolist()), Y50.tolist())))} distinct")

_, E1, _ = train(X1, Y1, len(vocab), seed=42, steps=1000)
_, E50, _ = train(X50, Y50, len(vocab), seed=42, steps=1000)
print(f"largest difference between the two trained C matrices: {(E1 - E50).abs().max().item():.2e}")

# The same thing done carelessly: gluing the text to itself 50 times as one string.
X50s, Y50s = examples(orig * 50, w2i)
new = {(tuple(x), y) for x, y in zip(X50s.tolist(), Y50s.tolist())} - \
      {(tuple(x), y) for x, y in zip(X1.tolist(), Y1.tolist())}
print("\ncontexts that only exist because the string was glued to itself:")
for x, y in sorted(new):
    print(f"   ({vocab[x[0]]}, {vocab[x[1]]}) -> {vocab[y]}")

print(f"\nmean over {len(SEEDS)} seeds, 1000 steps")
print(f"{'setting':>20} | cat-dog | mat-rug")
for name, (X, Y) in [("1x", (X1, Y1)), ("50x, examples copied", (X50, Y50)), ("50x, string glued", (X50s, Y50s))]:
    Es = [train(X, Y, len(vocab), seed=s, steps=1000)[1] for s in SEEDS]
    print(f"{name:>20} | {mean([cos(E, w2i, 'cat', 'dog') for E in Es]):7.2f} | "
          f"{mean([cos(E, w2i, 'mat', 'rug') for E in Es]):7.2f}")

# ---------------------------------------------------------------- B. a real bigger corpus
print("\n" + "=" * 78)
print("B. A BIGGER CORPUS: more words, more combinations, some sentences held out")
print("=" * 78)
animals = ["cat", "dog", "fox", "cow"]
verbs = ["sat", "lay", "slept"]
things = ["mat", "rug", "sofa", "bed"]
sentences = [f"the {a} {v} on the {t} ." for a, v, t in itertools.product(animals, verbs, things)]

# Hold out every sentence where the fox slept or the cow lay. The held-out loss then measures
# how the model does on sentences it never saw, which is what overfitting hurts.
held = [s for s in sentences if "fox slept" in s or "cow lay" in s]
seen = [s for s in sentences if s not in held]
g = torch.Generator().manual_seed(0)
seen = [seen[i] for i in torch.randperm(len(seen), generator=g)]  # shuffle sentence order

vocab = sorted({w for s in sentences for w in s.split()})
w2i = {w: i for i, w in enumerate(vocab)}
Xtr, Ytr = examples(seen, w2i)
Xva, Yva = examples(held, w2i)
print(f"vocabulary: {len(vocab)} words  {vocab}")
print(f"training: {len(seen)} sentences, {len(Ytr)} examples, "
      f"{len(set(zip(map(tuple, Xtr.tolist()), Ytr.tolist())))} distinct")
print(f"held out: {len(held)} sentences, {len(Yva)} examples")

# floor: the entropy of the training data itself, since identical contexts have different answers
counts = {}
for x, y in zip(map(tuple, Xtr.tolist()), Ytr.tolist()):
    counts.setdefault(x, {}).setdefault(y, 0)
    counts[x][y] += 1
floor = -sum(c * math.log(c / sum(ys.values())) for ys in counts.values() for c in ys.values()) / len(Ytr)
print(f"lowest possible training loss (entropy of the data): {floor:.3f}")


def group_sim(E, A, B):
    return mean([cos(E, w2i, a, c) for a in A for c in B if a != c])


rows = [("no weight decay", 0.0, 5, 8), ("smaller model", 0.0, 2, 4),
        ("weight decay 0.001", 0.001, 5, 8), ("weight decay 0.01", 0.01, 5, 8)]
results = []
for name, wd, m, h in rows:
    stats = []
    for s in SEEDS:
        model, E, n_params = train(Xtr, Ytr, len(vocab), seed=s, weight_decay=wd, m=m, h=h)
        with torch.no_grad():
            tr = F.cross_entropy(model(Xtr), Ytr).item()
            va = F.cross_entropy(model(Xva), Yva).item()
        stats.append([tr, va, cos(E, w2i, "cat", "dog"), cos(E, w2i, "mat", "rug"),
                      group_sim(E, animals, animals), group_sim(E, things, things),
                      group_sim(E, animals, things)])
    results.append((name, n_params, [mean(c) for c in zip(*stats)]))

print(f"\nmean over {len(SEEDS)} seeds, 3000 steps")
print(f"{'setting':>18} | params | train loss | held-out loss")
for name, n_params, col in results:
    print(f"{name:>18} | {n_params:6d} | {col[0]:10.3f} | {col[1]:13.3f}")
print(f"\n{'setting':>18} | cat-dog | mat-rug | animals | things | across")
for name, n_params, col in results:
    print(f"{name:>18} | {col[2]:7.2f} | {col[3]:7.2f} | {col[4]:7.2f} | {col[5]:6.2f} | {col[6]:6.2f}")
