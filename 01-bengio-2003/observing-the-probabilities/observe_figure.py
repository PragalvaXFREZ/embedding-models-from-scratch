# Draws the figures for the "Observing the Probabilities" blog with plain matplotlib.
# Same model, seed and training as observe_original.py, so the numbers match.
# Usage: python observe_figure.py <output dir>
import math
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import torch
import torch.nn.functional as F

text = "the cat sat on the mat the dog sat on the rug".split()
vocab = sorted(set(text))
w2i = {w: i for i, w in enumerate(vocab)}
V, m, n, h = len(vocab), 5, 3, 8
contexts = [[w2i[w] for w in text[i - 2:i]] for i in range(2, len(text))]
X = torch.tensor(contexts)
Y = torch.tensor([w2i[w] for w in text[2:]])

g = torch.Generator().manual_seed(42)
C = torch.randn(V, m, generator=g, requires_grad=True)
H = torch.randn((n - 1) * m, h, generator=g, requires_grad=True)
d = torch.randn(h, generator=g, requires_grad=True)
U = torch.randn(h, V, generator=g, requires_grad=True)
W = torch.randn((n - 1) * m, V, generator=g, requires_grad=True)
b = torch.randn(V, generator=g, requires_grad=True)
params = [C, H, d, U, W, b]

TRACKED = [("the cat", "sat"), ("the dog", "sat"), ("on the", "mat")]
rows = [contexts.index([w2i[w] for w in c.split()]) for c, _ in TRACKED]

STEPS = 1000
losses, grads, tracked, history = [], [], [], []
for step in range(STEPS + 1):
    history.append(C.detach().clone())
    x = C[X].view(X.shape[0], -1)
    y = b + x @ W + torch.tanh(x @ H + d) @ U
    loss = F.cross_entropy(y, Y)
    for p in params:
        p.grad = None
    loss.backward()
    losses.append(loss.item())
    grads.append(C.grad.norm().item())
    probs = F.softmax(y.detach(), dim=1)
    tracked.append([probs[r, w2i[t]].item() for r, (_, t) in zip(rows, TRACKED)])
    if step == STEPS:
        break
    with torch.no_grad():
        for p in params:
            p -= 0.1 * p.grad

Ch = torch.stack(history)  # [steps+1, V, m]


def cos_series(a, c):
    return F.cosine_similarity(Ch[:, w2i[a]], Ch[:, w2i[c]], dim=1).tolist()


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
steps = list(range(STEPS + 1))

# 1. probability of the right answer for the three examples, first 200 steps
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, (c, t) in enumerate(TRACKED):
    ax.plot(steps[:201], [p[i] for p in tracked[:201]], label=f"P({t} | {c})")
ax.axhline(0.5, color="gray", ls="--", lw=1)
ax.set_xlabel("training step")
ax.set_ylabel("probability of the right word")
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(out, "observing-the-probabilities-tracked.png"), dpi=150)
plt.close(fig)

# 2. loss and the gradient on C
floor = 2 * math.log(2) / len(Y)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
a1.plot(steps, losses)
a1.axhline(floor, color="gray", ls="--", lw=1, label=f"lowest possible ({floor:.3f})")
a1.set_yscale("log")
a1.set_yticks([0.139, 0.2, 0.5, 1, 2, 6.4])
a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
a1.yaxis.set_minor_formatter(NullFormatter())
a1.set_xlabel("training step")
a1.set_title("loss")
a1.legend()
a2.plot(steps, grads, color="tab:orange")
a2.set_yscale("log")
a2.set_xlabel("training step")
a2.set_title("size of the gradient on C")
fig.tight_layout()
fig.savefig(os.path.join(out, "observing-the-probabilities-loss.png"), dpi=150)
plt.close(fig)

# 3. cosine similarity of the two pairs during training
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(steps, cos_series("cat", "dog"), label="cat and dog")
ax.plot(steps, cos_series("mat", "rug"), label="mat and rug")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("training step")
ax.set_ylabel("cosine similarity")
ax.set_ylim(-1, 1)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out, "observing-the-probabilities-pairs.png"), dpi=150)
plt.close(fig)

# 4. cover: the one thing the model could never get past
plt.rcParams["mathtext.fontset"] = "cm"
fig = plt.figure(figsize=(9, 4.5))
fig.text(0.5, 0.5, r"$P(\mathrm{mat} \mid \mathrm{on,\,the}) = P(\mathrm{rug} \mid \mathrm{on,\,the}) = 0.5$",
         ha="center", va="center", fontsize=30)
fig.savefig(os.path.join(out, "observing-the-probabilities.png"), dpi=100, facecolor="white")
plt.close(fig)

print("final loss", round(losses[-1], 4), "tracked", [round(p, 3) for p in tracked[-1]])
