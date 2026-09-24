# Observations on the original 12-word corpus, for the "Observing the Probabilities" blog.
# Same model, seed and init order as bengio.py, so the numbers match the notebook.
import math
import torch
import torch.nn.functional as F

text = "the cat sat on the mat the dog sat on the rug".split()
vocab = sorted(set(text))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
V, m, n, h = len(vocab), 5, 3, 8

contexts = [[word_to_idx[w] for w in text[i - (n - 1):i]] for i in range(n - 1, len(text))]
targets = [word_to_idx[text[i]] for i in range(n - 1, len(text))]
X, Y = torch.tensor(contexts), torch.tensor(targets)

g = torch.Generator().manual_seed(42)
C = torch.randn(V, m, generator=g, requires_grad=True)
H = torch.randn((n - 1) * m, h, generator=g, requires_grad=True)
d = torch.randn(h, generator=g, requires_grad=True)
U = torch.randn(h, V, generator=g, requires_grad=True)
W = torch.randn((n - 1) * m, V, generator=g, requires_grad=True)
b = torch.randn(V, generator=g, requires_grad=True)
params = [C, H, d, U, W, b]


def logits(X):
    x = C[X].view(X.shape[0], -1)
    return b + x @ W + torch.tanh(x @ H + d) @ U


def show_predictions(title):
    with torch.no_grad():
        probs = F.softmax(logits(X), dim=1)
    print(f"\n{title}")
    print(f"{'context':>12} -> {'target':<6} | {'model picks':<11} | P(target)  -log P(target)")
    for ctx, tgt, p in zip(contexts, targets, probs):
        pick = p.argmax().item()
        c = " ".join(idx_to_word[i] for i in ctx)
        print(f"{c:>12} -> {idx_to_word[tgt]:<6} | {idx_to_word[pick]:<5} {p[pick].item():.2f} | "
              f"{p[tgt].item():9.3f}  {-math.log(p[tgt].item()):8.3f}")
    return probs


def neighbours(E, word):
    v = E[word_to_idx[word]]
    sims = (E @ v) / (E.norm(dim=1) * v.norm())
    return [(idx_to_word[i.item()], round(sims[i].item(), 2)) for i in sims.argsort(descending=True)[1:]]


def cos(E, a, c):
    return F.cosine_similarity(E[word_to_idx[a]], E[word_to_idx[c]], dim=0).item()


print("parameters:", {name: p.numel() for name, p in zip("C H d U W b".split(), params)},
      "total", sum(p.numel() for p in params))
print("training examples:", len(Y))

# ---- 1. untrained model
show_predictions("UNTRAINED model")
with torch.no_grad():
    print(f"\nuntrained loss: {F.cross_entropy(logits(X), Y).item():.4f}   "
          f"uniform guess ln(V) = {math.log(V):.4f}")

# ---- 2. training, watching loss, gradient size, and cat-dog similarity
C_before = C.detach().clone()
lr = 0.1
# the three examples the blog follows through training
TRACKED = [("the cat", "sat"), ("the dog", "sat"), ("on the", "mat")]
tracked_rows = [contexts.index([word_to_idx[w] for w in c.split()]) for c, _ in TRACKED]
print("\nstep |   loss  | grad norm of C | cos(cat,dog) | cos(mat,rug) | "
      + " | ".join(f"P({t}|{c})" for c, t in TRACKED))
for step in range(1001):
    loss = F.cross_entropy(logits(X), Y)
    for p in params:
        p.grad = None
    loss.backward()
    if step in (0, 10, 25, 50, 100, 200, 500, 1000):
        E = C.detach()
        print(f"{step:4d} | {loss.item():.4f} | {C.grad.norm().item():14.4f} | "
              f"{cos(E, 'cat', 'dog'):12.3f} | {cos(E, 'mat', 'rug'):12.3f} | " + " | ".join(
                  f"{F.softmax(logits(X), dim=1)[r, word_to_idx[t]].item():{len(c) + len(t) + 3}.3f}"
                  for r, (c, t) in zip(tracked_rows, TRACKED)))
        if step == 1000:
            rug_grad = C.grad[word_to_idx["rug"]].clone()
            break
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad

# ---- 3. trained model, and the loss floor
probs = show_predictions("TRAINED model (1000 steps)")
per_example = -torch.log(probs[torch.arange(len(Y)), Y])
floor = 2 * math.log(2) / len(Y)
print(f"\nsum of per-example losses: {per_example.sum().item():.4f}   mean: {per_example.mean().item():.4f}")
print(f"floor = (8 * 0 + 2 * ln 2) / 10 = {floor:.4f}")
on_the = probs[[i for i, c in enumerate(contexts) if c == [word_to_idx["on"], word_to_idx["the"]]][0]]
print("P(next | on the):", {idx_to_word[i]: round(on_the[i].item(), 3) for i in range(V) if on_the[i] > 0.01})
for p in (0.5, 0.7, 0.9, 0.99):
    print(f"  if P(mat|on the) = {p}: the two 'on the' examples cost {-math.log(p) - math.log(1 - p):.3f}")

# ---- 4. the embeddings
E = C.detach()
print()
for w in ["cat", "dog", "mat", "rug"]:
    print(f"{w} before: {neighbours(C_before, w)}")
    print(f"{w} after : {neighbours(E, w)}")
print()
for w in ["cat", "dog", "mat", "rug"]:
    in_ctx = sum(word_to_idx[w] in c for c in contexts)
    moved = (E[word_to_idx[w]] - C_before[word_to_idx[w]]).norm().item()
    print(f"{w:>4}: in {in_ctx} contexts, moved {moved:.3f} from its random start")
print("gradient on rug's row of C at the last step:", rug_grad.tolist())

# cat and dog get almost the same push, but the push is small next to where they started
move = {w: E[word_to_idx[w]] - C_before[word_to_idx[w]] for w in ("cat", "dog")}
print(f"\ndirection of cat's move vs dog's move (cosine): {F.cosine_similarity(move['cat'], move['dog'], dim=0).item():.2f}")
for w in ("cat", "dog"):
    print(f"{w}: moved {move[w].norm().item():.2f}, but its random start vector has length "
          f"{C_before[word_to_idx[w]].norm().item():.2f}")

# is "on the -> mat / rug" ambiguous because of the data, or because the model only sees 2 words?
print()
for k in range(1, 6):
    seen = {}
    for i in range(k, len(text)):
        seen.setdefault(" ".join(text[i - k:i]), set()).add(text[i])
    clash = {c: sorted(t) for c, t in seen.items() if len(t) > 1}
    print(f"context of {k} word(s): contexts with more than one right answer: {clash or 'none'}")
