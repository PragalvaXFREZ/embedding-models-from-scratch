import torch
import torch.nn.functional as F
# splits the sentence into words. 
text = "the cat sat on the mat the dog sat on the rug".split()
#set the words in a sorted order, repeated words are removed. 
vocab = sorted(set(text))
#an empty map of word to index
word_to_idx = {}
#mapping each word to the repective index
for i, w in enumerate(vocab):
    word_to_idx[w] = i 
#an empty map that will have index mapped out to words
idx_to_word = {}

# same thing but reverse done
for w, i in word_to_idx.items():
    idx_to_word[i] = w

#m is the number of Columns C has, it's the length of each word's feature vector
m = 5
#g is a random seed generator, so that our resuls remain reproducable by feeding in 42  
g = torch.Generator().manual_seed(42)

#C is our embedding matrix
C = torch.randn(len(vocab), m, generator=g, requires_grad=True)
# n is context window
n = 3
# context is what the model knows and uses to guess the target
contexts = []
targets = []
for i in range(n-1, len(text)):
    ctx = text[i - (n-1) : i]
    tgt = text[i]
    contexts.append([word_to_idx[w] for w in ctx])
    targets.append(word_to_idx[tgt])
# X is where our context's tensor is saved    
X = torch.tensor(contexts)
# Y is where our target's tensor is saved
Y = torch.tensor(targets)
# emb is the lookup 
emb = C[X]
# x is the concatenation of each C matrix, each word has a C matrix
x = emb.view(X.shape[0], -1)

# ------- The hidden layer
h = 8
H = torch.randn((n-1)*m, h, generator=g, requires_grad=True)
d = torch.randn(h, generator=g, requires_grad=True)

hidden = torch.tanh(x @ H + d)


V = len(vocab)
U = torch.randn(h, V, generator=g, requires_grad=True)
W = torch.randn((n-1)*m, V, generator=g, requires_grad=True)
b = torch.randn(V, generator=g, requires_grad=True)

y = b + x @ W + hidden @ U

probs = F.softmax(y, dim=1)

p_true = probs[torch.arange(len(Y)), Y]
loss = -torch.log(p_true).mean()
C_before = C.detach().clone()

params = [C, H, d, U, W, b]
lr = 0.1

for step in range(1000):
    emb = C[X]
    x = emb.view(X.shape[0], -1)
    hidden = torch.tanh(x @ H + d)
    y = b + x @ W + hidden @ U
    loss = F.cross_entropy(y, Y)

    for p in params:
        p.grad = None
    loss.backward()

    with torch.no_grad():
        for p in params:
            p -= lr * p.grad

    if step % 100 == 0:
        print(step, loss.item())

def neighbours(E, word):
    v = E[word_to_idx[word]]
    sims = (E @ v) / (E.norm(dim=1) * v.norm())
    order = sims.argsort(descending=True)
    return [(idx_to_word[i.item()], round(sims[i].item(), 3)) for i in order]

with torch.no_grad():
    for w in ["cat", "mat"]:
        print(w, "before:", neighbours(C_before, w))
        print(w, "after: ", neighbours(C, w))
