# Bengio et al. (2003): A Neural Probabilistic Language Model

A from-scratch PyTorch rebuild of [Bengio, Ducharme, Vincent & Jauvin (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf),
trained on a 12-word corpus: `the cat sat on the mat the dog sat on the rug`.

- [`bengio_2003.ipynb`](bengio_2003.ipynb): the full walkthrough. It covers why the loss stops at about 0.139,
  why `cat` never ends up near `dog`, and how to fix it. Start here.
- [`bengio.py`](bengio.py): the bare model and training loop.
- [`observing-the-probabilities/`](observing-the-probabilities/): experiments and figures for the blog post
  [Observing the Probabilities](https://pragalva.me/blog/observing-the-probabilities/).
  Run `python observe_figure.py <output dir>` to redraw the figures.

All scripts share the same seed and initialisation order, so their numbers match the notebook.
