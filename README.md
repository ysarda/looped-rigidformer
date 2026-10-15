# looped-rigidformer

Looped-transformer experiments built on RigidFormer, the object-centric transformer for
mesh-free rigid-body dynamics by Dou et al.

Upstream documentation is preserved verbatim in [UPSTREAM_README.md](UPSTREAM_README.md),
and it remains the reference for the data pipeline, the configs and the training and
evaluation scripts. This file covers only what differs here.

## Provenance

**The model, data pipeline, training code and assets are the work of the RigidFormer
authors, not ours.** Our contribution is only what has been added on top, as recorded
below.

This repository starts from an unmodified copy of the `RigidFormer/` directory of
[Frank-ZY-Dou/Dynamics-Modeling](https://github.com/Frank-ZY-Dou/Dynamics-Modeling) at
commit [`aef87174da8f2eaade385b10b946f0459cdd0ef2`](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/aef87174da8f2eaade385b10b946f0459cdd0ef2/RigidFormer)
(2026-10-12), the code release accompanying

> **RigidFormer: Learning Rigid Dynamics using Transformers.**
> Zhiyang Dou, Minghao Guo, Haixu Wu, Doug Roble, Tuur Stuyck, Wojciech Matusik.
> NeurIPS 2026. [arXiv:2605.09196](https://arxiv.org/abs/2605.09196) ·
> [project page](https://people.csail.mit.edu/frankzydou/projects/RigidFormer/index.html)

That copy is the first commit and contains nothing of ours. It is tagged
`upstream-aef8717`, so

```bash
git diff upstream-aef8717 -- src/        # every change we have made to the model
git diff upstream-aef8717 --stat         # the whole contribution, at a glance
```

always answers "what is ours and what is theirs" exactly, with no upstream remote
required. Their work is MIT licensed and [LICENSE](LICENSE) is retained unchanged.

### Why this is a copy rather than a fork

RigidFormer is one directory inside `Dynamics-Modeling`, a monorepo holding seven
unrelated projects (HandDiffSim, NeuralActuator, Penetration-Solving and others). It is
not a submodule and has no repository of its own, and GitHub forks whole repositories,
so forking would have meant carrying six projects we will never touch in order to track
one directory we will rewrite heavily.

This repository is a single project with a single focus: one model, one line of
experiments. Keeping it that way matters more here than an automatic upstream merge.
The code is a finished release accompanying a published paper, so upstream changes are
expected to be occasional fixes rather than redesigns, and the `upstream-aef8717` tag
preserves the one thing a fork would genuinely have bought us, a permanent and exact
diff of our contribution. Applying a future upstream fix means reading their patch and
porting it, which for a codebase we are modifying substantially is what a merge would
have degenerated into anyway.

## Differences from upstream

| Change | Why |
|---|---|
| `src/utils/sdpa_fallback.py`, imported by `src/models/decoder.py` | flash-attn has no Windows wheel and compiling it needs the CUDA toolkit. The fallback routes the decoder's unmasked attention through `F.scaled_dot_product_attention`, which is exact attention; the real package is still used when it is installed. |
| `pyproject.toml` replaces `requirements.txt` as the installed set | `uv` manages this environment. flash-attn is omitted per the above; `open3d` and `polyscope` are omitted because they are only needed to render rollout videos. |

`UPSTREAM_README.md` is their `README.md`, renamed so this file can take its place.

## Setup

```bash
uv sync
```

A CUDA GPU is required: the KNN and attention paths have no CPU implementation.

Data and pretrained weights download as upstream documents, into paths their
`.gitignore` already excludes:

```bash
hf download frankzydou/RigidFormer rigidformer_movia.pth --local-dir checkpoints
```

See [UPSTREAM_README.md](UPSTREAM_README.md) for the dataset download and the training
layout it writes.
