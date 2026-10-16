# looped-rigidformer

Looped transformers applied to RigidFormer, a learned simulator for rigid-body dynamics.

## RigidFormer

[RigidFormer](https://people.csail.mit.edu/frankzydou/projects/RigidFormer/index.html)
(Dou et al., NeurIPS 2026, [arXiv:2605.09196](https://arxiv.org/abs/2605.09196)) learns
how rigid objects move, collide and come to rest, directly from point samples of their
surfaces and without meshes. Each object becomes one token. A transformer reads the
recent history of every object in the scene and predicts the next state, and the
prediction is projected back onto a rigid motion so objects can't deform. A rollout
repeats that step hundreds of times, feeding each prediction back in as input.

## Looped transformers

A standard transformer applies a stack of distinct layers once each. A looped
transformer instead applies one shared block, or a small group of them, many times in a
row, so depth comes from repetition instead of extra parameters
([Dehghani et al. 2018](https://arxiv.org/abs/1807.03819),
[Giannou et al. 2023](https://arxiv.org/abs/2301.13196),
[Yang et al. 2023](https://arxiv.org/abs/2311.12424)). The number of iterations can
differ between training and inference, so the model can spend more compute on harder
inputs. Looped models tend to learn iterative procedures better than fixed-depth ones of
the same size.

## This repo

Both ideas are about repetition, which is why this repo combines them. Resolving contact
between rigid bodies is itself an iterative process: physics engines run a constraint
solver for several passes each step until the objects stop overlapping. A looped
decoder could learn that refinement as a loop of its own, and spend more iterations on
the frames where objects collide. The question is whether that improves long rollouts,
where impacts and settling dominate the error, at equal or smaller parameter count.

Runs are compared by rollout position error over 400 frames on MOVi-A, against a
1/8-size RigidFormer trained on the same 18-epoch budget as a control.

## Based on

The code started as an unmodified copy of the `RigidFormer/` directory of
[Frank-ZY-Dou/Dynamics-Modeling](https://github.com/Frank-ZY-Dou/Dynamics-Modeling) at
[`aef8717`](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/aef87174da8f2eaade385b10b946f0459cdd0ef2/RigidFormer),
tagged `upstream-aef8717` here, so `git diff upstream-aef8717` shows every change since.
Their MIT [LICENSE](LICENSE) is kept, and their README is preserved as
[UPSTREAM_README.md](UPSTREAM_README.md). It remains the reference for the data
pipeline, configs and scripts.

## Changes to the RigidFormer code

| Change | Why |
|---|---|
| `src/utils/sdpa_fallback.py`, imported by `src/models/decoder.py` | flash-attn has no Windows wheel. The fallback computes the same exact attention through `F.scaled_dot_product_attention`. |
| `pack_vertices.py`, `src/data/packed_vertices.py`, read by `src/data/multi_dataloader.py` | Loading opened ~160 small files per sample, which starved the GPU on Windows. Packing each scene into one memory-mapped file (bit-identical values) more than doubled training throughput. Unpacked scenes load as before. |
| `configs/movi_a_small.yaml` | The single-GPU control: 1/8 the parameters (17.9M), 18 epochs. Not comparable to the released weights. |
| `pyproject.toml` replaces `requirements.txt` | `uv` manages the environment. flash-attn, `open3d` and `polyscope` are left out; the last two are only needed to render videos. |

## Setup

```bash
uv sync
```

A CUDA GPU is required. Download the dataset as described in
[UPSTREAM_README.md](UPSTREAM_README.md), then pack it once (about an hour for MOVi-A):

```bash
uv run python pack_vertices.py datasets/MOVi-A
```

The released weights:

```bash
hf download frankzydou/RigidFormer rigidformer_movia.pth --local-dir checkpoints
```

On Windows `torchrun` cannot start, and one GPU needs no launcher:

```bash
uv run python src/train.py --config configs/movi_a_small.yaml --use_multi_dataset
```
