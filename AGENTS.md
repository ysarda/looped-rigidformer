# AGENTS.md

Notes for agents working in this repo. Read README.md for what this is and
UPSTREAM_README.md for how the model, data pipeline and scripts work. Upstream's
documentation is accurate; prefer it over guessing.

## This is vendored upstream code

The first commit is an unmodified copy of RigidFormer, tagged `upstream-aef8717`.
Everything after it is ours.

```bash
git diff upstream-aef8717 --stat     # our entire contribution
```

Keep that true. When changing upstream files, change as little as possible and say why
in a comment, so the diff stays readable as a patch someone could review against the
published code. New work belongs in new files where it reasonably can.

## Setup

```bash
uv sync
uv run pytest -q tests/
```

Everything runs through `uv run`. torch is pinned to 2.6.0+cu124; don't bump it
casually, upstream pins the same version.

The repo root is the import root: upstream code imports `from src.models...`, so run
commands from here, never from inside `src/`.

## Layout

```
src/            upstream model, data and training code
configs/        upstream YAML, one per dataset; every run is --config <file>
scripts/        upstream train/eval/render entry points
data_generation/  dataset construction; object_scales_*.json is referenced by configs
datasets/       gitignored, the HF dataset written to disk
checkpoints/    gitignored, downloaded and trained weights
exps/           gitignored, logs and checkpoints from runs
```

## Things that will bite you

**No flash-attn on this machine.** There is no Windows wheel. `src/models/decoder.py`
falls back to `src/utils/sdpa_fallback.py`, which is exact attention through SDPA.
Expect different speed and memory, not different numbers.

**One GPU, 12 GB.** Upstream trains MOVi-A on 4 GPUs x 32 scenes with a 768-dim decoder
over a 1024-dim encoder. Nothing at that size and batch fits here. Scale down and record
what was changed, because a shrunk model is not comparable to their released weights.

**A released checkpoint is refused if the config disagrees.** Upstream stores the config
in the weights file and checks it. Evaluate a checkpoint with the config it was trained
with.

**Validation loss does not track rollout quality.** Established at length on the previous
implementation: it fell monotonically across a span where 100-frame terminal RMSE rose
68%. Upstream agrees in practice, selecting checkpoints on `p_rmse_@200/300/400` and
setting `enable_validation: false`. Judge runs by rollout error at the horizon you care
about.

**The published 50/75/100 horizons are pre-impact.** Measured over 120 MOVi-A validation
scenes: no scene touches the ground before frame 75 at any threshold from 5 mm to 5 cm,
and one does before frame 100. Contact, settling and resting only appear past ~frame 90,
which is why upstream evaluates to 480 frames.

## Conventions

- Comments explain why, not what. Don't narrate changes in code comments; the diff and
  commit message are for that.
- No em dashes in prose. Commas.

## Before committing

```bash
uv run pytest -q tests/
```

**Ask before committing.** Don't commit or push unprompted.
