"""Pack every scene's per-frame vertex files into one array per scene (verts_packed.npy).

The loader otherwise opens one obj_<i>_pos.npy per object per frame, ~160 files per training sample. On Windows
each first open of a file costs ~6.6 ms (Defender scans it), so data loading, not the GPU, bounds training.
A packed scene is one file the loader memory-maps once per worker.

    verts_packed.npy     (T, sum_i Nv_i, 3), the stored arrays verbatim, objects concatenated in index order
    verts_packed_nv.npy  (Nobj,) int64, Nv_i, so object i is columns [sum_{j<i} Nv_j, sum_{j<=i} Nv_j)

The packed values are bit-identical to the source files, which stay in place and remain the reference.

    uv run python pack_vertices.py datasets/MOVi-A [--workers 8]
"""
import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def pack_scene(scene: str) -> str:
    scene = Path(scene)
    out, out_nv = scene / 'verts_packed.npy', scene / 'verts_packed_nv.npy'
    if out.exists() and out_nv.exists():
        return 'existing'
    frames = sorted(p for p in scene.glob('frame_*') if p.is_dir())
    if [int(f.name.split('_')[1]) for f in frames] != list(range(len(frames))):
        raise ValueError(f'{scene}: frame directories are not frame_00000 .. frame_{len(frames) - 1:05d}')
    n_obj = 0
    while (frames[0] / f'obj_{n_obj}_pos.npy').exists():
        n_obj += 1
    ref = [np.load(frames[0] / f'obj_{i}_pos.npy') for i in range(n_obj)]
    nv = np.array([V.shape[0] for V in ref], dtype=np.int64)
    dtype = ref[0].dtype
    packed = np.empty((len(frames), int(nv.sum()), 3), dtype=dtype)
    for t, f in enumerate(frames):
        if (f / f'obj_{n_obj}_pos.npy').exists():
            raise ValueError(f'{f}: more objects than frame_00000 ({n_obj})')
        off = 0
        for i in range(n_obj):
            V = np.load(f / f'obj_{i}_pos.npy')
            # the checks the loader makes per sample, made once here
            if V.ndim != 2 or V.shape != (nv[i], 3) or V.dtype != dtype:
                raise ValueError(f'{f}/obj_{i}_pos.npy: shape {V.shape} {V.dtype}, expected ({nv[i]}, 3) {dtype}')
            packed[t, off:off + nv[i]] = V
            off += nv[i]
    pid = os.getpid()
    tmp, tmp_nv = scene / f'verts_packed.{pid}.tmp.npy', scene / f'verts_packed_nv.{pid}.tmp.npy'
    np.save(tmp, packed)
    np.save(tmp_nv, nv)
    # nv last: the loader only uses a scene whose nv file exists, so a half-written pack is never read
    os.replace(tmp, out)
    os.replace(tmp_nv, out_nv)
    return 'new'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    scenes = sorted(glob.glob(str(args.root / 'movi_*')))
    t0 = time.time()
    counts = {'new': 0, 'existing': 0}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(pack_scene, scenes, chunksize=4), 1):
            counts[r] += 1
            if k % 100 == 0 or k == len(scenes):
                print(f'[pack] {k}/{len(scenes)} scenes, {time.time() - t0:.0f}s', flush=True)
    print(f"[pack] {args.root}: new {counts['new']}, existing {counts['existing']}, {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
