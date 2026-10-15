"""Read per-object vertices from a scene's verts_packed.npy (written by pack_vertices.py) instead of one file each.

Open memmaps live in a module-level dict rather than on the dataset, so they are never pickled into DataLoader
workers: each worker maps the scenes it touches itself, and the OS page cache is shared between them.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_open: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}


def _scene(scene_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    key = str(scene_dir)
    if key not in _open:
        nv_path = scene_dir / 'verts_packed_nv.npy'
        if nv_path.exists():
            nv = np.load(nv_path)
            _open[key] = (np.load(scene_dir / 'verts_packed.npy', mmap_mode='r'), np.concatenate([[0], np.cumsum(nv)]))
        else:
            _open[key] = None
    return _open[key]


def available(scene_dir: Path) -> bool:
    return _scene(Path(scene_dir)) is not None


def load_frame(frame_dir: Path) -> Optional[List[np.ndarray]]:
    """Per-object arrays of frame_dir exactly as its obj_<i>_pos.npy files hold them, or None if the scene is not packed."""
    frame_dir = Path(frame_dir)
    s = _scene(frame_dir.parent)
    if s is None:
        return None
    packed, off = s
    row = np.array(packed[int(frame_dir.name.split('_')[1])])
    return [row[off[i]:off[i + 1]] for i in range(len(off) - 1)]
