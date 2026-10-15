"""pack_vertices.py and src/data/packed_vertices.py reproduce the per-object vertex files exactly."""
import numpy as np
import pytest

from pack_vertices import pack_scene
from src.data import packed_vertices


def _scene(root, nv=(5, 3, 8), T=4, seed=0):
    rng = np.random.default_rng(seed)
    scene = root / 'movi_0'
    for t in range(T):
        f = scene / f'frame_{t:05d}'
        f.mkdir(parents=True)
        for i, n in enumerate(nv):
            np.save(f / f'obj_{i}_pos.npy', rng.standard_normal((n, 3)).astype(np.float32))
    return scene


def test_packed_frames_match_files(tmp_path):
    scene = _scene(tmp_path)
    assert pack_scene(str(scene)) == 'new'
    for f in sorted(scene.glob('frame_*')):
        objs = packed_vertices.load_frame(f)
        assert len(objs) == 3
        for i, V in enumerate(objs):
            W = np.load(f / f'obj_{i}_pos.npy')
            assert V.dtype == W.dtype and np.array_equal(V, W)
    assert pack_scene(str(scene)) == 'existing'


def test_unpacked_scene_falls_back(tmp_path):
    scene = _scene(tmp_path)
    assert packed_vertices.load_frame(scene / 'frame_00000') is None


def test_inconsistent_vertex_count_refused(tmp_path):
    scene = _scene(tmp_path)
    np.save(scene / 'frame_00002' / 'obj_1_pos.npy', np.zeros((4, 3), np.float32))
    with pytest.raises(ValueError, match='expected'):
        pack_scene(str(scene))
    assert not (scene / 'verts_packed_nv.npy').exists()
