"""Quick validation GIFs of exported rollouts (scripts/export_rollouts.py) against the ground truth, in matplotlib.

One 3D panel per model. Every object is its collision mesh at the predicted pose, drawn solid, with the ground truth
overlaid as a translucent copy outlined in the same colour. With two models, a third panel draws the second solid
over the first translucent, to show where the two disagree regardless of the ground truth. The style follows the rollout GIFs of the earlier
implementation (graphics/pipeline/visualize_hopnet.py).

    python scripts/plot_rollouts.py --rollouts released=exps/rollouts/released control=exps/rollouts/control_ep17 \\
        --data datasets/MOVi-A --busiest 3 --out exps/videos/compare

Output: <out>/<scene>_step_<s>.gif. The default plays every 3rd frame at 20 fps, a quarter of real time.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESHES = os.path.join(REPO, 'data_generation', 'objects')
FRAME_RATE = 240                                     # physical frames per second of the MOVi conversions
OBJECT_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
                 '#008300', '#4a3aa7', '#e34948', '#00a0b0', '#a05a2c')
INK, INK_MUTED, SURFACE = '#242422', '#6b6b64', '#fcfcfb'


def collision_mesh(shape):
    """Vertices and faces of the collision mesh in file order, the order of an object's vertices in the data."""
    m = trimesh.load(os.path.join(MESHES, shape, 'collision_geometry.obj'), process=False)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces)


def world_vertices(meshes, sizes, q_wxyz, t):
    """Per object, (K, Nv, 3) vertices at each of the K poses; the pose maps the unit mesh scaled by size."""
    out = []
    for i, (v, _) in enumerate(meshes):
        R = Rotation.from_quat(q_wxyz[:, i][:, [1, 2, 3, 0]]).as_matrix()              # (K, 3, 3)
        out.append(np.einsum('kij,nj->kni', R, v * float(sizes[i])) + t[:, i, None, :])
    return out


def busiest(scenes, data_dir, n):
    motion = {}
    for s in scenes:
        traj = np.load(os.path.join(data_dir, s, 'vis_traj.npy'))[0]
        motion[s] = float(np.linalg.norm(np.diff(traj[:, :, :3], axis=0), axis=-1).sum())
    return sorted(scenes, key=lambda s: -motion[s])[:n]


def render(scene, step, models, data_dir, out_path, every, fps, dpi):
    gt_traj = np.load(os.path.join(data_dir, scene, 'vis_traj.npy'))[0].astype(np.float64)
    rows = []
    for name, folder in models:
        d = np.load(os.path.join(folder, f'{scene}_ss{step}.npz'))
        frames = d['frame'].astype(int)
        meshes = [collision_mesh(str(s)) for s in d['shape']]
        pred = world_vertices(meshes, d['size'], d['rotation'].astype(np.float64), d['translation'].astype(np.float64))
        gt = world_vertices(meshes, d['size'], gt_traj[frames, :, 3:7], gt_traj[frames, :, :3])
        rows.append((name, frames, meshes, pred, gt, d['translation'].astype(np.float64)))
    frames = rows[0][1]
    if any(not np.array_equal(r[1], frames) for r in rows):
        raise SystemExit(f'{scene} step {step}: the exports predict different frames')
    gt_t = gt_traj[frames, :, :3]
    meshes = rows[0][2]

    # each panel: title, solid vertices, outlined vertices, mean centre distance between the two per frame
    def gap(a, b):
        return np.linalg.norm(a - b, axis=-1).mean(axis=1)
    panels = [(f'{name} (solid) vs ground truth\nmean centre error {{:.3f}} m', pred, gt, gap(t, gt_t))
              for name, _, _, pred, gt, t in rows]
    if len(rows) == 2:
        (a, _, _, pa, _, ta), (b, _, _, pb, _, tb) = rows
        panels.append((f'{b} (solid) vs {a} (outline)\nmean centre gap {{:.3f}} m', pb, pa, gap(tb, ta)))

    # fixed limits around the ground truth, so the view does not jump; a prediction that flies away leaves the frame,
    # which is itself the finding, rather than shrinking every object to fit it
    allv = np.concatenate([v.reshape(-1, 3) for v in rows[0][4]])
    lo, hi = allv.min(0), allv.max(0)
    pad = 0.05 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    lo[2] = 0.0

    fig = plt.figure(figsize=(6.4 * len(panels), 6.0), facecolor=SURFACE)
    axes = [fig.add_subplot(1, len(panels), p + 1, projection='3d') for p in range(len(panels))]
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.88, wspace=0.0)
    title = fig.suptitle('', fontsize=13, color=INK, y=0.97)
    fig.text(0.5, 0.905, 'translucent with outline = the reference each panel compares against', ha='center',
             fontsize=10, color=INK_MUTED)

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, out_path, dpi=dpi):
        for k in range(0, len(frames), every):
            for ax, (label, solid, ghost, err) in zip(axes, panels):
                ax.cla()
                gx, gy = np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]])
                ax.plot_surface(gx, gy, np.zeros((2, 2)), alpha=0.12, color='gray')
                for i, (_, faces) in enumerate(meshes):
                    c = OBJECT_COLORS[i % len(OBJECT_COLORS)]
                    ax.add_collection3d(Poly3DCollection(ghost[i][k][faces], facecolor=c, alpha=0.25, edgecolor=c,
                                                         linewidth=0.7))
                    ax.add_collection3d(Poly3DCollection(solid[i][k][faces], facecolors=c, alpha=0.95, linewidth=0,
                                                         shade=True))
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(lo[1], hi[1])
                ax.set_zlim(lo[2], hi[2])
                ax.set_axis_off()
                ax.set_box_aspect(tuple(hi - lo), zoom=1.35)
                ax.view_init(elev=24, azim=-62)
                ax.set_title(label.format(err[k]), fontsize=12, color=INK, y=0.93)
            title.set_text(f'{scene}, step size {step}: frame {frames[k]} ({frames[k] / FRAME_RATE:.2f} s)')
            writer.grab_frame(facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rollouts', nargs='+', required=True, help='<name>=<export folder>, one panel each')
    ap.add_argument('--data', required=True, help='dataset folder with the ground truth (vis_traj.npy)')
    ap.add_argument('--scenes', default='', help='comma-separated scene names')
    ap.add_argument('--busiest', type=int, default=0, help='the N test scenes with the most ground-truth motion')
    ap.add_argument('--steps', default='1')
    ap.add_argument('--every', type=int, default=3, help='draw every Nth predicted frame')
    ap.add_argument('--fps', type=float, default=20.0)
    ap.add_argument('--dpi', type=int, default=90)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    models = [tuple(s.split('=', 1)) for s in args.rollouts]
    index = json.load(open(os.path.join(models[0][1], 'index.json')))
    scenes = [s for s in args.scenes.split(',') if s] or busiest(index['scenes'], args.data, args.busiest)
    if not scenes:
        raise SystemExit('no scenes: pass --scenes or --busiest')
    os.makedirs(args.out, exist_ok=True)
    for scene in scenes:
        for step in (int(s) for s in args.steps.split(',')):
            path = os.path.join(args.out, f'{scene}_step_{step}.gif')
            render(scene, step, models, args.data, path, args.every, args.fps, args.dpi)
            print(path, flush=True)


if __name__ == '__main__':
    main()
