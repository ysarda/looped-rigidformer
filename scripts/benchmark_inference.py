"""Rollout speed of a checkpoint on MOVi-A test scenes, through the evaluation's own data and rollout code.

Runs the evaluation of scripts/eval.sh in this process, times each batch's rollout with the GPU synchronised before
and after, and stops after `--batches` timed batches. The first batch is run untimed, since it carries one-off CUDA
setup. Every model gets the same scenes, batch size and step size, so the times compare directly.

    python scripts/benchmark_inference.py --checkpoint checkpoints/rigidformer_movia.pth --config configs/movi_a.yaml
    python scripts/benchmark_inference.py --checkpoint <looped best.pth> --config configs/movi_a_small_looped.yaml \\
        --iterations 8

--iterations sets the looped decoder's pass count T for the run. Prints one JSON line starting with BENCHMARK.
"""
import argparse
import json
import os
import shutil
import socket
import sys
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
sys.path.insert(0, REPO)


class _Done(Exception):
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True, help='the config the checkpoint was trained with')
    ap.add_argument('--iterations', type=int, default=None, help='looped decoder passes T')
    ap.add_argument('--batches', type=int, default=4, help='timed batches, after one untimed')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--step_size', type=int, default=1)
    ap.add_argument('--label', default=None)
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    os.environ.update({'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '1', 'MASTER_ADDR': '127.0.0.1',
                       'MASTER_PORT': str(port)})

    import torch
    import train                                      # src/train.py
    from eval_cross_dataset import build_target_config

    tag = f'bench_A_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    cfg = build_target_config(yaml.safe_load(open(args.config)), 'A', tag)
    cfg['data']['eval_step_sizes'] = [args.step_size]
    cfg['training']['eval_batch_size'] = args.batch_size
    if args.iterations is not None:
        cfg['model']['decoder']['loop']['eval_iterations'] = args.iterations
        cfg.setdefault('inference', {})['allow_switch_mismatch'] = ['model.decoder.loop.eval_iterations']
    tmp_cfg = os.path.join(REPO, 'configs', f'_tmp_{tag}.yaml')
    yaml.safe_dump(cfg, open(tmp_cfg, 'w'), sort_keys=False)

    times, frames = [], []
    rollout = train.CUT3RDynamicsModel.rollout

    def timed_rollout(self, initial_batch, num_steps, dt=0.01, return_extras=True, warmup_frames=2):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = rollout(self, initial_batch, num_steps, dt=dt, return_extras=return_extras, warmup_frames=warmup_frames)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        frames.append(num_steps - warmup_frames)
        if len(times) > args.batches:
            raise _Done
        return out

    train.CUT3RDynamicsModel.rollout = timed_rollout
    sys.argv = ['train.py', '--config', tmp_cfg, '--checkpoint', args.checkpoint, '--inference', '--use_multi_dataset']
    try:
        train.main()
    except _Done:
        pass
    finally:
        os.remove(tmp_cfg)
        # the evaluation opens a log and checkpoint folder per run; a benchmark has nothing to keep in them
        for d in (cfg['paths']['log_dir'], cfg['paths']['checkpoint_dir']):
            shutil.rmtree(os.path.join(REPO, d, tag), ignore_errors=True)
    if len(times) <= args.batches:
        raise SystemExit(f'only {len(times)} rollouts ran; need {args.batches + 1}')

    t, f = times[1:], frames[1:]
    params = None
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict') or ck.get('state_dict') or {}
    if sd:
        params = sum(v.numel() for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()
                     and 'running_' not in k)
    result = {'label': args.label or os.path.basename(args.checkpoint), 'iterations': args.iterations,
              'batch_size': args.batch_size, 'step_size': args.step_size, 'predicted_frames': f[0],
              'rollout_s_mean': sum(t) / len(t), 'ms_per_step': 1000 * sum(t) / sum(f),
              'gpu': torch.cuda.get_device_name(0), 'params': params}
    print('BENCHMARK ' + json.dumps(result), flush=True)
    # dataloader workers of the interrupted evaluation can hold the process open
    os._exit(0)


if __name__ == '__main__':
    main()
