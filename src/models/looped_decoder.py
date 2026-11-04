"""Looped decoder: one decoder layer applied T times with shared weights, in place of N distinct layers.

Upstream's BidirectionalDecoder runs `num_layers` distinct layers once each. This runs a single shared layer for T
passes, with T drawn at random per training batch and fixed for evaluation, so depth becomes a runtime choice rather
than a parameter count. Everything else is upstream's forward pass, unchanged: register tokens, anchor RoPE and FiLM
on the step size.

Two readouts, set by model.decoder.loop.readout:

'slots' (the first variant). The predictor reads the decoder outputs listed in train.py's layer indices,
[0, 1, 2, num_layers] for a 4-layer decoder; after T passes the slots hold passes 0, 1, max(1, min(2, T - 1)) and T.
Block AttnRes stays on, its per-layer pseudo-queries shared into one, and its depth attention mixes the original
object tokens back in. Trained this way the model settled after about two passes: the predictor got most of what it
needed from the early slots, and only the final output was supervised, so later passes had nothing to do.

'last' (Yang et al. 2023, "Looped Transformers are Better at Learning Learning Algorithms"). The predictor reads
only the current pass, in every slot. Each pass's input is the previous pass's output plus the original tokens
(input injection), so AttnRes is off. In training the last `loss_window` passes are each read out, sent through the
predictor and physics step, and supervised, with the loss averaged over them: every pass must be a usable answer,
which teaches the loop to converge rather than to depend on one depth. Passes before the window run without
gradient (truncated backpropagation), which bounds memory at large T.

With `normalize`, each pass's input is norm(previous output) + original tokens, with one learned norm shared by the
passes. The decoder layers are pre-norm residual, so nothing else shrinks the state: without it, the first 'last'
run's state grew about 13% per pass (RMS 0.12 to 12 by pass 20, 159 by pass 40) and the loss went NaN in epoch 1.

The supervised passes travel as copies of the batch: the decoder returns them stacked along the batch dimension,
the predictor's other inputs are repeated to match (pass_stacking_hook), train.py repeats the rest of the step's
inputs and the batch it computes the loss against (repeat_for_passes), and upstream's loss, a mean over every anchor
of the batch, becomes the mean over the supervised passes.

Config (model.decoder.loop):
    enable: true
    train_min_iterations: 2        # T ~ uniform{min..max} per training batch
    train_max_iterations: 6
    eval_iterations: 4             # T in eval mode; evaluate another T with inference.allow_switch_mismatch
    readout: slots                 # or 'last'
    loss_window: 1                 # 'last' only: supervised passes per training batch, min(window, T)
    normalize: false               # 'last' only: normalise the state before re-injecting the tokens
"""
import copy
from contextlib import nullcontext

import torch

from src.models.decoder import BidirectionalDecoder


class _Repeat:
    """Stands in for the decoder's ModuleList: the same layer, `n` times."""

    def __init__(self, layer, n):
        self.layer, self.n = layer, n

    def __iter__(self):
        return (self.layer for _ in range(self.n))

    def __len__(self):
        return self.n


class _SharedQuery:
    """Stands in for the per-layer AttnRes pseudo-queries: one query for every pass."""

    def __init__(self, query):
        self.query = query

    def __getitem__(self, idx):
        return self.query


class _InjectedPasses:
    """Stands in for the decoder's ModuleList in the 'last' readout: pass k feeds the layer its input plus the first
    pass's input (the original tokens, registers included), and passes before `first_grad` run without gradient.
    `norm`, if given, is applied to the previous output before the tokens are added."""

    def __init__(self, layer, n, first_grad, norm=None):
        self.layer, self.n, self.first_grad, self.norm = layer, n, first_grad, norm

    def __iter__(self):
        injected = []

        def run(k):
            def call(state, obj, *args, **kwargs):
                with (torch.no_grad() if k < self.first_grad else nullcontext()):
                    if k == 1:
                        injected.append(obj)
                    else:
                        obj = (self.norm(obj) if self.norm is not None else obj) + injected[0]
                    return self.layer(state, obj, *args, **kwargs)
            return call
        return (run(k) for k in range(1, self.n + 1))

    def __len__(self):
        return self.n


class LoopedDecoder(BidirectionalDecoder):
    def __init__(self, *args, loop=None, **kwargs):
        super().__init__(*args, **kwargs)
        loop = loop or {}
        self.train_min = int(loop.get('train_min_iterations', 2))
        self.train_max = int(loop.get('train_max_iterations', 6))
        self.eval_iterations = int(loop.get('eval_iterations', self.num_layers))
        self.readout = loop.get('readout', 'slots')
        self.loss_window = int(loop.get('loss_window', 1))
        self.normalize = bool(loop.get('normalize', False))
        if not 1 <= self.train_min <= self.train_max or self.eval_iterations < 1:
            raise ValueError(f'decoder.loop needs 1 <= train_min <= train_max and eval_iterations >= 1, got '
                             f'{self.train_min}, {self.train_max}, {self.eval_iterations}')
        if self.readout not in ('slots', 'last'):
            raise ValueError(f"decoder.loop.readout must be 'slots' or 'last', got {self.readout!r}")
        if self.loss_window < 1 or (self.readout == 'slots' and self.loss_window != 1):
            raise ValueError(f"decoder.loop.loss_window must be >= 1, and 1 with readout 'slots'; got {self.loss_window}")
        if self.readout == 'last' and self.block_attn_res:
            raise ValueError("decoder.loop.readout 'last' injects the input itself; turn block_attn_res off")
        if self.normalize and self.readout != 'last':
            raise ValueError("decoder.loop.normalize applies to readout 'last' only")
        self.nominal_layers = self.num_layers
        self.nominal_indices = set(self.layer_indices_to_save)
        if self.nominal_indices != {0, 1, 2, self.nominal_layers}:
            raise ValueError(f'LoopedDecoder maps the predictor layers [0, 1, 2, num_layers], got '
                             f'{sorted(self.nominal_indices)}')

        # keep the first layer as the shared one; drop the others so they hold no parameters
        shared = self.layers[0]
        del self.layers
        self.shared_layer = shared
        # a fresh copy of the output norm: same type (RMS or layer norm) and width, its own weights
        self.loop_norm = copy.deepcopy(self.obs_norm) if self.normalize else None
        if self.block_attn_res:
            query = self.attn_res_queries[0]
            del self.attn_res_queries
            self.shared_attn_res_query = query
        self.last_iterations = self.eval_iterations
        self.last_window = 1
        print(f'[Decoder] Looped: 1 shared layer, T ~ uniform{{{self.train_min}..{self.train_max}}} in training, '
              f'T = {self.eval_iterations} in evaluation, readout {self.readout!r}'
              + (f', loss on the last {self.loss_window} passes' if self.readout == 'last' else '')
              + (', state normalised between passes' if self.normalize else ''), flush=True)

    def iterations(self):
        if self.training:
            return int(torch.randint(self.train_min, self.train_max + 1, ()).item())
        return self.eval_iterations

    def forward(self, *args, **kwargs):
        T = self.iterations()
        K = min(self.loss_window, T) if self.training else 1
        self.last_iterations, self.last_window = T, K
        # upstream's forward iterates self.layers, indexes self.attn_res_queries and saves the outputs whose
        # 1-based layer index is in layer_indices_to_save; point all three at T passes of the shared layer
        if self.readout == 'slots':
            self.layers = _Repeat(self.shared_layer, T)
            # slot 2 falls back to pass 1 when there is no pass between it and the last
            slots = [0, 1, max(1, min(2, T - 1)), T]
        else:
            self.layers = _InjectedPasses(self.shared_layer, T, T - K + 1 if K > 1 else 1, self.loop_norm)
            slots = list(range(T - K + 1, T + 1))
        if self.block_attn_res:
            self.attn_res_queries = _SharedQuery(self.shared_attn_res_query)
        self.num_layers = T
        self.layer_indices_to_save = set(slots)
        try:
            out = super().forward(*args, **kwargs)
        finally:
            del self.layers
            if self.block_attn_res:
                del self.attn_res_queries
            self.num_layers = self.nominal_layers
            self.layer_indices_to_save = self.nominal_indices
        if not self.return_all_layers:
            return out
        # upstream returns the saved passes once each, in pass order
        saved = sorted(set(slots))
        states, objs = out
        if self.readout == 'slots':
            states = [states[saved.index(s)] for s in slots] if states is not None and len(states) == len(saved) else states
            return states, [objs[saved.index(s)] for s in slots]
        # upstream normalises the last saved output only; every supervised pass is a readout, so normalise each
        objs = [self.obs_norm(o) for o in objs[:-1]] + [objs[-1]]
        obj = torch.cat(objs, dim=0) if K > 1 else objs[-1]
        state = torch.cat(states, dim=0) if (K > 1 and states is not None) else (states[-1] if states else None)
        n = len(self.nominal_indices)
        return [state] * n, [obj] * n


def pass_stacking_hook(module, args, kwargs):
    """Forward pre-hook for the predictor: when the decoder features hold K supervised passes stacked along the batch,
    repeat the predictor's other inputs K times to match."""
    feats = args[0] if args else kwargs['multi_scale_features']
    rows = feats[-1].shape[0]
    query = args[1] if len(args) > 1 else None
    base = (query if query is not None else kwargs['query_pos']).shape[0]
    if rows == base:
        return None
    if rows % base:
        raise ValueError(f'decoder rows {rows} are not a multiple of the predictor batch {base}')
    K = rows // base

    def rep(x):
        if torch.is_tensor(x) and x.shape[:1] == (base,):
            return x.repeat(K, *[1] * (x.dim() - 1))
        if isinstance(x, (list, tuple)) and x and torch.is_tensor(x[0]) and x[0].shape[0] == base:
            return type(x)(rep(v) for v in x)
        return x
    args = (args[0],) + tuple(rep(a) for a in args[1:])
    return args, {k: (v if k == 'multi_scale_features' else rep(v)) for k, v in kwargs.items()}


def repeat_for_passes(K, *tensors):
    """K pass-major copies of each tensor along dim 0 (None passes through)."""
    return tuple(None if x is None else x.repeat(K, *[1] * (x.dim() - 1)) for x in tensors)


def repeat_batch_for_passes(batch, predictions):
    """The batch the loss compares against, repeated once per supervised pass the predictions hold."""
    B = batch['obj_positions'].shape[0]
    K = predictions['obj_positions_norm'].shape[0] // B
    if K == 1:
        return batch
    # only what the loss reads: a shape test alone could match an unbatched tensor whose first size equals B
    keys = ('obj_positions', 'valid_obj_mask', 'valid_vtx_mask', 'anchor_idx', 'skip_step_size', 'time_step_id')
    return {k: (repeat_for_passes(K, v)[0] if k in keys and torch.is_tensor(v) else v) for k, v in batch.items()}
