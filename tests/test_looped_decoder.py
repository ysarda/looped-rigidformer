"""src/models/looped_decoder.py runs upstream's decoder with one shared layer for T passes."""
import pytest
import torch

from src.models.decoder import BidirectionalDecoder
from src.models.looped_decoder import LoopedDecoder

D, NQ = 64, 4
KW = dict(state_dim=D, num_layers=4, num_heads=1, dropout=0.0, num_register_tokens=4, rope_set_aggregation=True,
          rope_dim=48, dim_per_coord_multi=4, layer_indices_to_save={0, 1, 2, 4}, elementwise_gate=True,
          state_update_enabled=False, objects_self_attention=True, block_attn_res=True, block_attn_res_block_size=4,
          block_attn_res_final_aggregate=True, film_conditioning=True)
LOOP = dict(train_min_iterations=2, train_max_iterations=6, eval_iterations=4)


def _inputs(B=2, N=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (None, torch.randn(B, N, D, generator=g)), dict(
        obj_anchor_pos=torch.randn(B, N, NQ * 3, generator=g), time_features=torch.rand(B, 2, generator=g))


def _looped(**loop):
    torch.manual_seed(0)
    m = LoopedDecoder(**KW, loop=dict(LOOP, **loop))
    # upstream initialises the AttnRes queries and FiLM to zero; perturb so a wrong wiring shows
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m


def test_one_shared_layer_holds_the_decoder_parameters():
    looped, plain = _looped(), BidirectionalDecoder(**KW)
    per_layer = sum(p.numel() for p in plain.layers[0].parameters()) + D          # one layer + one AttnRes query
    assert sum(p.numel() for p in plain.parameters()) - sum(p.numel() for p in looped.parameters()) == 3 * per_layer


@pytest.mark.parametrize('T', [1, 2, 3, 4, 7])
def test_T_passes_equal_a_stack_of_T_copies_of_the_layer(T):
    looped = _looped(eval_iterations=T).eval()
    plain = BidirectionalDecoder(**dict(KW, num_layers=T, layer_indices_to_save={0, 1, max(1, min(2, T - 1)), T})).eval()
    with torch.no_grad():
        shared = looped.state_dict()
        for k, v in plain.state_dict().items():
            if k.startswith('layers.'):
                v.copy_(shared['shared_layer.' + k.split('.', 2)[2]])
            elif k.startswith('attn_res_queries.'):
                v.copy_(shared['shared_attn_res_query'])
            else:
                v.copy_(shared[k])
    args, kw = _inputs()
    _, a = looped(*args, **kw)
    _, b = plain(*args, **kw)
    if T <= 2:
        b = [b[0], b[1], b[1], b[-1]]
    assert len(a) == len(b) == 4
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)


def test_training_draws_T_in_range_and_evaluation_uses_eval_iterations():
    m = _looped()
    calls = []
    m.shared_layer.register_forward_hook(lambda *_: calls.append(1))
    args, kw = _inputs()
    seen = set()
    m.train()
    for _ in range(60):
        calls.clear()
        _, objs = m(*args, **kw)
        assert len(calls) == m.last_iterations and len(objs) == 4
        seen.add(m.last_iterations)
    assert seen == set(range(2, 7))
    m.eval()
    calls.clear()
    m(*args, **kw)
    assert len(calls) == 4
    # the forward leaves the nominal 4-layer view in place for anything that inspects the module afterwards
    assert m.num_layers == 4 and m.layer_indices_to_save == {0, 1, 2, 4} and not hasattr(m, 'layers')


def test_bad_loop_settings_refused():
    with pytest.raises(ValueError):
        LoopedDecoder(**KW, loop=dict(LOOP, train_min_iterations=0))
    with pytest.raises(ValueError):
        LoopedDecoder(**dict(KW, layer_indices_to_save={0, 4}), loop=LOOP)


# readout 'last' (Yang et al. 2023): input injection, per-pass readout, loss window, truncated backpropagation
KW_LAST = dict(KW, block_attn_res=False, block_attn_res_final_aggregate=False)


def _last(**loop):
    torch.manual_seed(0)
    m = LoopedDecoder(**KW_LAST, loop=dict(LOOP, readout='last', **loop))
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m


def test_last_readout_puts_the_final_pass_in_every_slot():
    m = _last(eval_iterations=5).eval()
    args, kw = _inputs()
    _, objs = m(*args, **kw)
    assert len(objs) == 4 and all(o is objs[0] for o in objs) and objs[0].shape == (2, 5, D)


@pytest.mark.parametrize('T,K', [(5, 3), (2, 4), (6, 1)])
def test_training_stacks_the_last_K_passes_each_equal_to_evaluating_at_that_depth(T, K):
    m = _last(train_min_iterations=T, train_max_iterations=T, loss_window=K).eval()
    args, kw = _inputs()
    # training mode on the decoder alone, which picks T and the window; the layers stay deterministic (upstream's
    # layers are stochastic in training mode even at dropout 0)
    m.training = True
    _, objs = m(*args, **kw)
    K = min(K, T)
    assert m.last_window == K and objs[0].shape[0] == K * 2
    m.eval()
    for j, depth in enumerate(range(T - K + 1, T + 1)):
        m.eval_iterations = depth
        _, ref = m(*args, **kw)
        torch.testing.assert_close(objs[0][j * 2:(j + 1) * 2], ref[0])


def test_passes_before_the_window_run_without_gradient():
    def grads(K):
        m = _last(train_min_iterations=6, train_max_iterations=6, loss_window=K).eval()
        m.training = True
        args, kw = _inputs()
        m(*args, **kw)[1][0].sum().backward()
        return {n for n, p in m.shared_layer.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}

    m = _last(train_min_iterations=6, train_max_iterations=6, loss_window=2).train()
    grad = []
    m.shared_layer.register_forward_hook(lambda *_: grad.append(torch.is_grad_enabled()))
    args, kw = _inputs()
    _, objs = m(*args, **kw)
    assert grad == [False] * 4 + [True] * 2
    # the window still trains every parameter that full backpropagation does
    assert grads(2) and grads(2) == grads(6)


@pytest.mark.parametrize('normalize', [False, True])
def test_every_pass_after_the_first_adds_the_original_tokens_to_its_input(normalize):
    m = _last(eval_iterations=4, normalize=normalize).eval()
    ins, outs = [], []
    m.shared_layer.register_forward_pre_hook(lambda mod, a: ins.append(a[1]))
    m.shared_layer.register_forward_hook(lambda mod, a, o: outs.append(o[1]))
    args, kw = _inputs()
    m(*args, **kw)
    assert len(ins) == 4
    for k in range(1, 4):
        prev = m.loop_norm(outs[k - 1]) if normalize else outs[k - 1]
        torch.testing.assert_close(ins[k], prev + ins[0])


def test_normalized_state_stays_bounded_over_many_passes():
    def last_input_rms(normalize):
        m = _last(eval_iterations=60, normalize=normalize).eval()
        # amplify the layer's update so an unnormalised state grows from pass to pass, as it did in training
        with torch.no_grad():
            for p in m.shared_layer.parameters():
                p.mul_(3.0)
        ins = []
        m.shared_layer.register_forward_pre_hook(lambda mod, a: ins.append(a[1].pow(2).mean().sqrt().item()))
        args, kw = _inputs()
        m(*args, **kw)
        return ins[-1] / ins[0]
    unbounded = last_input_rms(False)
    assert unbounded != unbounded or unbounded > 100          # NaN counts: the toy model overflows in fp32
    assert last_input_rms(True) < 20


def test_last_readout_needs_attn_res_off_and_slots_needs_window_1():
    with pytest.raises(ValueError):
        LoopedDecoder(**KW, loop=dict(LOOP, readout='last'))
    with pytest.raises(ValueError):
        LoopedDecoder(**KW, loop=dict(LOOP, loss_window=2))
    with pytest.raises(ValueError):
        LoopedDecoder(**KW, loop=dict(LOOP, normalize=True))


def test_predictor_hook_repeats_its_other_inputs_per_pass():
    from src.models.looped_decoder import pass_stacking_hook

    class Pred(torch.nn.Module):
        def forward(self, multi_scale_features, query, ref_pose=None, multi_scale_state_features=None, obj_mask=None):
            return multi_scale_features[-1].sum(-1, keepdim=True) + query + ref_pose

    p = Pred()
    p.register_forward_pre_hook(pass_stacking_hook, with_kwargs=True)
    q, r = torch.randn(2, 5, 1), torch.randn(2, 5, 1)
    feats = torch.randn(3 * 2, 5, D)
    out = p([feats] * 4, q, ref_pose=r, multi_scale_state_features=None, obj_mask=torch.ones(2, 5, dtype=torch.bool))
    torch.testing.assert_close(out, feats.sum(-1, keepdim=True) + q.repeat(3, 1, 1) + r.repeat(3, 1, 1))
    same = torch.randn(2, 5, D)
    torch.testing.assert_close(p([same] * 4, q, ref_pose=r), same.sum(-1, keepdim=True) + q + r)


def test_loss_batch_repeats_only_what_the_loss_reads():
    from src.models.looped_decoder import repeat_batch_for_passes
    B = 4
    batch = {'obj_positions': torch.randn(B, 8, 3, 6, 3), 'valid_obj_mask': torch.ones(B, 3, dtype=torch.bool),
             'ref_obj_positions': torch.randn(B, 6, 3), 'dataset_name': ['a'] * B}
    out = repeat_batch_for_passes(batch, {'obj_positions_norm': torch.zeros(3 * B, 6)})
    assert out['obj_positions'].shape[0] == 3 * B and out['valid_obj_mask'].shape[0] == 3 * B
    torch.testing.assert_close(out['obj_positions'][2 * B:], batch['obj_positions'])
    assert out['ref_obj_positions'] is batch['ref_obj_positions'] and out['dataset_name'] is batch['dataset_name']
