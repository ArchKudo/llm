import math
from hypothesis import given, strategies as st, settings, assume
from microgpt import (
    linear,
    softmax,
    rmsnorm,
    gpt,
    matrix,
    state,
    N_EMBED,
    BLOCK_SZ,
    Value,
)


@given(st.floats(min_value=0.1, max_value=10), st.floats(min_value=0.1, max_value=10))
def test_chain_operations(a, b):
    v1 = Value(a)
    v2 = Value(b)
    v3 = ((v1 + v2) * v1).exp() / v2
    expected = math.exp((a + b) * a) / b
    assert math.isclose(v3.data, expected)


@given(st.floats(min_value=0.1, max_value=10), st.floats(min_value=0.1, max_value=10))
def test_backward_simple_add(a, b):
    v1 = Value(a)
    v2 = Value(b)
    v3 = v1 + v2
    v3.backward()
    assert math.isclose(v1.grad, 1)
    assert math.isclose(v2.grad, 1)


@given(st.floats(min_value=0.1, max_value=10), st.floats(min_value=0.1, max_value=10))
def test_backward_simple_mul(a, b):
    v1 = Value(a)
    v2 = Value(b)
    v3 = v1 * v2
    v3.backward()
    assert math.isclose(v1.grad, b)
    assert math.isclose(v2.grad, a)


@given(st.floats(min_value=0.1, max_value=10))
def test_backward_exp(a):
    v1 = Value(a)
    v2 = v1.exp()
    v2.backward()
    assert math.isclose(v1.grad, math.exp(a))


@given(st.floats(min_value=0.1, max_value=10))
def test_backward_log(a):
    v1 = Value(a)
    v2 = v1.log()
    v2.backward()
    assert math.isclose(v1.grad, 1 / a)


@given(st.floats(min_value=0.1, max_value=10))
def test_backward_pow(a):
    v1 = Value(a)
    v2 = v1**2
    v2.backward()
    assert math.isclose(v1.grad, 2 * a)


@given(st.floats(min_value=-10, max_value=10))
def test_backward_relu(a):
    v1 = Value(a)
    v2 = v1.relu()
    v2.backward()
    assert math.isclose(v1.grad, float(a > 0))


@given(st.floats(min_value=0.1, max_value=10), st.floats(min_value=0.1, max_value=10))
def test_backward_composed(a, b):
    v1 = Value(a)
    v2 = Value(b)
    v3 = ((v1 * v2) + v1.log() - v2.exp()) / v1
    v3.backward()
    assert math.isfinite(v1.grad)
    assert math.isfinite(v2.grad)


@given(st.integers(min_value=50, max_value=1000))
@settings(deadline=None, max_examples=9)
def test_large_multidim_backward_pylist(n):
    # Create large 2D python lists
    import random

    a = [[random.uniform(1, 10) for _ in range(n)] for _ in range(n)]
    b = [[random.uniform(1, 10) for _ in range(n)] for _ in range(n)]
    # Convert to Value lists
    va = [[Value(x) for x in row] for row in a]
    vb = [[Value(x) for x in row] for row in b]
    # Compose all primitive operations elementwise
    vadd = [[x + y for x, y in zip(rowa, rowb)] for rowa, rowb in zip(va, vb)]
    vmul = [[x * y for x, y in zip(rowa, rowb)] for rowa, rowb in zip(va, vb)]
    vsub = [[x - y for x, y in zip(rowa, rowb)] for rowa, rowb in zip(va, vb)]
    vdiv = [[x / y for x, y in zip(rowa, rowb)] for rowa, rowb in zip(va, vb)]
    vpow = [[x**2.0 for x in row] for row in va]
    vlog = [[x.log() for x in row] for row in va]
    vexp = [[x.exp() for x in row] for row in va]
    vrelu = [[x.relu() for x in row] for row in va]
    # Flatten all results into a single list
    all_vals = [vadd, vmul, vsub, vdiv, vpow, vlog, vexp, vrelu]
    flat = [item for arr in all_vals for row in arr for item in row]
    assume(len(flat) > 0)
    result = flat[0]
    for v in flat[1:]:
        result = result + v
    # Backward pass
    result.backward()
    # Check all grads are finite
    grads = [v.grad for v in flat]
    assert all(math.isfinite(g) for g in grads)


def test_state_matrix_shapes():
    vocab_sz = 1000
    state_dict, state_params = state(vocab_sz)
    # Check embedding shapes
    assert len(state_dict["wte"]) == vocab_sz
    assert len(state_dict["wte"][0]) == N_EMBED
    assert len(state_dict["wpe"]) == BLOCK_SZ
    assert len(state_dict["wpe"][0]) == N_EMBED
    assert len(state_dict["lm_head"]) == vocab_sz
    assert len(state_dict["lm_head"][0]) == N_EMBED
    # Check at least one layer's attention and MLP weights
    for key in state_dict:
        for row in state_dict[key]:
            for elem in row:
                assert isinstance(elem, Value)
    assert isinstance(state_params, list)
    assert all(isinstance(p, Value) for p in state_params)


@given(st.integers(min_value=1, max_value=32), st.integers(min_value=1, max_value=32))
def test_matrix_in_state_integration(nout, nin):
    mat = matrix(nout, nin)
    # Simulate adding to a state dict and flattening
    state_dict = {"test": mat}
    state_params = [elem for row in state_dict.values() for row in mat for elem in row]
    assert len(state_params) == nout * nin
    assert all(isinstance(p, Value) for p in state_params)


@settings(deadline=None)
@given(
    st.integers(min_value=1, max_value=2000), st.integers(min_value=1, max_value=2000)
)
def test_linear_large(nout, nin):
    vec = [Value(float(i)) for i in range(nin)]
    weight = [[Value(float(i * nin + j)) for j in range(nin)] for i in range(nout)]
    out = linear(vec, weight)
    assert len(out) == nout
    for v in out:
        assert isinstance(v, Value)
    # Manual calculation for correctness
    for i, wo in enumerate(weight):
        expected = sum(wo[j].data * vec[j].data for j in range(nin))
        assert abs(out[i].data - expected) < 1e-6


@settings(deadline=None)
@given(st.lists(st.floats(min_value=-100, max_value=100), min_size=1, max_size=2000))
def test_softmax_large(vals):
    vals = [Value(v) for v in vals]
    out = softmax(vals)
    assert len(out) == len(vals)
    total = sum(v.data for v in out)
    assert abs(total - 1.0) < 1e-6
    for v in out:
        assert v.data >= 0


@settings(deadline=None)
@given(st.lists(st.floats(min_value=-100, max_value=100), min_size=1, max_size=2000))
def test_rmsnorm_large(vals):
    vals = [Value(v) for v in vals]
    out = rmsnorm(vals)
    assert len(out) == len(vals)


def test_gpt_large():
    # Use large vocab and block size for stress test
    large_vocab = 2000
    large_block = 16
    large_embed = 16
    large_layer = 4
    # Build dummy state_dict
    dummy_state = {
        "wte": matrix(large_vocab, large_embed),
        "wpe": matrix(large_block, large_embed),
        "lm_head": matrix(large_vocab, large_embed),
    }
    for i in range(large_layer):
        dummy_state[f"layer{i}.attn_wq"] = matrix(large_embed, large_embed)
        dummy_state[f"layer{i}.attn_wk"] = matrix(large_embed, large_embed)
        dummy_state[f"layer{i}.attn_wv"] = matrix(large_embed, large_embed)
        dummy_state[f"layer{i}.attn_wo"] = matrix(large_embed, large_embed)
        dummy_state[f"layer{i}.mlp_fc1"] = matrix(4 * large_embed, large_embed)
        dummy_state[f"layer{i}.mlp_fc2"] = matrix(large_embed, 4 * large_embed)
    keys = [[] for _ in range(large_layer)]
    vals = [[] for _ in range(large_layer)]
    # Try a few positions and tokens
    for tokid in range(3):
        for posid in range(3):
            logits = gpt(tokid, posid, keys, vals, dummy_state)
            assert isinstance(logits, list)
            assert all(isinstance(val, Value) for val in logits)
            assert len(logits) == large_vocab
