from hypothesis import given, strategies as st, settings, assume
from microgpt import Value
import math


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


@given(st.integers(min_value=50, max_value=100))
@settings(deadline=None, max_examples=2)
def test_large_multidim_backward_pylist(n):
    # Create large 2D python lists
    import random

    a = [[random.uniform(1, 3) for _ in range(n)] for _ in range(n)]
    b = [[random.uniform(1, 3) for _ in range(n)] for _ in range(n)]
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
