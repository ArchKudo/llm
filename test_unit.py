from microgpt import Value
from hypothesis import given, strategies as st
import math


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_init_and_repr(x):
    v = Value(x)
    assert v.data == x
    assert v.grad == 0
    assert v._children == ()
    assert v._local_grads == ()
    assert f"Value(data={x}" in repr(v)


@given(
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
)
def test_add(a, b):
    v1 = Value(a)
    v2 = Value(b)
    v3 = v1 + v2
    assert math.isclose(v3.data, a + b)
    assert v3._children == (v1, v2)
    assert v3._local_grads == (1, 1)


@given(
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
)
def test_mul(a, b):
    v1 = Value(a)
    v2 = Value(b)
    v3 = v1 * v2
    assert math.isclose(v3.data, a * b)
    assert v3._children == (v1, v2)
    assert v3._local_grads == (b, a)


@given(
    st.floats(min_value=0.1, max_value=10),
    st.floats(min_value=0.1, max_value=3, exclude_min=True),
)
def test_pow(a, b):
    v1 = Value(a)
    v2 = v1**b
    assert math.isclose(v2.data, a**b)
    assert v2._children == (v1,)
    assert math.isclose(v2._local_grads[0], b * a ** (b - 1))


@given(st.floats(min_value=0.1, max_value=100))
def test_log(a):
    v1 = Value(a)
    v2 = v1.log()
    assert math.isclose(v2.data, math.log(a))
    assert v2._children == (v1,)
    assert math.isclose(v2._local_grads[0], 1 / a)


@given(st.floats(min_value=-744, max_value=709))
def test_exp(a):
    v1 = Value(a)
    v2 = v1.exp()
    assert math.isclose(v2.data, math.exp(a))
    assert v2._children == (v1,)
    assert math.isclose(v2._local_grads[0], math.exp(a))


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_relu(a):
    v1 = Value(a)
    v2 = v1.relu()
    assert math.isclose(v2.data, max(0, a))
    assert v2._children == (v1,)
    assert v2._local_grads[0] == float(a > 0)


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_neg(a):
    v1 = Value(a)
    v2 = -v1
    assert math.isclose(v2.data, -a)


@given(
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
)
def test_sub(a, b):
    v1 = Value(a)
    v2 = Value(b)
    v3 = v1 - v2
    assert math.isclose(v3.data, a - b)


@given(
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
)
def test_radd(a, b):
    v1 = Value(a)
    assert math.isclose((b + v1).data, b + a)


@given(
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
)
def test_rsub(a, b):
    v1 = Value(a)
    assert math.isclose((b - v1).data, b - a)


@given(
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
)
def test_rmul(a, b):
    v1 = Value(a)
    assert math.isclose((b * v1).data, b * a)


@given(st.floats(min_value=0.1, max_value=10), st.floats(min_value=0.1, max_value=10))
def test_truediv(a, b):
    v1 = Value(a)
    v2 = Value(b)
    v3 = v1 / v2
    assert math.isclose(v3.data, a / b)


@given(st.floats(min_value=0.1, max_value=10), st.floats(min_value=0.1, max_value=10))
def test_rtruediv(a, b):
    v1 = Value(a)
    assert math.isclose((b / v1).data, b / a)
