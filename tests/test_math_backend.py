from diffaultpy.math_backend import MathBackend


def test_math_backend_numpy():
    backend = MathBackend()
    assert backend.backend == 'numpy'
    assert backend.pi == 3.141592653589793
    assert backend.array([1, 2, 3])[0] == 1
    assert backend.linspace(0, 1, 10)[0] == 0
    assert backend.arange(0, 10, 1)[0] == 0
    assert backend.ones((3, 3))[0][0] == 1
    assert backend.zeros((3, 3))[0][0] == 0
    assert backend.max(backend.array([1, 2, 3])) == 3
    assert backend.min(backend.array([1, 2, 3])) == 1
    assert backend.roll(backend.array([1, 2, 3]), 1, 0)[0] == 3
    return

def test_math_backend_torch():
    backend = MathBackend(backend='torch')
    assert backend.backend == 'torch'
    assert backend.pi == 3.141592653589793
    assert backend.array([1, 2, 3])[0] == 1
    assert backend.linspace(0, 1, 10)[0] == 0
    assert backend.arange(0, 10, 1)[0] == 0
    assert backend.ones((3, 3))[0][0] == 1
    assert backend.zeros((3, 3))[0][0] == 0
    assert backend.max(backend.array([1, 2, 3])) == 3
    assert backend.min(backend.array([1, 2, 3])) == 1
    assert backend.roll(backend.array([1, 2, 3]), 1, 0)[0] == 3
    return

def test_math_backend_unsupported():
    try:
        backend = MathBackend(backend='generic')
    except ValueError as e:
        assert str(e) == "Unsupported backend"
    return