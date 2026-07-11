from app.services.vector_utils import to_float_list


class _NonIterableVector:
    def __init__(self, values):
        self._values = values

    def to_list(self):
        return self._values


class _NumpyStyleVector:
    def __init__(self, values):
        self._values = values

    def to_numpy(self):
        return self._values


def test_plain_iterable_vector_converts_to_floats():
    assert to_float_list(["1.5", 2, 3.25]) == [1.5, 2.0, 3.25]


def test_non_iterable_pgvector_style_to_list_converts_to_floats():
    value = _NonIterableVector(["0.1", 0.2, 3])

    assert to_float_list(value) == [0.1, 0.2, 3.0]


def test_to_numpy_style_vector_converts_to_floats():
    value = _NumpyStyleVector(("4", 5.5))

    assert to_float_list(value) == [4.0, 5.5]
