import pytest


def add(a: int, b: int) -> int:
    return a + b


def test_add():
    assert add(2, 3) == 5


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (2, 3, 5),
        (0, 0, 0),
        (-1, 1, 0),
    ],
)
def test_addition(a: int, b: int, expected: int):
    assert a + b == expected


@pytest.mark.slow
def test_big_computation():
    result = sum(range(10_000_000))  # something heavy
    assert result > 0


@pytest.mark.parametrize(
    "a,b",
    [
        (2, 2),
        pytest.param(2, 0, marks=pytest.mark.xfail(reason="division by zero")),
    ],
)
def test_division(a: int, b: int):
    assert a / b == a // b
