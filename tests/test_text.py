import pytest


def normalize(s: str) -> str:
    """Trim whitespace and make lowercase."""
    return s.strip().lower()


@pytest.mark.parametrize(
    "raw, expected",
    [
        (" Hello ", "hello"),  # trims spaces
        ("WORLD", "world"),  # makes lowercase
        ("python", "python"),  # already fine
        ("\tMixED CaSe\n", "mixed case"),  # trims tabs/newlines, lowercases
    ],
)
def test_normalize(raw: str, expected: str):
    assert normalize(raw) == expected
