import sys

import pytest


@pytest.mark.skip(reason="API not available in test environment")
def test_external_api():
    assert 1 == 1


@pytest.mark.skipif(sys.platform == "win32", reason="Doesn't run on Windows")
def test_unix_only():
    assert 2 + 2 == 4
