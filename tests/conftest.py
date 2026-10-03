import numpy as np
import pytest

from tonematch.audio import SR, synth_di
from tonematch.chain import Renderer
from tonematch.mock import mock_catalog


@pytest.fixture(scope="session")
def di():
    return synth_di(SR, 6.0, seed=3)


@pytest.fixture(scope="session")
def catalog():
    return mock_catalog()


@pytest.fixture(scope="session")
def renderer(catalog):
    return Renderer(catalog)
