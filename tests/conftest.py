import chainer
import pytest


@pytest.fixture(scope='function')
def desable_experimental_warning():
    org_config = chainer.disable_experimental_feature_warning
    chainer.disable_experimental_feature_warning = True
    try:
        yield
    finally:
        chainer.disable_experimental_feature_warning = org_config
