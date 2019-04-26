import chainer
import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--value-check-runtime',
        dest='value-check-runtime', default='onnxruntime',
        choices=['skip', 'onnxruntime', 'mxnet'], help='select test runtime')


@pytest.fixture(scope='function')
def disable_experimental_warning():
    org_config = chainer.disable_experimental_feature_warning
    chainer.disable_experimental_feature_warning = True
    try:
        yield
    finally:
        chainer.disable_experimental_feature_warning = org_config
