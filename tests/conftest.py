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


@pytest.fixture(scope='function')
def check_model_expect(request):
    selected_runtime = request.config.getoption('value-check-runtime')
    if selected_runtime == 'onnxruntime':
        from onnx_chainer.testing.test_onnxruntime import check_model_expect  # NOQA
        _checker = check_model_expect
    elif selected_runtime == 'mxnet':
        from onnx_chainer.testing.test_mxnet import check_model_expect
        _checker = check_model_expect
    else:
        def empty_func(*args, **kwargs):
            pass
        _checker = empty_func
    return _checker
