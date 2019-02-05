import pkg_resources

from onnx_chainer.export import convert_parameter  # NOQA
from onnx_chainer.export import export  # NOQA

from onnx_chainer.export import MINIMUM_OPSET_VERSION  # NOQA


__version__ = pkg_resources.get_distribution('onnx-chainer').version
