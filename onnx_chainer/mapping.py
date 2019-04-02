from onnx_chainer import functions
from onnx_chainer.functions.converter import FunctionConverter


_supported_function_node_set = {
    # Activation
    'ClippedReLU',
    'ELU',
    'HardSigmoid',
    'LeakyReLU',
    'LogSoftmax',
    'PReLUFunction',
    'ReLU',
    'Sigmoid',
    'Softmax',
    'Softplus',
    'Tanh',

    # Array
    'Cast',
    'Concat',
    'Copy',
    'Depth2Space',
    'ExpandDims',
    'GetItem',
    'Pad',
    'Reshape',
    'Space2Depth',
    'SplitAxis',
    'Squeeze',
    'Tile',
    'Transpose',
    'Where',

    # Connection
    'Convolution2DFunction',
    'ConvolutionND',
    'Deconvolution2DFunction',
    'DeconvolutionND',
    'EmbedIDFunction',
    'LinearFunction',

    # Math
    'Add',
    'AddConstant',
    'Absolute',
    'BroadcastTo',
    'Div',
    'Mul',
    'MulConstant',
    'Neg',
    'PowVarConst',
    'Sub',
    'Clip',
    'Exp',
    'Identity',
    'MatMul',
    'Maximum',
    'Minimum',
    'Sqrt',
    'LinearInterpolate',
    'LogSumExp',
    'Max',
    'Mean',
    'Min',
    'Prod',
    'Sum',
    'Square',

    # Noise
    'Dropout',

    # Pooling
    'AveragePooling2D',
    'AveragePoolingND',
    'MaxPooling2D',
    'MaxPoolingND',
    'ROIPooling2D',
    'Unpooling2D',

    # Normalization
    'BatchNormalization',
    'FixedBatchNormalization',
    'LocalResponseNormalization',
    'NormalizeL2',

    # Loss
    'SoftmaxCrossEntropy',
}

_converters = None


def _get_converters():
    global _converters

    if _converters is not None:
        return _converters

    _converters = {
        name: FunctionConverter(getattr(functions, 'convert_'+name, None))
        for name in _supported_function_node_set}
    return _converters


converters = _get_converters()
