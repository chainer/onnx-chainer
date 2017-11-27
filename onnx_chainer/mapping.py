from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

dtypes = {v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items()}

operators = {
    # Activation
    'ELU': 'Elu',
    'HardSigmoid': 'HardSigmoid',
    'LeakyReLU': 'LeakyRelu',
    'LogSoftmax': 'LogSoftmax',
    'PReLUFunction': 'PRelu',
    'ReLU': 'Relu',
    'Sigmoid': 'Sigmoid',
    'Softmax': 'Softmax',
    'Softplus': 'Softplus',
    'Tanh': 'Tanh',

    # Array
    'Cast': 'Cast',
    'Concat': 'Concat',
    'Depth2Space': 'DepthToSpace',
    'Pad': 'Pad',
    'Reshape': 'Reshape',
    'Space2Depth': 'SpaceToDepth',
    'SplitAxis': 'Split',
    'Squeeze': 'Squeeze',
    'Tile': 'Tile',
    'Transpose': 'Transpose',

    # Connection
    'Convolution2DFunction': 'Conv',
    'LinearFunction': 'FC',

    # Pooling
    'AveragePooling2D': 'AveragePool',
    'MaxPooling2D': 'MaxPool',

    # Normalization
    'BatchNormalization': 'BatchNormalization',
    'FixedBatchNormalization': 'BatchNormalization',

    # Math
    'Add': 'Add',
    'Sub': 'Sub',
    'Mul': 'Mul',
    'Neg': 'Neg',
    'Absolute': 'Abs',
    'Div': 'Div',
}
