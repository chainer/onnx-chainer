OPSET_VERSION = 6

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
    'ConvolutionND': 'Conv',
    'Deconvolution2DFunction': 'ConvTranspose',
    'DeconvolutionND': 'ConvTranspose',
    'EmbedIDFunction': 'Gather',
    'LinearFunction': 'Gemm',

    # Math
    'Add': 'Add',
    'Absolute': 'Abs',
    'Div': 'Div',
    'Mul': 'Mul',
    'Neg': 'Neg',
    'PowVarConst': 'Pow',
    'Sub': 'Sub',
    'Clip': 'Clip',
    'Exp': 'Exp',
    'Identity': 'Identity',
    'MatMul': 'MatMul',
    'Maximum': 'Max',
    'Minimum': 'Min',
    'Sqrt': 'Sqrt',
    'Sum': 'ReduceSum',

    # Noise
    'Dropout': 'Dropout',

    # Pooling
    'AveragePooling2D': 'AveragePool',
    'AveragePoolingND': 'AveragePool',
    'MaxPooling2D': 'MaxPool',
    'MaxPoolingND': 'MaxPool',

    # Normalization
    'BatchNormalization': 'BatchNormalization',
    'FixedBatchNormalization': 'BatchNormalization',
    'LocalResponseNormalization': 'LRN',

}
