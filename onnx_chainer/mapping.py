TENSOR_TYPE_TO_NAME = {
    0: 'UNDEFINED',
    1: 'FLOAT',
    2: 'UINT8',
    3: 'INT8',
    4: 'UINT16',
    5: 'INT16',
    6: 'INT32',
    7: 'INT64',
    8: 'STRING',
    9: 'BOOL',
    10: 'FLOAT16',
    11: 'DOUBLE',
    12: 'UINT32',
    13: 'UINT64',
    14: 'COMPLEX64',
    15: 'COMPLEX128',
}

# Chainer Function -> Operator set IDs
operators = {
    # Activation
    'ClippedReLU': (1, 6),
    'ELU': (1, 6),
    'HardSigmoid': (1, 6),
    'LeakyReLU': (1, 6),
    'LogSoftmax': (1,),
    'PReLUFunction': (1, 6, 7),
    'ReLU': (1, 6),
    'Sigmoid': (1, 6),
    'Softmax': (1,),
    'Softplus': (1,),
    'Tanh': (1, 6),

    # Array
    'Cast': (1, 6),
    'Concat': (1, 4),
    'Copy': (1,),
    'Depth2Space': (1,),
    'ExpandDims': (1,),
    'GetItem': (1,),
    'Pad': (1, 2),
    'Reshape': (1, 5),
    'Space2Depth': (1,),
    'SplitAxis': (1, 2),
    'Squeeze': (1,),
    'Tile': (1, 6),
    'Transpose': (1,),

    # Connection
    'Convolution2DFunction': (1,),
    'ConvolutionND': (1,),
    'Deconvolution2DFunction': (1,),
    'DeconvolutionND': (1,),
    'EmbedIDFunction': (1,),
    'LinearFunction': (1, 6, 7),

    # Math
    'Add': (1, 6, 7),
    'AddConstant': (1, 6, 7),
    'Absolute': (1, 6),
    'BroadcastTo': (8,),
    'Div': (1, 6, 7),
    'Mul': (1, 6, 7),
    'MulConstant': (1, 6, 7),
    'Neg': (1, 6),
    'PowVarConst': (1, 7),
    'Sub': (1, 6, 7),
    'Clip': (1, 6),
    'Exp': (1, 6),
    'Identity': (1,),
    'MatMul': (1, 6, 7),
    'Maximum': (1, 6, 8),
    'Minimum': (1, 6, 8),
    'Sqrt': (1, 6),
    'LinearInterpolate': (1, 6, 7),
    'LogSumExp': (1,),
    'Max': (1,),
    'Mean': (1,),
    'Min': (1,),
    'Prod': (1,),
    'Sum': (1,),
    'Square': (1, 6, 7),

    # Noise
    'Dropout': (1, 6, 7),

    # Pooling
    'AveragePooling2D': (1, 7),
    'AveragePoolingND': (1, 7),
    'MaxPooling2D': (1, 8),
    'MaxPoolingND': (1, 8),
    'ROIPooling2D': (1,),
    'Unpooling2D': (7, 9),

    # Normalization
    'BatchNormalization': (1, 6, 7),
    'FixedBatchNormalization': (1, 6, 7),
    'LocalResponseNormalization': (1,),
    'NormalizeL2': (1,),

    # Loss
    'SoftmaxCrossEntropy': (9,),
}
