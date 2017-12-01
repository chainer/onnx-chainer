from setuptools import setup

setup(
    name='onnx-chainer',
    packages=[
        'onnx_chainer',
        'onnx_chainer.functions',
        'onnx_chainer.functions.activation',
        'onnx_chainer.functions.array',
        'onnx_chainer.functions.connection',
        'onnx_chainer.functions.math',
        'onnx_chainer.functions.noise',
        'onnx_chainer.functions.normalization',
        'onnx_chainer.functions.pooling',
    ],
    version='0.2.1b4',
    description='ONNX support for Chainer',
    author='Shunta Saito',
    author_email='shunta@preferred.jp',
    url='https://github.com/mitmul/onnx-chainer',
    keywords='ONNX Chainer model converter deep learning',
    install_requires=[
        'chainer>=3.1.0',
        'onnx==0.2.1'
    ],
    tests_require=[
        'chainer>=3.1.0',
        'onnx==0.2.1',
        'onnx-caffe2',
        'numpy',
    ],
    license='MIT',
)
