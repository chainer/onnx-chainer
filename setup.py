from setuptools import setup

setup(
    name='onnx-chainer',
    version='0.2.1',
    description='ONNX support for Chainer',
    url='https://github.com/mitmul/onnx-chainer',
    keywords='ONNX Chainer model converter deep learning',
    install_requires=['chainer>=2.0.0', 'onnx>=0.2.1'],
    tests_require=['mxnet>=0.11.0', 'onnx>=0.2', 'numpy'],
    include_package_data=True,
    license='Apache 2.0'
)
