from setuptools import setup

setup(
    name='onnx-chainer',
    packages=['onnx_chainer'],
    version='0.2.1b1',
    description='ONNX support for Chainer',
    author='Shunta Saito',
    author_email='shunta@preferred.jp',
    url='https://github.com/mitmul/onnx-chainer',
    keywords='ONNX Chainer model converter deep learning',
    install_requires=[
        'chainer>=3.1.0',
        'onnx==0.2.1'
    ],
    tests_require=['chainer>=3.1.0', 'onnx==0.2.1', 'numpy'],
    license='MIT',
)
