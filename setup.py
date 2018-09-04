from setuptools import setup

setup(
    name='onnx-chainer',
    packages=[
        'onnx_chainer',
        'onnx_chainer.functions',
        'onnx_chainer.testing',
    ],
    version='1.2.2a2',
    description='Convert a Chainer model into ONNX',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Shunta Saito',
    author_email='shunta@preferred.jp',
    url='https://github.com/chainer/onnx-chainer',
    keywords='ONNX Chainer model converter deep learning',
    install_requires=[
        'chainer>=3.2.0',
        'onnx>=1.2.1'
    ],
    tests_require=[
        'chainer>=3.2.0',
        'onnx>=1.2.1',
        'numpy',
    ],
    license='MIT License',
)
