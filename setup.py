from setuptools import setup


requirements = {
    'install': [
        'packaging>=19.0',
        'chainer>=5.0.0',
        'onnx>=1.4.0,<1.6',
    ],
    'stylecheck': [
        'autopep8',
        'hacking',
    ],
    'test': [
        'pytest',
        'chainercv>=0.11.0',
    ],
    'test-cpu': [
        '-r test',
        'onnxruntime==0.4.0',
    ],
    'test-gpu': [
        '-r test',
        # 'cupy',  # installed 'cupy-cudaXX' before
        # onnxruntime-gpu is better but prebuild version requires CUDA9.1
        # CUDA9.1 is not supported latest cuDNN, so decided to use CPU version
        'onnxruntime==0.4.0',
    ],
    'travis': [
        '-r stylecheck',
        '-r test-cpu',
        'pytest-cov',
        'codecov',
    ],
    'chainerci': [
        '-r test-gpu',
        'pytest-cov',
    ]
}


def reduce_requirements(key):
    # Resolve recursive requirements notation (-r)
    reqs = requirements[key]
    resolved_reqs = []
    for req in reqs:
        if req.startswith('-r'):
            depend_key = req[2:].lstrip()
            reduce_requirements(depend_key)
            resolved_reqs += requirements[depend_key]
        else:
            resolved_reqs.append(req)
    requirements[key] = resolved_reqs


for k in requirements.keys():
    reduce_requirements(k)


setup(
    name='onnx-chainer',
    packages=[
        'onnx_chainer',
        'onnx_chainer.functions',
        'onnx_chainer.testing',
    ],
    version='1.5.0',
    description='Convert a Chainer model into ONNX',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Shunta Saito',
    author_email='shunta@preferred.jp',
    url='https://github.com/chainer/onnx-chainer',
    keywords='ONNX Chainer model converter deep learning',
    install_requires=requirements['install'],
    tests_require=requirements['test'],
    extras_require={k: v for k, v in requirements.items() if k != 'install'},
    license='MIT License',
)
