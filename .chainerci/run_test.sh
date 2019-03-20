#! /usr/bin/env sh
set -eux

docker run --runtime=nvidia -i --rm \
    -v $(pwd):/root/onnx-chainer --workdir /root/onnx-chainer \
    disktnk/onnx-chainer:ci-py${PYTHON_VER} \
    sh -ex << EOD
pip install ${CHAINER_INSTALL} cupy-cuda100
pip install ${CHAINER_INSTALL} chainer
pip install -U -e .[chainerci]
pytest -x -s -vvvs tests/ --cov onnx_chainer
EOD
