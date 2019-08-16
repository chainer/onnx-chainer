#! /usr/bin/env sh
set -eux

docker run --runtime=nvidia -i --rm \
    -v $(pwd):/root/onnx-chainer --workdir /root/onnx-chainer \
    disktnk/onnx-chainer:ci-py${PYTHON_VER} \
    sh -ex << EOD
pip install ${CHAINER_INSTALL} cupy-cuda101
pip install ${CHAINER_INSTALL} chainer
pip install -U -e .[flexci]
pytest -x -s -vvvs tests/ --cov onnx_chainer
EOD
