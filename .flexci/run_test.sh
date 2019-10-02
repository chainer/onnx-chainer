#! /usr/bin/env sh
set -eux

cat <<EOM >test_script.sh
set -eux

pip install ${CHAINER_INSTALL} cupy-cuda101
pip install ${CHAINER_INSTALL} chainer
pip install -U -e .[flexci]
pytest -x -s -vvvs tests/ --cov onnx_chainer

. .flexci/run_example.sh "${EXAMPLE_ARGS}"
EOM

docker run --runtime=nvidia -i --rm \
    -v $(pwd):/root/onnx-chainer --workdir /root/onnx-chainer \
    disktnk/onnx-chainer:ci-py${PYTHON_VER} \
    /bin/bash /root/onnx-chainer/test_script.sh
