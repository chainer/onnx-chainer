#! /usr/bin/env sh
set -eux

export INSTALL_CUPY=""
export EXAMPLE_ARGS=""
export DOCKER_RUNTIME_ARG=""
export PYTEST_ARGS="-m \"not gpu\""
if [ -n "${GPU+x}" ]; then
    export INSTALL_CUPY="on"
    export EXAMPLE_ARGS="-G "${GPU}
    export DOCKER_RUNTIME_ARG="--runtime=nvidia"
    export PYTEST_ARGS=""
fi
if [ -z "${ONNX_VER+x}" ]; then export ONNX_VER=""; fi

cat <<EOM >test_script.sh
set -eux

if [[ "${INSTALL_CUPY}" == "on" ]]; then pip install 'cupy-cuda101<7.0.0'; fi
pip install 'chainer<7.0.0'
pip install -e .[flexci]
if [[ "${ONNX_VER}" != "" ]]; then pip install onnx==${ONNX_VER}; fi
pip list -v
pytest -x -s -vvvs ${PYTEST_ARGS} tests/ --cov onnx_chainer

pip install chainercv
export CHAINERCV_DOWNLOAD_REPORT="OFF"
for dir in \`ls examples\`
do
  if [[ -f examples/\${dir}/export.py ]]; then
    python examples/\${dir}/export.py -T ${EXAMPLE_ARGS}
  fi
done
EOM

docker run ${DOCKER_RUNTIME_ARG} -i --rm \
    -v $(pwd):/root/onnx-chainer --workdir /root/onnx-chainer \
    disktnk/onnx-chainer:ci-py${PYTHON_VER} \
    /bin/bash /root/onnx-chainer/test_script.sh
