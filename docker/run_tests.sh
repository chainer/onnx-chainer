#!/bin/bash

run_test () {
    python_version=$1
    chainer_version=$2
    tag_name=python${python_version}-chainer${chainer_version}
    docker run --rm \
    -v $PWD:/root/onnx-chainer \
    -v $PWD/.chainer:/root/.chainer \
    -ti mitmul/onnx-chainer:${tag_name} \
    bash -c "cd /root/onnx-chainer && python setup.py develop && py.test -x -s -vvvs tests/"
}

run_test 2.7 3.5.0
run_test 3.5 3.5.0
run_test 3.6 3.5.0

run_test 2.7 4.2.0
run_test 3.5 4.2.0
run_test 3.6 4.2.0

