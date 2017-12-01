FROM ubuntu:trusty

MAINTAINER Shunta Saito <shunta.saito@gmail.com>

RUN apt-get update
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    cmake \
    git \
    libgoogle-glog-dev \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86_64.sh -o miniconda.sh
RUN bash miniconda.sh -b -p miniconda
ENV PATH /miniconda/bin:$PATH
ENV LD_LIBRARY_PATH /miniconda/lib:$LD_LIBRARY_PATH
ENV CPATH /miniconda/include:$CPATH

RUN conda config --set always_yes yes --set changeps1 no
RUN conda install -c conda-forge protobuf
RUN conda install -c conda-forge numpy 

RUN pip install chainer==3.1.0
RUN pip install future
RUN pip install hypothesis
RUN pip install flake8
RUN pip install hacking
RUN pip install autopep8
RUN pip install mock
RUN pip install pytest
RUN pip install six
RUN pip install git+git://github.com/onnx/onnx.git@a8bd1facd321ea52f0a6f88600138606711b7a96

RUN git clone --recursive https://github.com/caffe2/caffe2.git
RUN cd caffe2 && mkdir build && cd build \
    && cmake .. \
    -DUSE_CUDA=OFF \
    -DUSE_NCCL=OFF \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    -DUSE_LEVELDB=OFF \
    -DUSE_LMDB=OFF \
    -DUSE_OPENCV=OFF \
    -DBUILD_TEST=OFF \
    -DBUILD_BENCHMARK=OFF \
    -DPYTHON_INCLUDE_DIR=/miniconda/include \
    -DPYTHON_INCLUDE_DIR2=/miniconda/include/python3.6m \
    -DPYTHON_LIBRARY=/miniconda/lib/libpython3.6m.so \
    -DPYTHON_LIBRARY_DEBUG=/miniconda/lib/libpython3.6m.so \
    -DPROTOBUF_INCLUDE_DIR=/miniconda/include \
    -DPROTOBUF_LIBRARY=/miniconda/lib/libprotobuf.so \
    -DPROTOBUF_LIBRARY_DEBUG=/miniconda/lib/libprotobuf.so \
    -DPROTOBUF_LITE_LIBRARY=/miniconda/lib/libprotobuf-lite.so \
    -DPROTOBUF_LITE_LIBRARY_DEBUG=/miniconda/lib/libprotobuf-lite.so \
    -DPROTOBUF_PROTOC_LIBRARY=/miniconda/lib/libprotoc.so \
    -DPROTOBUF_PROTOC_LIBRARY_DEBUG=/miniconda/lib/libprotoc.so \
    && make -j"$(nproc)" install \
    && ldconfig \
    && make clean \
    && cd .. \
    && rm -rf build
ENV PYTHONPATH /usr/local/
RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

