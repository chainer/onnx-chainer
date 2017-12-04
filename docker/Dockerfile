FROM ubuntu:trusty

MAINTAINER Shunta Saito <shunta.saito@gmail.com>

RUN apt-get update
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgoogle-glog-dev \
    libprotobuf-dev \
    protobuf-compiler

RUN curl -L https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86_64.sh -o miniconda.sh
RUN bash miniconda.sh -b -p miniconda
ENV PATH /miniconda/bin:$PATH
ENV LD_LIBRARY_PATH /miniconda/lib:$LD_LIBRARY_PATH
ENV CPATH /miniconda/include:$CPATH

RUN conda config --set always_yes yes --set changeps1 no
RUN conda install -c conda-forge protobuf
RUN conda install -c conda-forge numpy 
RUN conda install -c conda-forge scipy 
RUN conda install -c conda-forge cmake

RUN pip install chainer==3.1.0
RUN pip install future
RUN pip install hypothesis
RUN pip install flake8
RUN pip install hacking
RUN pip install autopep8
RUN pip install mock
RUN pip install pytest
RUN pip install six
RUN pip install git+git://github.com/onnx/onnx.git@8d0fc40162e0df1d14a0eba9e9696568153e54fd

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
    -DPYTHON_EXECUTABLE=/miniconda/bin/python \
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
RUN pip install git+git://github.com/onnx/onnx-caffe2@7aaf817ecf1c4ec34d881633bc494f7f00d2d6b5

# Install MXNet
RUN pip install mxnet
RUN cd $HOME && git clone https://github.com/onnx/onnx-mxnet.git && cd onnx-mxnet && python setup.py install

# Install NNVM
RUN cd $HOME && git clone --recursive https://github.com/dmlc/nnvm \
    && cd nnvm && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=../install .. \
    && make -j$(nproc) install \
    && cd ../python && python setup.py install

# Install TVM
RUN cd $HOME && git clone --recursive https://github.com/dmlc/tvm \
    && cd tvm && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=../install .. \
    && make -j$(nproc) install \
    && cd ../python && python setup.py install \
    && cd $HOME/tvm/topi && python setup.py install