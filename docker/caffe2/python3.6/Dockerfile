FROM mitmul/python:3.6

MAINTAINER Shunta Saito <shunta.saito@gmail.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    wget \
    git \
    libgflags-dev \
    libgtest-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libsnappy-dev \
    libprotobuf-dev \
    libprotoc-dev \
    libgoogle-glog-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir future hypothesis numpy protobuf six
RUN pip install --no-cache-dir \
    flask \
    graphviz \
    jupyter \
    matplotlib \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    tornado

RUN git clone -b v0.8.1-fixed --recursive https://github.com/mitmul/caffe2.git
RUN cd caffe2 && mkdir build && cd build && \
    cmake .. \
    -DUSE_CUDA=OFF \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    -DUSE_MPI=OFF \
    -DUSE_GLOO=OFF \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
    -DPYTHON_EXECUTABLE=/usr/bin/python \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so && \
    make -j"$(nproc)" install && \
    ldconfig && \
    make clean && \
    cd .. && \
    rm -rf build
ENV PYTHONPATH /usr/local
ENV PYTHONPATH /usr/local/lib/python3/dist-packages
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
