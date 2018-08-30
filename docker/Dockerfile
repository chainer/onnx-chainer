FROM ubuntu:16.04

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
    python-dev \
    python-pip \
    python-dbg \
    python3-dev \
    python3-pip \
    python3-dbg \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev

RUN apt-get purge -y cmake
RUN curl -L -O https://cmake.org/files/v3.11/cmake-3.11.1-Linux-x86_64.sh && \
    bash cmake-3.11.1-Linux-x86_64.sh --skip-license && rm -rf cmake-3.11.1-Linux-x86_64.sh

WORKDIR /root

# Build LLVM
RUN cd /root && \
    curl -L -O http://releases.llvm.org/6.0.1/llvm-6.0.1.src.tar.xz && \
    tar xf llvm-6.0.1.src.tar.xz && rm -rf llvm-6.0.1.src.tar.xz && \
    cd /root/llvm-6.0.1.src && \
    mkdir build && cd build && \
    cmake .. && \
    cmake --build . -- -j16 && \
    cmake --build . --target install && \
    rm -rf /root/llvm-6.0.1.src

# Install pyenv
RUN curl -L -O https://github.com/pyenv/pyenv/archive/v1.2.4.tar.gz && \
    tar zxvf v1.2.4.tar.gz && rm -rf v1.2.4.tar.gz && \
    mv pyenv-1.2.4 .pyenv

ENV PYENV_ROOT=/root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN eval "$(pyenv init -)"

ARG PYTHON_VERSION="3.5"

RUN if [ ${PYTHON_VERSION} = "3.6" ]; then \
        pyenv install 3.6.5; \
        pyenv global 3.6.5; \
        pyenv rehash; \
    elif [ ${PYTHON_VERSION} = "3.5" ]; then \
        pyenv install 3.5.5; \
        pyenv global 3.5.5; \
        pyenv rehash; \
    elif [ ${PYTHON_VERSION} = "2.7" ]; then \
        pyenv install 2.7.14; \
        pyenv global 2.7.14; \
        pyenv rehash; \
    fi

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
    tornado \
    pytest \
    future \
    hypothesis 

ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/lib:$LIBRARY_PATH

# Install MXNet
ARG MXNET_VERSION="pre"
RUN if [ ${MXNET_VERSION} = "pre" ]; then \
        pip install mxnet --pre; \
    else
        pip install mxnet==${MXNET_VERSION};
    fi

# Install TVM
ARG TVM_VERSION="0.4"
RUN git clone --recursive https://github.com/dmlc/tvm && \
    cd tvm && git checkout -b tag refs/tags/v0.4 && \
    mkdir build && cd build \
    && cmake -DUSE_LLVM=ON .. \
    && make -j$(nproc) install
RUN cd $HOME/tvm/python && python setup.py install
RUN cd $HOME/tvm/topi/python && python setup.py install
RUN cd $HOME/tvm/nnvm/python && python setup.py install

# Install Chainer
ARG CHAINER_VERSION="4.4.0"
RUN pip install chainer==${CHAINER_VERSION}

# Install ChainerCV
ARG CHAINERCV_VERSION="0.10.0"
RUN pip install chainercv==${CHAINERCV_VERSION}
