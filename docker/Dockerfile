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

# Install pyenv
RUN curl -L -O https://github.com/pyenv/pyenv/archive/v1.2.8.tar.gz && \
    tar zxvf v1.2.8.tar.gz && rm -rf v1.2.8.tar.gz && \
    mv pyenv-1.2.8 .pyenv

ENV PYENV_ROOT=/root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN eval "$(pyenv init -)"

ARG PYTHON_VERSION

RUN if [ ${PYTHON_VERSION} = "3.7" ]; then \
        pyenv install 3.7.1; \
        pyenv global 3.7.1; \
        pyenv rehash; \
    elif [ ${PYTHON_VERSION} = "3.6" ]; then \
        pyenv install 3.6.7; \
        pyenv global 3.6.7; \
        pyenv rehash; \
    elif [ ${PYTHON_VERSION} = "3.5" ]; then \
        pyenv install 3.5.5; \
        pyenv global 3.5.5; \
        pyenv rehash; \
    fi

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir future hypothesis numpy protobuf six pytest

ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/lib:$LIBRARY_PATH

# Install Chainer
ARG CHAINER_VERSION
RUN pip install chainer==${CHAINER_VERSION}

# Install ChainerCV
ARG CHAINERCV_VERSION
RUN pip install chainercv==${CHAINERCV_VERSION}
