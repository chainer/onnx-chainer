FROM ubuntu:16.04

MAINTAINER Shunta Saito <shunta.saito@gmail.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    cmake \
    curl \
    wget \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev

RUN apt-get purge -y cmake
RUN curl -L -O https://github.com/Kitware/CMake/archive/v3.12.4.tar.gz && \
    tar zxvf v3.12.4.tar.gz && rm -rf v3.12.4.tar.gz && \
    cd CMake-3.12.4 && \
    ./bootstrap && make -j8 && make install

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
        CONFIGURE_OPTS="--enable-shared" pyenv install 3.7.1; \
        pyenv global 3.7.1; \
        pyenv rehash; \
    elif [ ${PYTHON_VERSION} = "3.6" ]; then \
        CONFIGURE_OPTS="--enable-shared" pyenv install 3.6.7; \
        pyenv global 3.6.7; \
        pyenv rehash; \
    elif [ ${PYTHON_VERSION} = "3.5" ]; then \
        CONFIGURE_OPTS="--enable-shared" pyenv install 3.5.5; \
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
