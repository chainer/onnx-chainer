FROM ubuntu:trusty

MAINTAINER Shunta Saito <shunta.saito@gmail.com>

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git

WORKDIR /root

# Install miniconda
RUN curl -L https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p miniconda && rm -rf miniconda.sh
ENV PATH /root/miniconda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /root/miniconda/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /root/miniconda/lib:$LIBRARY_PATH
ENV CPATH /root/miniconda/include:$CPATH

RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -n base conda

# Install ONNX
RUN conda install -c conda-forge protobuf numpy
RUN ONNX_ML=1 pip install onnx==1.1.1 && \
    python -c 'import onnx; print(onnx.__version__)'

# Install cmake
RUN conda install cmake

# Install caffe2
RUN conda install -c caffe2 caffe2-gcc4.8

# Dependencies for test
RUN pip install future hypothesis flake8 hacking autopep8 mock pytest six

# Install Chainer
RUN pip install chainer==4.0.0

# Install MXNet
RUN pip install mxnet==1.1.0.post0 onnx-mxnet==0.4.2

# Install NNVM
RUN git clone --recursive https://github.com/dmlc/nnvm $HOME/nnvm
RUN cd $HOME/nnvm && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=../install ..
RUN cd $HOME/nnvm/build && make -j$(nproc) install
RUN cd $HOME/nnvm/python && python setup.py install

# Install TVM
RUN git clone --recursive https://github.com/dmlc/tvm $HOME/tvm
RUN cd $HOME/tvm && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=../install .. \
    && make -j$(nproc) install
RUN cd $HOME/tvm/python && python setup.py install
RUN cd $HOME/tvm/topi/python && python setup.py install
