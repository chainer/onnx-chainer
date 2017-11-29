FROM ubuntu:trusty

MAINTAINER Shunta Saito <shunta.saito@gmail.com>

RUN apt-get update
RUN apt-get install -y build-essential git wget
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p $HOME/miniconda
RUN export PATH="$HOME/miniconda/bin:$PATH"
RUN hash -r
RUN conda config --set always_yes yes --set changeps1 no
RUN conda info -a
RUN conda install protobuf
RUN conda install glog
RUN hash -r
RUN CFLAGS="-I$HOME/miniconda/include" pip install git+git://github.com/onnx/onnx.git@a8bd1facd321ea52f0a6f88600138606711b7a96
RUN pip install chainer==3.1.0
RUN pip install -e .
