FROM mitmul/caffe2:cpu-python3.5

# Dependencies for test
RUN pip install future hypothesis flake8 hacking autopep8 mock pytest six
RUN apt-get purge -y cmake
RUN curl -L -O https://cmake.org/files/v3.11/cmake-3.11.1-Linux-x86_64.sh && \
    bash cmake-3.11.1-Linux-x86_64.sh --skip-license && rm -rf cmake-3.11.1-Linux-x86_64.sh

# Install Chainer
RUN pip install chainer==4.0.0

# Install MXNet
RUN pip install mxnet==1.1.0.post0 onnx-mxnet==0.4.2

# Install NNVM
RUN git clone --recursive https://github.com/dmlc/nnvm $HOME/nnvm
RUN cd $HOME/nnvm && git checkout 60c685d4d395a2374e7cb234447e2533a4990105 && \
    mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../install ..
RUN cd $HOME/nnvm/build && make -j$(nproc) install
RUN cd $HOME/nnvm/python && python setup.py install

# Install TVM
RUN git clone --recursive https://github.com/dmlc/tvm $HOME/tvm
RUN cd $HOME/tvm && git checkout 27230692c14cfee63db69f9aab42a1fbb1910ef6 && \
    mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=../install .. \
    && make -j$(nproc) install
RUN cd $HOME/tvm/python && python setup.py install
RUN cd $HOME/tvm/topi/python && python setup.py install
