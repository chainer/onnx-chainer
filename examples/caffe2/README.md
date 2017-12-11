

```bash
git clone --recursive https://github.com/caffe2/caffe2.git
cd caffe2 && mkdir build && cd build
PYTHON_DIR=~/.pyenv/versions/onnx-chainer
echo $PYTHON_DIR
cmake .. \
-DUSE_CUDA=OFF \
-DUSE_NCCL=OFF \
-DUSE_NNPACK=OFF \
-DUSE_ROCKSDB=OFF \
-DUSE_LEVELDB=OFF \
-DUSE_LMDB=OFF \
-DUSE_OPENCV=OFF \
-DBUILD_TEST=OFF \
-DBUILD_BENCHMARK=OFF \
-DPYTHON_EXECUTABLE=$PYTHON_DIR/bin/python \
-DPYTHON_INCLUDE_DIR=$PYTHON_DIR/include \
-DPYTHON_INCLUDE_DIR2=$PYTHON_DIR/include/python3.6m \
-DPYTHON_LIBRARY=$PYTHON_DIR/lib/libpython3.6m.so \
-DPYTHON_LIBRARY_DEBUG=$PYTHON_DIR/lib/libpython3.6m.so \
-DPROTOBUF_INCLUDE_DIR=$PYTHON_DIR/include \
-DPROTOBUF_LIBRARY=$PYTHON_DIR/lib/libprotobuf.so \
-DPROTOBUF_LIBRARY_DEBUG=$PYTHON_DIR/lib/libprotobuf.so \
-DPROTOBUF_LITE_LIBRARY=$PYTHON_DIR/lib/libprotobuf-lite.so \
-DPROTOBUF_LITE_LIBRARY_DEBUG=$PYTHON_DIR/lib/libprotobuf-lite.so \
-DPROTOBUF_PROTOC_LIBRARY=$PYTHON_DIR/lib/libprotoc.so \
-DPROTOBUF_PROTOC_LIBRARY_DEBUG=$PYTHON_DIR/lib/libprotoc.so
-DCMAKE_INSTALL_PREFIX=../
&& make -j"$(nproc)" instal