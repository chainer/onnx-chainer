**Build scripts**

```bash
$ export BUILD_PY_VER=36
$ docker build -t disktnk/onnx-chainer:ci-py${BUILD_PY_VER} -f .chainerci/Dockerfile --build-arg PYTHON_VERSION=${BUILD_PY_VER} .
```
