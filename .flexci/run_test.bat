@echo off

set CUDA_VER=%1
set PY_VER=%2
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VER%"
set PY_PATH=C:\Development\Python\Python%PY_VER%
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PY_PATH%;%PY_PATH%\Scripts\%PATH%

pip install "cupy-cuda101<7.0.0"
pip install "chainer<7.0.0"
pip install -e .[flexci]
pip list -v

pytest -x -s -vvvs tests --cov onnx_chainer
