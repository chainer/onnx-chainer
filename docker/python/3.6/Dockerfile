FROM ubuntu:16.04

MAINTAINER Shunta Saito <shunta.saito@gmail.com>

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    python3.6-dev \
    python3.6-dbg \
    && rm -rf /var/lib/apt/lists/* && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && rm -rf get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python-config python-config /usr/bin/python3.6-config 1
RUN update-alternatives --install /usr/bin/python-dbg python-dbg /usr/bin/python3.6-dbg 1
RUN update-alternatives --install /usr/bin/python-dbg-config python-dbg-config /usr/bin/python3.6-dbg-config 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python3.6-config 1
RUN update-alternatives --install /usr/bin/python3-dbg python3-dbg /usr/bin/python3.6-dbg 1
RUN update-alternatives --install /usr/bin/python3-dbg-config python3-dbg-config /usr/bin/python3.6-dbg-config 1
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip 1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir future hypothesis six
