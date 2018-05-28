FROM ubuntu:16.04

MAINTAINER Shunta Saito <shunta.saito@gmail.com>

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-dbg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.5 1
RUN update-alternatives --install /usr/bin/python-config python-config /usr/bin/python3.5-config 1
RUN update-alternatives --install /usr/bin/python-dbg python-dbg /usr/bin/python3.5-dbg 1
RUN update-alternatives --install /usr/bin/python-dbg-config python-dbg-config /usr/bin/python3.5-dbg-config 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
RUN update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python3.5-config 1
RUN update-alternatives --install /usr/bin/python3-dbg python3-dbg /usr/bin/python3.5-dbg 1
RUN update-alternatives --install /usr/bin/python3-dbg-config python3-dbg-config /usr/bin/python3.5-dbg-config 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir future hypothesis six
