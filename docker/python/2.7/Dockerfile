FROM ubuntu:16.04

MAINTAINER Shunta Saito <shunta.saito@gmail.com>

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python-dev \
    python-pip \
    python-dbg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir future hypothesis six
