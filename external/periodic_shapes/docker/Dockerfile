FROM ubuntu:18.04
RUN apt update

RUN apt install -y --no-install-recommends \
    python3 \
    python3-dev \
    git \
    curl \
    wget \
    axel \
    vim
RUN curl -k https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN apt install -y --no-install-recommends \
    libspatialindex-dev
RUN pip3 install trimesh[easy]
