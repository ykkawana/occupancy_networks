from nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y \
    && apt install -y \
    make \
    build-essential \
    wget \
    curl \
    llvm \
    vim \
    git \

RUN wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb && \
    dpkg -i ./mesa_18.3.3-0.deb || true && \
    apt install -f
RUN git clone git@github.com:mmatl/pyopengl.git && \
    pip3 install ./pyopengl
RUN pip3 install \
    numpy \
    trimesh
