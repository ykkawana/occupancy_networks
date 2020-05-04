from nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y \
    && apt install -y \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python-openssl \
    vim \
    git

ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
    eval "$(pyenv init -)"
RUN CFLAGS=-I/usr/include \
    LDFLAGS=-L/usr/lib \
    pyenv install -v 3.7.4 && \
    pyenv global 3.7.4
RUN wget https://github.com/Kitware/CMake/releases/download/v3.17.1/cmake-3.17.1.tar.gz && \
    tar zxvf cmake-3.17.1.tar.gz && \
    cd cmake-3.17.1 && \
     ./bootstrap && \
     make -j8 && \
     make install && \
     echo 'export PATH=$HOME/cmake-3.17.1/bin/:$PATH' >> ~/.bashrc
RUN chmod 777 ~/.bashrc
RUN source ~/.bashrc
RUN pip3 install numpy 
COPY ./ ./
RUN pip3 install -r requirements.txt
