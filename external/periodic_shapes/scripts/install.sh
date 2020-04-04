#!/bin/bash

pip3 install -r requirements.txt
pushd external
    git clone https://github.com/NVIDIAGameWorks/kaolin.git
    pushd kaolin
        python3 setup.py build_ext --inplace
        python3 setup.py install
    popd
popd
