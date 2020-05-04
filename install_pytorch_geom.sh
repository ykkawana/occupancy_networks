CUDA=cu101
pip3 install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html --upgrade --no-cache-dir
pip3 install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html   --upgrade  --no-cache-dir
pip3 install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html --upgrade  --no-cache-dir
pip3 install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html --upgrade  --no-cache-dir
pip3 install torch-geometric --no-cache-dir --upgrade
