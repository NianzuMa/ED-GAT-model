## Set up on Ubuntu Linux 18.04

```
### conda create -n graph python=3.7
### conda activate graph
### conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

#### `https://github.com/rusty1s/pytorch_geometric`

```
export CUDA=cu102

pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric
```

pip install pipreqs