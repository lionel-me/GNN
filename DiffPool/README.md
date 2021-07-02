# DiffPool
基于论文[Hierarchical Graph Representation Learning with Differentiable Pooling](https://papers.nips.cc/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf)利用dgl工具实现的简单的example，用于图分类任务。

目前想法是利用diffpool模型代替cost model中的node embedding与reduction过程，并与graphsage&reduction的实现效果进行对比。

鉴于cost model为回归而非分类任务，目前想法是用MSE代替模型中原本的交叉熵损失函数。

[dgl 原仓库地址](https://github.com/dmlc/dgl/tree/master/examples/pytorch/diffpool)

## Requirements
- pytorch 1.0+
- dgl

## How to run
在两个图分类数据集上进行训练

`python train.py --dataset ENZYMES --pool_ratio 0.10 --num_pool 1 --epochs 1000`

`python train.py --dataset DD --pool_ratio 0.15 --num_pool 1  --batch-size 10`

模型修改进行中......
在model文件夹里写了一个cost_model.py，基本是encoder.py的复制，就改了一下损失函数。