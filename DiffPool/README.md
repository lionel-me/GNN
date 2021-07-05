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

可通过将train.py中17行的encoder改为cost_model来调用cost_model中的DiffPool类。

## Datasets
在计算图与运行时间数据集未知的情况下，先寻找别的graph regession的数据集，测试修改损失函数后能否有预想的训练效果。

在[dgl.dataset.TUDataset](http://docs.dgl.ai/api/python/dgl.data.html#tu-dataset)中提供了图分类，回归的多种数据集以供调用，具体文档可参考其中给出的链接

https://chrsmrrs.github.io/datasets/docs/datasets
