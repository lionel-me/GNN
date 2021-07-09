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

## Results

在enzymes数据集上进行了400轮训练，得到测试集上准确率为60%，稍逊于论文中提及的64.23%

## QM9
`python train_regression.py --pool_ratio 0.10 --num_pool 1 --epoch 40 --load_epoch 30`

train_regression.py 修改自train.py，默认调用了dgl中的QM9数据集，选择cut off = 2来减少图中边的数量，使其性质接近cost model中的计算图，先选择19个regression target中的第一个'mu'，进行回归训练，目前正在训练中，看起来正在收敛。
loss为mse，由于回归的label mu有些数据接近0，在此先不适用mape而是用mae来考察模型的训练效果。
### 训练参数与结果
训练的过程中经历了一系列服务器使用与gpu有效利用的问题，主要原因在于本人的经验不足，在此不表。

首次训练采用adam优化器，设置lr=0.04,在学习约100个epoch后训练集上的loss开始停止下降，观察后猜想可能是lr设置过大导致的，因此将lr调小至0.008进行400epoch训练，最终得到训练集、验证集、测试集上的mae分别为0.2216、0.2625、0.2557。

后续步骤继续调小lr至0.001，进行400epoch训练后得到结果为0.2047、0.2686、0.2457

同时尝试在模型中加入可训练的偏置参数用以调整模型输出的整体分布，训练lr仍为0.001，训练400epochs，得到结果为0.2117、0.2533、0.2442

同时在lr=0.001，训练400epochs后的模型参数基础上又进行了训练，观察到训练集的mae仍在下降但验证集上的mae基本停滞，猜想模型已经在训练集上过拟合。

下一步计划，选取数据集的alpha回归参数作为kernel feature加入特征，并将原本的node feature Z即原子序号作为opcode进行embedding，总之使数据集尽量贴近将要拿到的cost model数据集的特性并进行训练，在此基础之上调整模型的参数
