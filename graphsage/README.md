# GrapgSage
基于论文Inductive Representation Learning on Large Graphs，利用pytorch & dgl工具实现的graphsage的example,以及对于此方法在Google cost model中应用的尝试
+ [paperlink](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
+ [dgl 原仓库地址](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage)
## Requirements
请根据自身硬件情况进行安装
+ [pytorch](https://pytorch.org/get-started/locally/)
+ [dgl](https://www.dgl.ai/pages/start.html)
## How to run
`python graphsage.py --dataet cora --aggregator-type mean --gpu 0`
可用的数据集包含cora、citeseer、pumbed，聚合函数包括mean、pool、lstm、gcn，注意到在gcn聚合函数下，graphsage相当与Kipf et al所提出GCN的归纳扩展版本。
