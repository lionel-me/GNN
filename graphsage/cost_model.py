import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv

class cost_model(nn.Module):
    def __init__(self,
                in_feats,
                n_hidden,
                n_gnn_layers,
                n_transformer_head,
                n_transformer_layers,
                dropout,
                aggre_type
                ):
        super(cost_model,self).__init__()
        self.gnnlayers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        #GNN layers:
        self.gnnlayers.append(SAGEConv(in_feats,n_hidden,aggre_type))
        for i in range(n_gnn_layers - 1):
            self.gnnlayers.append(SAGEConv(n_hidden,n_hidden,aggre_type))

        #feeddorward before reduction:
        self.feedforward = nn.Sequential(
            nn.Linear(n_hidden,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,n_hidden),
            nn.ReLU(),
        )

        #reduction:
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=n_transformer_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,n_transformer_layers)

        #feed forward after reduction
        self.output = nn.Sequential(
            nn.Linear(n_hidden,1),
            nn.ReLU()
        )

    def forward(self,graph,input):
        h = self.dropout(input)
        for l, layer in enumerate(self.gnnlayers):
            h = layer(graph, h)
        h = self.feedforward(h)#num_node*n_hidden
        x,y = h.shape
        h = h.reshape(x,1,y)
        h = self.transformer_encoder(h)
        h = h.reshape(x,y)
        h = torch.sum(h,0)
        h = self.output(h)
        return h


if __name__ == '__main__':
    u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
    g = dgl.graph((u, v))
    g.ndata['x'] = 10 * torch.rand(g.num_nodes(), 256)
    feats = g.ndata['x']
    label = torch.rand(1)

    cuda = True
    torch.cuda.set_device(0)
    feats = feats.cuda()
    label = label.cuda()
    g = g.to('cuda:0')

    in_feats = 256
    n_hidden = 512
    n_gnn_layers = 3
    n_transformer_head = 4
    n_transformer_layers = 2
    dropout = 0.2
    aggre_type = 'mean'
    lr = 0.1


    model = cost_model(in_feats,n_hidden,n_gnn_layers,n_transformer_head,n_transformer_layers,dropout,aggre_type)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),lr)
    print('label:',label)
    for epoch in range(50):
        pre = model(g,feats)
        loss = F.mse_loss(pre,label)
        if(epoch % 10 == 0):
            print(epoch,pre,loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()      
