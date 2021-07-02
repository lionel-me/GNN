import os
import numpy as np
import torch
import dgl
import networkx as nx
import argparse
import random
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import tu

from model.encoder import DiffPool
from data_utils import pre_process

class Costmodel(nn.Module):
    """
    google cost model v2
    """

    