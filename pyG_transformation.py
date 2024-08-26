from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os
#import cPickle as cp
#import _pickle as cp  # python3 compatability
import networkx as nx
import pdb
import argparse

import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges, k_hop_subgraph,
                                   to_scipy_sparse_matrix)



def GNNGraph_pyG(g, g_label, n_labels, n_features):

    # transform graph into edge_index tensor
    x, y = zip(*g.edges())
    num_edges = len(x)        
    edge_pairs = np.ndarray(shape=(2,num_edges), dtype=np.int32)
    edge_pairs[0, :] = x
    edge_pairs[1, :] = y

    edge_index = torch.tensor(edge_pairs,dtype=torch.long)

    # transfrom features and labels to feature matrix 
    x =  torch.tensor(n_features,dtype=torch.float32)

    z = torch.tensor(n_labels,dtype=torch.long)

    data = Data(x=x, z=z,
                        edge_index=edge_index, y=g_label)
    return data