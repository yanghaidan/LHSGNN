import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.datasets import Planetoid, CitationFull, Flickr, Twitch, Coauthor
from torch_geometric.utils import from_networkx, train_test_split_edges, add_self_loops, negative_sampling, k_hop_subgraph
from torch_geometric.transforms import RandomLinkSplit

import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os.path as osp
from itertools import chain



class Disease(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['disease.pt']

    def process(self):
        path = 'data1/disease_lp/'
        edges = pd.read_csv(path + 'disease_lp.edges.csv')
        labels = np.load(path + 'disease_lp.labels.npy')
        features = sp.load_npz(path + 'disease_lp.feats.npz').todense()
        dataset = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edges.values).t().contiguous(),
            y=torch.tensor(labels)
        )
        torch.save(dataset, self.processed_paths[0])


class Airport(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['airport.pt']

    def process(self):
        data_path = '../data/airport'
        dataset_str = 'airport'
        graph = pickle.load(open(osp.join(data_path, dataset_str + '.p'), 'rb'))
        dataset = from_networkx(graph)
        dataset.x = dataset.feat
        dataset.feat = None
        torch.save(dataset, self.processed_paths[0])


def load_data(data_name):
    if data_name == 'cora':
        dataset = Planetoid('../data/Planetoid', name='Cora')
    elif data_name == 'cora_ml':
        dataset = CitationFull('../data/CitationFull', name='Cora_Ml')
    elif data_name == 'citeseer':
        dataset = CitationFull('../data/CitationFull', name='CiteSeer')
    elif data_name == 'pubmed':
        dataset = Planetoid('../data/Planetoid', name='PubMed')
    elif data_name == 'airport':
        dataset = Airport('../data/Airport')
    elif data_name == 'disease':
        dataset = Disease('data1/Disease')
    elif data_name == 'twitch_en':
        dataset = Twitch('../data/Twitch', name='EN')
    elif data_name == 'cs':
        dataset = Coauthor('../data/Coauthor', name='cs')
    else:
        raise ValueError('Invalid dataset!')
    return dataset
