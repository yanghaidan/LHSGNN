# #pytorch
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d
#pytorch_geometric
import torch_geometric.nn
from torch_geometric.data import Data, DataLoader

#utils
import numpy as np
import sys, copy, math, time, pdb, os
from decimal import *
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from tqdm import tqdm
from itertools import chain
from sklearn import metrics


#local files
from util_functions import *
torch.cuda.set_device(1)

# DGCNN
class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, train_dataset, GNN=torch_geometric.nn.GCNConv, k=0.6):
        super(DGCNN, self).__init__()

        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)
        print('k used in SortPooling is: ' + str(k))

        self.convs = ModuleList()
        self.convs.append(GNN(train_dataset[0].x.shape[1], hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, x, edge_index, batch):
        xs = [x]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index))]
        x = torch.cat(xs[1:], dim=-1)


        # Global pooling.
        x = torch_geometric.nn.global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x



def train(model,device,optimizer,train_loader):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.batch)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(model,device,loader):
    model.eval()

    y_pred, y_true = [], []
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)

        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))

    pred = torch.max(torch.cat(y_pred), torch.tensor([0.]))
    pred[pred > 0] = 1

    return metrics.accuracy_score(torch.cat(y_true), pred), metrics.roc_auc_score(torch.cat(y_true),
                                                                                  torch.cat(y_pred))


#train_pos代表正训练样本的节点列表、train_pos_subgraphs代表正训练样本的边列表情况
def seam(A,test_pos, test_pos_subgraphs ,test_neg, test_neg_subgraphs, train_pos,
        train_pos_subgraphs, train_neg, train_neg_subgraphs, args):
    
    # GNN
    node_information = None

    # generate node embeddings
    if args.use_embedding:
        A1 = A.copy()
        
        for index,pos in enumerate(test_pos):
            for i,j in test_pos_subgraphs[index]:
                A1[pos[i],pos[j]] = 0
                A1[pos[j],pos[i]] = 0
        for index,neg in enumerate(test_neg):
            for i,j in test_neg_subgraphs[index]: 
                A1[neg[i],neg[j]] = 0
                A1[neg[j],neg[i]] = 0 
        
        
        embeddings = generate_node2vec_embeddings_motif(A1, 128, True, train_pos, train_neg, train_neg_subgraphs, test_pos, test_pos_subgraphs, test_neg, test_neg_subgraphs)
      
        
    
        node_information = embeddings

    # get input feature matrix
    if args.use_attribute and attributes is not None:
        if node_information is not None:
            node_information = np.concatenate([node_information, attributes], axis=1)
        else:
            node_information = attributes

    start_extraction = time.perf_counter()

   
    # Extract subgraphs
    train_dataset, test_dataset, max_n_label = motifs2subgraphs(
        nx.from_scipy_sparse_array(A),
        train_pos,
        train_pos_subgraphs,
        train_neg,
        train_neg_subgraphs,
        test_pos,
        test_pos_subgraphs,
        test_neg,
        test_neg_subgraphs,
        args.distance_label_mode,
        args.motif_edge_mode,
        args.hop,
        args.max_nodes_per_hop,
        node_information,
        args.no_parallel
        )

    extraction_time = time.perf_counter() - start_extraction

    print('# train: %d, # test: %d' % (len(train_dataset), len(test_dataset)))
    # transform labels into one hot matrix and append them to feature matrix
    for data in chain(train_dataset, test_dataset):
        one_hot_matrix = F.one_hot(data.z, max_n_label + 1).to(torch.float)
        data.x = torch.cat((data.x, one_hot_matrix), 1)

    # split validation data
    random.shuffle(train_dataset)
    val_num = int(0.1 * len(train_dataset))
    val_dataset = train_dataset[:val_num]
    train_dataset = train_dataset[val_num:]

    if args.save_datasets or args.only_save_datasets:
        print('Start saving datasets to disk.')
        # file_path = 'data/subgraph_datasets/' + args.data_name + '/'
        # train_file_name = args.data_name + '.train.pt'
        # val_file_name =  args.data_name + '.val.pt'
        # test_file_name = args.data_name + '.test.pt'
        torch.save(train_dataset, file_path + train_file_name)
        torch.save(val_dataset, file_path + val_file_name)
        torch.save(test_dataset, file_path + test_file_name)
        print('Datasets succesfully saved.')

        if args.only_save_datasets:
            quit(0)

    start_gnn = time.perf_counter()

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = DGCNN(hidden_channels=32, num_layers=3, train_dataset=train_dataset).to(device)


    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    # scaler = torch.cuda.amp.GradScaler()

    print('Starting GNN training.')
    best_val_auc = best_loss = test_auc = best_epoch = 0
    for epoch in range(1, args.num_epochs + 1):
        start = time.perf_counter()
        loss = train(model,device,optimizer,train_loader)
        val_acc, val_auc = test(model,device,val_loader)
        if val_auc >= best_val_auc:
            best_epoch = epoch
            best_loss = loss
            best_val_auc = val_auc
            test_acc, test_auc = test(model,device,test_loader)
            print(
                f'\033[0mEpoch: {epoch:02d},\033[92m Loss: {loss:.4f},\033[93m Val_ACC: {val_acc:.4f}, Val_AUC: {val_auc:.4f},'
                f'\033[94m Test_ACC: {test_acc:.4f}, Test_AUC: {test_auc:.4f}    \033[96m (Execution time: {time.perf_counter() - start:.2f}s)\033[0m')
        else:
            print(
                f'\033[0mEpoch: {epoch:02d},\033[92m Loss: {loss:.4f},\033[93m Val_ACC: {val_acc:.4f}, Val_AUC: {val_auc:.4f},'
                f'\033[0m Test_ACC: {test_acc:.4f}, Test_AUC: {test_auc:.4f}    \033[96m (Execution time: {time.perf_counter() - start:.2f}s)\033[0m')
    
    gnn_time = time.perf_counter() - start_gnn

    # report final accuracy
    print('\033[95mFinal test performance: epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
        best_epoch, best_loss, test_acc, test_auc))
    
    del model
    torch.cuda.empty_cache()

    return test_acc, test_auc ,extraction_time, gnn_time

