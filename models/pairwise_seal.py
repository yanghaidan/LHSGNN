#pytorch
import torch
import torch.nn.functional as F
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




def pairwise_seal(A,motif_test_pos, motif_test_pos_subgraphs ,motif_test_neg, motif_test_neg_subgraphs, args, sampling_time):


    args.num_epochs = 50
    args.learning_rate = 0.0001


    #train a SEAL GNN for liked prediction
    train_pos, train_neg, test_pos, test_neg = sample_neg(A,max_train_num=args.max_train_num)

    node_information = None
    if args.use_embedding:
        embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
        node_information = embeddings        

    train_dataset, test_dataset, max_n_label = links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=args.hop,
                                                max_nodes_per_hop=args.max_nodes_per_hop, node_information=node_information,
                                                no_parallel=args.no_parallel)


    print('Start extracting motifs links')

    #extract all needed links for motif
    test_links_pos=([], [])
    test_links_neg=([], [])

    motif_edge_index = []
    dealbreaker_edge_index = []
    

    index = 0
    m_ind = 0

    for m in motif_test_pos:
        motif_edge_index.append(index)
        # add all edges that need to be predicted.
        for edge in motif_test_pos_subgraphs[m_ind]:
            test_links_pos[0].append(m[edge[0]])
            test_links_pos[1].append(m[edge[1]])
            index += 1
        m_ind += 1

    m_ind = 0
    for m in motif_test_neg:
        motif_edge_index.append(index)
        # add all edges that need to be predicted.
        for edge in motif_test_neg_subgraphs[m_ind]:
            test_links_neg[0].append(m[edge[0]])
            test_links_neg[1].append(m[edge[1]])
            index += 1
        m_ind += 1  

    motif_edge_index.append(index)


    #generate their subgraphs
    _, motif_test_dataset, motif_max_n_label = links2subgraphs(A, None, None, test_links_pos, test_links_neg, h=args.hop,
                                                 max_nodes_per_hop=args.max_nodes_per_hop, node_information=node_information,
                                                 no_parallel=args.no_parallel)

    #decide max label
    max_n_label = max(max_n_label,motif_max_n_label)

    # transform labels into one hot matrix and append them to feature matrix
    for data in motif_test_dataset:
        one_hot_matrix = F.one_hot(data.z, max_n_label + 1).to(torch.float)
        if data.x is not None:
            data.x = torch.cat((data.x,one_hot_matrix),1)
        else:
            data.x = one_hot_matrix   

    #run prediction on these subgraphs
    motif_test_loader = DataLoader(motif_test_dataset, batch_size=args.batch_size)


    # transform labels into one hot matrix and append them to feature matrix
    for data in chain(train_dataset, test_dataset):
        one_hot_matrix = F.one_hot(data.z, max_n_label + 1).to(torch.float)
        if data.x is not None:
            data.x = torch.cat((data.x,one_hot_matrix),1)
        else:
            data.x = one_hot_matrix

    # split validation data
    random.shuffle(train_dataset)
    val_num = int(0.1 * len(train_dataset))
    val_dataset = train_dataset[:val_num]
    train_dataset = train_dataset[val_num:]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # DGCNN
    class DGCNN(torch.nn.Module):
        def __init__(self, hidden_channels, num_layers, GNN=torch_geometric.nn.GCNConv, k=0.6):
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


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCNN(hidden_channels=32, num_layers=3).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)


    def train():
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

        return total_loss / len(train_dataset)


    @torch.no_grad()
    def test(loader):
        model.eval()

        y_pred, y_true = [], []
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)

            y_pred.append(logits.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))

        pred = torch.max(torch.cat(y_pred)  ,torch.tensor([0.]))
        pred[pred > 0] = 1

        return metrics.accuracy_score(torch.cat(y_true), pred), metrics.roc_auc_score(torch.cat(y_true), torch.cat(y_pred))

    print('Starting GNN training.')
    best_val_auc = best_loss = test_auc = best_epoch = 0
    for epoch in range(1, args.num_epochs + 1):
        start = time.perf_counter()
        loss = train()
        val_acc, val_auc = test(val_loader)
        if val_auc >= best_val_auc:
            best_epoch = epoch
            best_loss = loss
            best_val_auc = val_auc
            test_acc, test_auc = test(test_loader)
            print(f'\033[0mEpoch: {epoch:02d},\033[92m Loss: {loss:.4f},\033[93m Val_ACC: {val_acc:.4f}, Val_AUC: {val_auc:.4f},'
                  f'\033[94m Test_ACC: {test_acc:.4f}, Test_AUC: {test_auc:.4f}    \033[96m (Execution time: {time.perf_counter()-start:.2f}s)\033[0m')
        else:
            print(f'\033[0mEpoch: {epoch:02d},\033[92m Loss: {loss:.4f},\033[93m Val_ACC: {val_acc:.4f}, Val_AUC: {val_auc:.4f},'
                  f'\033[0m Test_ACC: {test_acc:.4f}, Test_AUC: {test_auc:.4f}    \033[96m (Execution time: {time.perf_counter()-start:.2f}s)\033[0m')

    print('GNN link prediction training done.')


    @torch.no_grad()
    def test_pairwise(loader):
        model.eval()

        y_pred, y_true = [], []
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)

            y_pred.append(logits.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))

        y_pred_prob = torch.sigmoid(torch.cat(y_pred))
        y_true = torch.cat(y_true)

        motif_pred_prob = []
        motif_pred_true = []

        if args.motif == 'star' or args.motif == 'dbstar' or args.motif == 'clique' or args.motif == 'circle' or args.motif == 'path' or args.motif == 'tailed_triangle' or args.motif == 'chordal_cycle':
            
            if args.aggregation =='mul':

                print('Aggration done using: MUL')
                #multiply probabilities to get motif probability
                for motif in range(0,len(motif_test_pos)+len(motif_test_neg)):

                    motif_prob = 1
                    #motif edges
                    for i in range(motif_edge_index[motif],motif_edge_index[motif+1]):
                        motif_prob *= y_pred_prob[i].item()
                    
                    motif_pred_prob.append(motif_prob)
                    motif_pred_true.append(int(y_true[motif_edge_index[motif]].item()))
            

            if args.aggregation == 'avg':

                print('Aggration done using: AVG')

                for motif in range(0,len(motif_test_pos)+len(motif_test_neg)):
                
                    edge_count = 0
                    prob_sum = 0

                    #motif edges
                    for i in range(motif_edge_index[motif],motif_edge_index[motif+1]):
                        prob_sum += y_pred_prob[i].item()
                        edge_count += 1
                    
                    if edge_count != 0:
                        motif_prob = prob_sum / edge_count
                    else:
                        motif_prob = 0.0
                    motif_pred_prob.append(motif_prob)
                    motif_pred_true.append(int(y_true[motif_edge_index[motif]].item()))

            if args.aggregation == 'min':

                print('Aggration done using: MIN')
                
                for motif in range(0,len(motif_test_pos)+len(motif_test_neg)):

                    prob_min = 1

                    #motif edges
                    for i in range(motif_edge_index[motif],motif_edge_index[motif+1]):
                        prob_min = min(y_pred_prob[i].item(),prob_min)

                    motif_prob = prob_min

                    motif_pred_prob.append(motif_prob)
                    motif_pred_true.append(int(y_true[motif_edge_index[motif]].item()))

            
        motif_auc_score = metrics.roc_auc_score(motif_pred_true,motif_pred_prob)

        #turn prediction probability to prediction
        motif_pred = torch.tensor(motif_pred_prob)

        threshold = 0.5

        motif_pred[motif_pred > threshold] = 1
        motif_pred[motif_pred < threshold] = 0

        motif_acc_score = metrics.accuracy_score(motif_pred_true,motif_pred)

        # report final accuracy
        print('\033[95mFinal test performance using pairwise GNN link prediction: acc %.5f auc %.5f\033[0m' % (
        motif_acc_score, motif_auc_score))

        print_results(args, motif_acc_score, motif_auc_score, sampling_time, 0, 0)

        return 

    if args.aggregation == 'all':
        aggregations = ['mul','avg','min']
    else:
        aggregations = [args.aggregation]

    for agg in aggregations:
        args.aggregation = agg
        test_pairwise(motif_test_loader)

    return 

    