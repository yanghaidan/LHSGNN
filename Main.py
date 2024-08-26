# pytorch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d
# pytorch_geometric
import torch_geometric.nn
from torch_geometric.data import Data, DataLoader
# utils
import numpy as np
import sys, copy, math, time, pdb, os
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from tqdm import tqdm
from itertools import chain
from sklearn import metrics
# project files
from options import args_parser
from util_functions import *
from sampling import *
from models.naive_prediction import *
from models.pairwise_seal import *
from models.seam import *
from utils.data_utils import load_data
from models.lhsgnn import *

import ray
import psutil
from scipy.sparse import csr_matrix

def main_func():
    # fix for python 3.8
    try:
        mp.set_start_method("fork")
    except:
        pass

    args = args_parser()

    print('Input parameters:\n', args)

    # sanitize inputs
    if args.test_ratio >= 1 or args.test_ratio <= 0:
        args.test_ratio = 0.1
        print('Invalid test ratio provided. Test ratio automatically set to 0.1')

    if args.max_motifs_node == 0:
        args.max_motifs_node = None    

    assert args.prediction_method == 'naive' or args.prediction_method == 'seal' or args.prediction_method == 'seam' or args.prediction_method == 'lhsgnn', 'We only support naive/seal/seam/lhsgnn as prediction method right now!'
    assert args.motif == 'clique' or args.motif == 'star' or args.motif == 'dense' or args.motif=='dbstar' or args.motif == 'circle' or args.motif == 'path' or args.motif == 'tailed_triangle' or args.motif == 'chordal_cycle', 'We only support star/dbstar/clique/dense/circle/path/tailed_triangle/chordal_cycle as motifs right now!'
    # assert (args.motif_k > 2), "Must provide motif-k bigger than 2"
    assert args.naive_method == 'jaccard' or args.naive_method == 'cn' or args.naive_method == 'aa' or args.naive_method == 'all', 'We only supposet jacard/cn/aa for now!'
    assert args.aggregation == 'mul' or args.aggregation == 'avg' or args.aggregation == 'min' or args.aggregation == 'all', 'We only support mul/avg/min right now!'
    assert args.distance_label_mode >= 0 and args.distance_label_mode <= 3 , 'We only distance_label_mode from 0 to 3 right now!'

    # set up timing variables
    sampling_time = extraction_time = gnn_time = None

    # set max number of cores
    set_max_cpu_count(args.max_cores)
    
    #init ray framework
    print("Ray framework initialized with",max_cpu_count(),"cores.")
    ray.init(num_cpus=max_cpu_count())
    # to run distributed
    #ray.init(address='auto')
    print('''This cluster consists of
        {} nodes in total
        {} CPU resources in total
    '''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))


    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        print("CUDA: #", torch.cuda.device_count(), " ", torch.cuda.get_device_name())
        torch.cuda.manual_seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # generate file path
    if args.scratch_mode:
        file_path = os.environ['SCRATCH'] + '/subgraph_datasets/'
    else:
        file_path = 'data/subgraph_datasets/'

    # generate file names
    train_file_name = str(args.data_name) + '_train_motif' + str(args.motif) + '_k' + str(args.motif_k) + (
        ('_f' + str(args.motif_f)) if args.motif == 4 else '') + '.pt'
    val_file_name = str(args.data_name) + '_val_motif' + str(args.motif) + '_k' + str(args.motif_k) + (
        ('_f' + str(args.motif_f)) if args.motif == 4 else '') + '.pt'
    test_file_name = str(args.data_name) + '_test_motif' + str(args.motif) + '_k' + str(args.motif_k) + (
        ('_f' + str(args.motif_f)) if args.motif == 4 else '') + '.pt'

    file_not_found = False
    if args.load_datasets and args.prediction_method == 'seam':
        # load subgraph datasets from disk
        try:
            start = time.perf_counter()
            print('Start loading datasets from disk.')
            train_dataset = torch.load(file_path + train_file_name)
            val_dataset = torch.load(file_path + val_file_name)
            test_dataset = torch.load(file_path + test_file_name)
            print('Datasets succesfully loaded. (Execution time: %.2fs)' % (time.perf_counter() - start))
        except FileNotFoundError:
            print("Dataset files not found. Dataset generation initialized.")
            file_not_found = True
    
    if not args.load_datasets or file_not_found or args.prediction_method != 'seam' or args.prediction_method != 'lhsgnn':
        # generate subgraph Datasets
        if args.hop != 'auto':
            args.hop = int(args.hop)
        if args.max_nodes_per_hop is not None:
            args.max_nodes_per_hop = int(args.max_nodes_per_hop)

        '''Prepare data'''
        args.file_dir = os.path.dirname(os.path.realpath('__file__'))

        # build observed network
        if args.data_name is not None:  # use .mat network
            # dataset = load_data(args.data_name)
            # net = dataset.data.edge_index
            args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
            data = sio.loadmat(args.data_dir)
            net = data['net']

            if 'group' in data:
                # load node attributes (here a.k.a. node classes)
                if type(data['group']) == np.ndarray:
                    attributes = data['group'].astype('float32')
                else:
                    attributes = data['group'].toarray().astype('float32')
            else: 
                attributes = None
            # check whether net is symmetric (for small nets only)
            if False:
                net_ = net.toarray()
                assert (np.allclose(net_, net_.T, atol=1e-8))

        start_sampling = time.perf_counter()
        
        # sample both positive and negative train/test motifs from input graph net
        train_pos, train_pos_subgraphs, train_neg, train_neg_subgraphs, test_pos, test_pos_subgraphs, test_neg, test_neg_subgraphs = sample_neg_motif(
            net, args.test_ratio, motif=args.motif, k=args.motif_k, f=args.motif_f, motif_edge_mode=args.motif_edge_mode,
            max_train_num=args.max_train_num, max_motifs_per_node=args.max_motifs_node, no_parallel=args.no_parallel)
        

        sampling_time = time.perf_counter() - start_sampling

        '''Train and apply classifier'''
        A = net.copy()
        

        ##############################
        ##### PREDICTION METHODS #####
        ##############################
        

        if args.prediction_method == 'naive':

            if args.naive_method == 'all':
                naive_methods = ['jacard','aa','cn']
            else:
                naive_methods = [args.naive_method]

            if args.aggregation == 'all':
                aggregations = ['mul','avg','min']
            else:
                aggregations = [args.aggregation]

            for nm in naive_methods:
                for agg in aggregations:
                    args.naive_method = nm
                    args.aggregation = agg

                    acc_, roc_auc_ = naive_motif_pred(nx.from_scipy_sparse_array(A), test_pos, test_pos_subgraphs,
                             test_neg, test_neg_subgraphs, 
                             args.motif, args.motif_f, args.naive_method, args.aggregation, args.prediction_threshold, args.naive_edge_mode)

                    print_results(args, acc_, roc_auc_, sampling_time, 0, 0)

        elif args.prediction_method == 'seal' :

            # pairwise link prediction GNN
            pairwise_seal(A, test_pos, test_pos_subgraphs, test_neg, test_neg_subgraphs, args, sampling_time)  


        elif args.prediction_method == 'seam':
            acc_result, auc_result, extraction_time, gnn_time = seam(A, test_pos, test_pos_subgraphs, test_neg, test_neg_subgraphs, train_pos,
                                                                    train_pos_subgraphs, train_neg, train_neg_subgraphs, args)
                
            print_results(args, acc_result, auc_result, sampling_time, extraction_time, gnn_time)
        elif args.prediction_method == 'lhsgnn':
            
            acc_result, auc_result, extraction_time, gnn_time = lhsgnn(A, test_pos, test_pos_subgraphs, test_neg, test_neg_subgraphs, train_pos,
                                                                train_pos_subgraphs, train_neg, train_neg_subgraphs, args)
            
            print_results(args, acc_result, auc_result, sampling_time, extraction_time, gnn_time)

if __name__ == '__main__':
    
    main_func()
    
