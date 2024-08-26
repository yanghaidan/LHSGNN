from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp

import scipy.sparse.csgraph as csg

from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/software/node2vec/src' % cur_dir)
import node2vec
import multiprocessing as mp
from itertools import islice,chain
from pyG_transformation import GNNGraph_pyG
import ray
import ctypes
import os
import argparse
import Weighted_rm
max_cores = 0

def max_cpu_count():
    return max_cores

def set_max_cpu_count(x):
    
    global max_cores
    if x is not None:
        max_cores = min(mp.cpu_count(),x)
    else:
        max_cores = mp.cpu_count()

def mask_motif_edges_subgraph(g, sub,  g_label, k, motif_edge_mode):

    # removes edges according to the subgraph mask and dealbreaker subgraph mask
    for i,j in sub:
        if g.has_edge(i,j):
            g.remove_edge(i,j)
    
    # for i,j in db_sub:
    #     if g.has_edge(i,j):
    #         g.remove_edge(i,j) 
    return

def motifs2subgraphs(input_graph, train_pos, train_pos_subgraphs, train_neg, train_neg_subgraphs, test_pos, test_pos_subgraphs, test_neg, test_neg_subgraphs,
                    distance_label_mode, motif_edge_mode, h=1, max_nodes_per_hop=None, node_information=None, no_parallel=False):
    # automatically select h from {1, 2}
    if h == 'auto':
        h = 2;

    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    
    def helper(input_graph, motifs, subgraphs, g_label):
        g_list = []
        if no_parallel:
            for m,s in tqdm(zip(motifs,subgraphs),total=len(motifs)):
                gl, max_nl = motif_subgraph_extraction_labeling(
                    m, s, input_graph, g_label, distance_label_mode, motif_edge_mode, h, max_nodes_per_hop, node_information
                )
                max_n_label['value'] = max(max_nl, max_n_label['value'])
                g_list.append(gl)
            return g_list
        else:
            # the parallel extraction code
            start = time.time()
            '''
            pool = mp.Pool(max_cpu_count())

            for g, n_labels, n_features in tqdm(pool.imap_unordered(
                parallel_motif_worker, 
                [(m, s, dealbreaker_subgraph, input_graph, g_label, motif, distance_label_mode, motif_edge_mode, h, max_nodes_per_hop, node_information) for m,s in zip(motifs,subgraphs)],chunksize=32), total=len(motifs)):
                g_list.append(GNNGraph_pyG(g, g_label, n_labels, n_features))
                max_n_label['value'] = max(max(n_labels), max_n_label['value'])

            pool.close()
            '''
            input_graph_x = ray.put(input_graph)
            g_label_x = ray.put(g_label)
            distance_label_mode_x = ray.put(distance_label_mode)
            motif_edge_mode_x = ray.put(motif_edge_mode)
            h_x = ray.put(h)
            max_nodes_per_hop_x = ray.put(max_nodes_per_hop)
            node_information_x = ray.put(node_information)
            for gl, max_nl in tqdm(ray.get([parallel_ray_motif_worker.remote(m, s, input_graph_x, g_label_x, distance_label_mode_x, motif_edge_mode_x, h_x, max_nodes_per_hop_x, node_information_x) for m, s in zip(motifs, subgraphs)]),total=len(motifs)):
                g_list.append(gl)
                max_n_label['value'] = max(max_nl, max_n_label['value'])
            
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
            
            return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs, test_graphs = None, None
    
    if train_pos and train_neg:
        train_graphs = helper(input_graph, train_pos, train_pos_subgraphs, 1) + helper(input_graph, train_neg, train_neg_subgraphs, 0)
    if test_pos and test_neg:
        test_graphs = helper(input_graph, test_pos, test_pos_subgraphs, 1) + helper(input_graph, test_neg, test_neg_subgraphs, 0)
    elif test_pos:
        test_graphs = helper(input_graph, test_pos, test_pos_subgraphs, 1)
    
    return train_graphs, test_graphs, max_n_label['value']



def parallel_motif_worker(x):
    return motif_subgraph_extraction_labeling(*x)

@ray.remote
def parallel_ray_motif_worker(indx, subx, input_graph, g_label, distance_label_mode, motif_edge_mode, h, max_nodes_per_hop, node_information):
    return motif_subgraph_extraction_labeling(indx, subx, input_graph, g_label, distance_label_mode, motif_edge_mode, h, max_nodes_per_hop, node_information)

def motif_subgraph_extraction_labeling(ind, sub, input_graph, g_label, distance_label_mode, motif_edge_mode, h=1, max_nodes_per_hop=None,

                                 node_information=None):
    # 'ind': 
    # extract the h-hop enclosing subgraph around motif formed by 'ind' set of vertices    
    dist = 0
    k = len(ind)

    nodes = set(ind)
    visited = set(ind)
    fringe = set(ind)

    nodes_dist = [0] * k

    #extract h-hop enclosing subgraph
    for dist in range(1, h+1):
        old_fringe = fringe.copy()
        for node in old_fringe:
            fringe.update(set(input_graph.neighbors(node)))

        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break        

        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe) 

    # put motif nodes on top
    for i in ind:
        if i in nodes:
            nodes.remove(i)

 
    nodes = list(ind) + list(nodes)

    #relabeling for subgraph
    relabel_mapping = {}
    for i in range(0,len(nodes)):
        relabel_mapping.update({nodes[i]:i})

    #extract and relabel subgraph
    g  = input_graph.subgraph(nodes)
    g = nx.relabel_nodes(g,relabel_mapping)

    # mask motif edges in subgraph
    mask_motif_edges_subgraph(g, sub, g_label, k, motif_edge_mode)

    #compute distance labels, which are inner label and outer label
    labels, distances = node_label_motif(g,sub,len(ind),distance_label_mode)

    # get input node features
    features = None
    if node_information is not None:
        features = node_information[nodes]

    if distance_label_mode:
        # generate final feature matrix X
        if features is not None:
            features = np.concatenate((distances,features),axis=1)
        else:
            features = distances
    elif features is None:
        features=np.array([])

    


    #report graph size
    # print('Subgraph:')
    # print('#nodes:',g.number_of_nodes(),"#edges",g.number_of_edges())
    
    # return g, labels, features
    # construct the inner label feature matrix XH, outer label feature matrix XL.
    gl = GNNGraph_pyG(g, g_label, labels, features)
    # print('Memory Consumption x: ',gl.x.storage().size())
    # print('Memory Consumption edge_index: ',gl.edge_index.storage().size())
   
    max_nl = max(labels)
    return gl, max_nl

    # 4.6 inner label and outer label
def node_label_motif(subgraph, motif_sub, k=3, distance_label_mode=False):
    
    #init variables
    K = subgraph.number_of_nodes()
    distances = None
    subgraph_truncated = subgraph

    # options: 0 = no distance labels, 1 = default distance labels, 2 = experimental distance labels, 3 = both labels
    if distance_label_mode == 1 or distance_label_mode == 3:

        #save all edges for motif vertices
        edges = []
        for i in range(0,k):
            edges.append(list(subgraph_truncated.edges(i)))
        edges.append([])

        #remove vertices v1 to vk
        for i in range(1,k):
            subgraph_truncated.remove_node(i)

       
        for i in range(0,k):

            #compute distance to vertex vi
            dist_to_i_dict = nx.single_source_shortest_path_length(subgraph_truncated, i)

            #create distance vector to vi
            dist_to_i = []
            for j in range (k,K):
                if j in dist_to_i_dict.keys():
                    dist_to_i.append(dist_to_i_dict[j])
                else:
                    dist_to_i.append(0)

            #append to distance matrix L
            dist_to_i_np = np.concatenate((np.array([0]*k), np.array(dist_to_i)))

            # outer label concat with a zero matrix
            if distances is None:
                distances = dist_to_i_np
            else:
                distances = np.concatenate((distances, dist_to_i_np))

            #remove node vi, readd node v(i+1)
            if len(edges[i+1]):
                subgraph_truncated.add_edges_from(edges[i+1])
            else:
                subgraph_truncated.add_node(i+1)

            for j in chain(range(i+1),range(i+2,k)):
                if subgraph_truncated.has_node(j):
                    subgraph_truncated.remove_node(j)          

        #create k x K matrix 
        # distances = distances.reshape((K,k),order='F')

        # add all vertices back to graph
        for i in range(0,k):
            subgraph_truncated.add_edges_from(edges[i])

        #check if all motif nodes were added back
        for i in range(0,k):
            if not subgraph_truncated.has_node(i):
                subgraph_truncated.add_node(i)

    if distance_label_mode == 2 or distance_label_mode == 3:
        #EXPERIMENTAL DISTANCE LABEL MODE

        # distance extraction to each of the k nodes involved in the motif
        for i in range(0,k):

            #compute distance to vertex vi
            dist_to_i_dict = nx.single_source_shortest_path_length(subgraph_truncated, i)

            #create distance vector to vi
            dist_to_i = []
            for j in range (k,K):
                if j in dist_to_i_dict.keys():
                    dist_to_i.append(dist_to_i_dict[j])
                else:
                    dist_to_i.append(0)

            #append to distance matrix L
            dist_to_i_np = np.concatenate((np.array([0]*k), np.array(dist_to_i)))

            if distances is None:
                distances = dist_to_i_np
            else:
                distances = np.concatenate((distances, dist_to_i_np))       
        
        #create k x K matrix 
        # distances = distances.reshape((K,k),order='F')

    if distance_label_mode == 1 or distance_label_mode == 2:
        #create k x K matrix 
        distances = distances.reshape((K,k),order='F')
    elif distance_label_mode == 3:
        #create 2k x K matrix 
        distances = distances.reshape((K,2*k),order='F')

    #apply inner label to inner vertices
    labels = []
    for i in range(1,k+1):
        labels.append(i)

    #concat inner label with zero matrix,
    labels = labels + [0] * (K-k)
   

    return labels, distances


def generate_node2vec_embeddings_motif(A, emd_size=128, negative_injection=False, train_pos=None, train_neg=None, train_neg_subgraphs=None, test_pos = None, test_pos_subgraphs = None, test_neg = None, test_neg_subgraphs = None):
    

    nx_G = nx.from_scipy_sparse_array(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=emd_size, window=10, min_count=0, sg=1, workers=8)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings


def count_motifs(graph_name):
    input_file_path = os.path.join(graph_name, "Count.out")
    output_file_path = os.path.join(graph_name, "motif.count")

    try:
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            for line in input_file:
                nums = [int(x) for x in line.split()]
                output_file.write(str(nums[0]) + " " +str(nums[1]+nums[2]) + " " + str(nums[3]) + " ")
                output_file.write(str(nums[4]+nums[5]) + " " + str(nums[6]+nums[7]) + " " +str(nums[8]) + " " + str(nums[9]+nums[10]+nums[11]) + " " + str(nums[12]+nums[13]) + " "+str(nums[14])+"\n")
        print(f"Motif counts written to {output_file_path}")
    except FileNotFoundError:
        print(f"File {input_file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def process_and_reconstruct_motifs(graph_name):
    input_path = os.path.join(graph_name, "motif.count")
    output_path = os.path.join(graph_name, "reconstruct_network_new")
    
    counts = [0 for _ in range(9)]
    try:
        with open(input_path, 'r') as input_file:
            for line in input_file:
                nums = [int(x) for x in line.split()]
                for i in range(9):
                    counts[i] += nums[i]
    except FileNotFoundError:
        print(f"File {input_path} not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading {input_path}: {e}")
        return
    
    try:
        with open(input_path, 'r') as input_file, open(output_path, 'w') as output:
            id = 0 
            for line in input_file:
                nums = [int(x) for x in line.split()]
                for i in range(9):
                    motif = f"motif{i}"
                    if counts[i] != 0 and nums[i] != 0:
                        weight = 1.0 * nums[i] / counts[i]
                        output.write(f"{motif} {id} {weight}\n")
                id += 1
        print(f"Reconstructed network written to {output_path}")
    except Exception as e:
        print(f"An error occurred while processing {input_path}: {e}")



def motif_count(A,train_pos=None, train_pos_subgraphs = None, train_neg=None, train_neg_subgraphs=None, test_pos = None, test_pos_subgraphs = None ,test_neg = None, test_neg_subgraphs = None, args = None):

    
    edges = set()
    for i in range(A.shape[0]):
        for j in range(i+1, A.shape[1]):
            if A[i, j] == 1:
               
                edge = (min(i, j), max(i, j))
                edges.add(edge)

    num_nodes = A.shape[0]
    num_edges = len(edges)

    with open('data/' + args.data_name + '/edges.txt', 'w') as outf:
        outf.write(f"{num_nodes} {num_edges}\n")
        for edge in edges:
            outf.write(f"{edge[0]} {edge[1]}\n")
    
   
    lib = ctypes.CDLL('./orca.so')

    
    lib.init.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))
    lib.init.restype = ctypes.c_int
    lib.run_count4.restype = None

    
    argc = 5
    argv = (ctypes.c_char_p * argc)()
    argv[0] = ctypes.create_string_buffer(b"orca.exe").raw
    argv[1] = ctypes.create_string_buffer(b"node").raw
    argv[2] = ctypes.create_string_buffer(b"4").raw
    
    input_file = "data/" + args.data_name + "/edges.txt"
    output_file = "data/" + args.data_name + "/Count.out"
    
    argv[3] = ctypes.create_string_buffer(input_file.encode('utf-8')).raw  
    argv[4] = ctypes.create_string_buffer(output_file.encode('utf-8')).raw  

    
    init_result = lib.init(argc, argv)
    if init_result == 0:
        print("Initialization failed")
    else:
        
        lib.run_count4()
    
    graph_name = "data/" + args.data_name
    count_motifs(graph_name)
    process_and_reconstruct_motifs(graph_name)


def generate_node2vecmotif_embeddings_motif(A, emd_size=128, negative_injection=False, train_pos=None, train_pos_subgraphs = None, train_neg=None, train_neg_subgraphs=None, test_pos = None, test_pos_subgraphs = None ,test_neg = None, test_neg_subgraphs = None, args = None):


    motif_count(A, train_pos, train_pos_subgraphs, train_neg, train_neg_subgraphs, test_pos, test_pos_subgraphs,test_neg, test_neg_subgraphs,args)
        
    input_file = "data/" + args.data_name + "/edges.txt"
    input_file1 = "data/" + args.data_name + "/reconstruct_network_new"
        
    nx_G, nx_G_new = read_graph(input_file, input_file1)
    G = Weighted_rm.Graph(nx_G, nx_G_new, index = 0.3, is_directed = False, p = 1, q = 1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)   
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=emd_size, window=10, min_count=0, sg=1, workers=8)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    
    return embeddings

def read_graph(input_file, input_file1):
    G = nx.read_edgelist(input_file, nodetype=str, data=(('weight', float),), create_using=nx.Graph())
    G_new = nx.read_edgelist(input_file1, nodetype=str, data=(('weight', float),), create_using=nx.Graph())
    
    return G, G_new

def print_results(args,acc_result,auc_result,sampling_time,extraction_time,gnn_time):

    #create filename
    if args.results_file_prefix != '':
        results_file = args.results_file_prefix+'_results1.txt'
    else:
        results_file = 'results.txt'
    
    write_column_names = True

    #check if file already exists
    if os.path.isfile(results_file): 
        with open(results_file,'r') as f:   
            if len(f.readlines()) > 0:
                write_column_names = False

    # write to file
    with open(results_file, 'a+') as f:
        #check if first row exists
        if write_column_names:
            f.write('input_graph prediction_method prediction_threshold motif_select motif_k motif_f hop motif_edge_mode distance_label_mode use_embedding naive_method aggregation_method learning_rate num_epochs max_nodes_per_hop RNG_seed test_ratio max_train_num batch_size max_cores sampling_time extraction_time gnn_time acc_result auc_result\n')    

        result = [args.data_name, args.prediction_method,args.prediction_threshold,args.motif,args.motif_k,args.motif_f,args.hop, args.motif_edge_mode, args.distance_label_mode, args.use_embedding, args.naive_method, args.aggregation, args.learning_rate, args.num_epochs, args.max_nodes_per_hop, args.seed,args.test_ratio,args.max_train_num,args.batch_size,args.max_cores,sampling_time,extraction_time,gnn_time,acc_result,auc_result]
        result_string = ' '.join(map(str, result)) 
        f.write(result_string+'\n')


###########################
# ORIGINAL SEAL FUNCTIONS #
###########################

def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None,
               all_unknown_as_negative=False):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)

    # sample positive links if not specified
    if train_pos is None and test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])  

    # sample negative links for train/test
    train_num = len(train_pos[0]) if train_pos else 0
    test_num = len(test_pos[0]) if test_pos else 0
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    if not all_unknown_as_negative:
        # sample a portion unknown links as train_negs and test_negs (no overlap)
        while len(neg[0]) < train_num + test_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg  = (neg[0][:train_num], neg[1][:train_num])
        test_neg = (neg[0][train_num:], neg[1][train_num:])
    else:
        # regard all unknown links as test_negs, sample a portion from them as train_negs
        while len(neg[0]) < train_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg  = (neg[0], neg[1])
        test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net==0, k=1))
        test_neg = (test_neg_i.tolist(), test_neg_j.tolist())
    return train_pos, train_neg, test_pos, test_neg


    
def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=1, 
                    max_nodes_per_hop=None, node_information=None, no_parallel=False):
    # automatically select h from {1, 2}
    if h == 'auto':
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(
            val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    
    def helper(A, links, g_label):
        g_list = []
        if no_parallel:
            for i, j in tqdm(zip(links[0], links[1])):
                g, n_labels, n_features = subgraph_extraction_labeling(
                    (i, j), A, g_label, h, max_nodes_per_hop, node_information
                )
                max_n_label['value'] = max(max(n_labels), max_n_label['value'])
                g_list.append(GNNGraph_pyG(g, g_label, n_labels, n_features))
            return g_list
        else:
            # the parallel extraction code

            start = time.time()
            '''
            pool = mp.Pool(max_cpu_count())
            results = pool.map_async(
                parallel_worker, 
                [((i, j), A, h, max_nodes_per_hop, node_information) for i, j in zip(links[0], links[1])]
            )
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            g_list = [GNNGraph_pyG(g, g_label, n_labels, n_features) for g, n_labels, n_features in results]
            max_n_label['value'] = max(
                max([max(n_labels) for _, n_labels, _ in results]), max_n_label['value']
            )
            '''
            A_x = ray.put(A)
            h_x = ray.put(h) # this might not exactly be needed
            max_nodes_per_hop_x = ray.put(max_nodes_per_hop)
            node_information_x = ray.put(node_information)
            g_label_x = ray.put(g_label)
            for gl, max_nl in tqdm(ray.get([parallel_ray_link_worker.remote((i, j), A_x, g_label_x, h_x, max_nodes_per_hop_x, node_information_x) for i, j in zip(links[0], links[1])]),total=len(links[0])):
                g_list.append(gl)
                max_n_label['value'] = max(max_nl, max_n_label['value'])
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
            return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs, test_graphs = None, None

    if train_pos and train_neg:
        train_graphs = helper(A, train_pos, 1) + helper(A, train_neg, 0)
    if test_pos and test_neg:
        test_graphs = helper(A, test_pos, 1) + helper(A, test_neg, 0)
    elif test_pos:
        test_graphs = helper(A, test_pos, 1)
    
    return train_graphs, test_graphs, max_n_label['value']

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)

@ray.remote
def parallel_ray_link_worker(ind, A, g_label, h, max_nodes_per_hop, node_information):
    return subgraph_extraction_labeling(ind, A, g_label, h, max_nodes_per_hop, node_information)

def subgraph_extraction_labeling(ind, A, g_label, h=1, max_nodes_per_hop=None,
                                 node_information=None):

    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes) 
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]

    if features is None:
        features = np.array([])    
    # construct nx graph
    g = nx.from_scipy_sparse_array(subgraph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    
    
    gl = GNNGraph_pyG(g, g_label, labels.tolist(), features)
    max_nl = max(labels.tolist())
    # return g, labels.tolist(), features
    return gl, max_nl


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels

def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1  # inject negative train
        A[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_array(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=emd_size, window=10, min_count=0, sg=1, 
            workers=8)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings

def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)
    
        
def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)


def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

