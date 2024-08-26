import networkx as nx
import torch
from math import floor, ceil, fsum
import numpy as np
import itertools
from sklearn import metrics
from itertools import combinations
from tqdm import tqdm
from decimal import *

def naive_dense_motif_prediction(g,
                            vertex_set,
                            missed_edges,
                            score = 'jaccard',
                            edge_threshold = 1,
                            naive_edge_mode = 0):

    k = len(vertex_set)
    all_possible_edges_num = (k*(k-1))//2
    min_edge_num = ceil(all_possible_edges_num*edge_threshold)
    
    edge_storage = []
    if naive_edge_mode == 1:
        for u, v in missed_edges:
            if g.has_edge(vertex_set[u],vertex_set[v]):
                g.remove_edge(vertex_set[u],vertex_set[v])
                edge_storage.append((vertex_set[u],vertex_set[v]))
    
    goal_edges = []
    for u, v in missed_edges:
        goal_edges.append((vertex_set[u], vertex_set[v]))

    if score == 'jaccard':
        link_pred_scores = nx.jaccard_coefficient(g, goal_edges)
    elif score == 'cn':
        for vertex in vertex_set:
            g.nodes[vertex]['community'] = vertex
        link_pred_scores = nx.cn_soundarajan_hopcroft(g, goal_edges)
    elif score == 'aa':
        link_pred_scores = nx.adamic_adar_index(g, goal_edges)
    
    for u, v in edge_storage:
        g.add_edge(u,v)

    getcontext().prec = 64
    inf_norm = Decimal(1)
    edge_scores = []
    for _,_,p in link_pred_scores:
        edge_scores.append(Decimal(p))
        if p > inf_norm:
            inf_norm = Decimal(p)
            
    if inf_norm > 1:
        for i in range(len(edge_scores)):
            edge_scores[i] = edge_scores[i]/inf_norm
    
    curr_edge_num = all_possible_edges_num - len(edge_scores)

    if curr_edge_num >= min_edge_num:
        return 1
    
    tbl = min_edge_num*[Decimal(0)]
    tbl[curr_edge_num] = Decimal(1)
    
    for edge_score in edge_scores:
        for i in reversed(range(1,min_edge_num)):
            tbl[i] = tbl[i]*(1-edge_score) + tbl[i-1]*edge_score
        tbl[0] = tbl[0]*(1-edge_score)
    
    return (1-min(fsum(tbl),1))
    
def naive_motif_score(g,
                      vertex_set,
                      missed_edges,
                      motif_type='clique',
                      score='jaccard',
                      aggregation='mul',
                      edge_threshold=1,
                      naive_edge_mode=0):


    if len(missed_edges) == 0:
        return 1

    k = len(vertex_set)
    if motif_type == 'clique':
        motif_edge_num = k*(k-1)/2
        center_vertices = vertex_set
    elif motif_type == 'star' or motif_type == 'dbstar':
        motif_edge_num = k-1
        center_vertices = [vertex_set[0]]
    elif motif_type == 'circle':
        motif_edge_num = k
        center_vertices = vertex_set
    elif motif_type == 'path':
        motif_edge_num = k-1
        center_vertices = vertex_set
    elif motif_type == 'tailed_triangle':
        motif_edge_num = k
        center_vertices = vertex_set
    elif motif_type == 'chordal_cycle':
        motif_edge_num = k*(k-1)/2-1
        center_vertices = vertex_set

    goal_edges = []
    for u, v in missed_edges:
        goal_edges.append((vertex_set[u], vertex_set[v]))
    
    edge_storage = []
    if naive_edge_mode == 1:
        for u, v in missed_edges:
            if g.has_edge(vertex_set[u],vertex_set[v]):
                g.remove_edge(vertex_set[u],vertex_set[v])
                edge_storage.append((vertex_set[u],vertex_set[v]))
    
    goal_edges_scores = []
    inf_norm = 1
    if score == 'jaccard':
        link_pred_scores = nx.jaccard_coefficient(g, goal_edges)
    elif score == 'cn':
        for vertex in vertex_set:
            g.nodes[vertex]['community'] = vertex
        link_pred_scores = nx.cn_soundarajan_hopcroft(g, goal_edges)
    elif score == 'aa':
        link_pred_scores = nx.adamic_adar_index(g, goal_edges)



    for u, v, p in link_pred_scores:
        if p > inf_norm:
            inf_norm = p
        goal_edges_scores.append((u, v, p))
        
    for u, v in edge_storage:
        g.add_edge(u,v)

    # Score aggregation
    if aggregation == 'mul':
        motif_score = 1

        for u, v, p in goal_edges_scores:
            normalized_score = p/inf_norm
            if u not in center_vertices and v not in center_vertices: #deal-breaker
                motif_score *= (1-normalized_score)
            else:
                motif_score*= normalized_score
    elif aggregation == 'avg':

        # ignore already existing links, only compute average of the links which we have to predict.
        motif_score = 0
        edge_count = 0

        for u, v, p in goal_edges_scores:
            normalized_score = p/inf_norm
            if u not in center_vertices and v not in center_vertices: #deal-breaker
                motif_score += 1-normalized_score
                edge_count += 1
            else:
                motif_score += normalized_score
                edge_count += 1

        motif_score= motif_score / edge_count

    else:
        #min score
        motif_score = 1

        for u, v, p in goal_edges_scores:
            normalized_score = p/inf_norm
            if u not in center_vertices and v not in center_vertices: #deal-breaker
                motif_score = min(1-normalized_score,motif_score)
            else:
                motif_score = min(normalized_score,motif_score)

    motif_score = max(motif_score, 0)
    return motif_score

def naive_motif_pred(net,
                     test_pos, test_pos_subgraphs,
                     test_neg, test_neg_subgraphs,
                     motif_type, edge_threshold,
                     score_type,
                     aggregation,
                     threshold,
                     naive_edge_mode=1):

    prediction = []
    true = []

    for vertex_set, missed_edges in zip(test_pos, test_pos_subgraphs):
        prediction.append(naive_motif_score(net, vertex_set, missed_edges, 
                                            motif_type,
                                            score_type,
                                            aggregation,
                                            edge_threshold,
                                            naive_edge_mode))
        true.append(1)

    for vertex_set, missed_edges in zip(test_neg, test_neg_subgraphs):
        prediction.append(naive_motif_score(net, vertex_set, missed_edges,
                                         motif_type,
                                         score_type,
                                         aggregation,
                                         edge_threshold,
                                         naive_edge_mode))
        true.append(0)

    roc_auc_ = metrics.roc_auc_score(true, prediction)
    true = torch.Tensor(true)
    prediction = torch.Tensor(prediction)
    prediction[prediction >= threshold] = 1
    prediction[prediction < threshold] = 0
    acc_ = metrics.accuracy_score(true, prediction)

    # report final accuracy
    print('\033[95mFinal test performance using naive link prediction: acc %.5f auc %.5f\033[0m' % (
    acc_, roc_auc_))

    return acc_, roc_auc_

