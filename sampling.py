from __future__ import print_function
import numpy as np
import random
from sympy import N
from tqdm import tqdm
import os, sys, pdb, math, time
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
import scipy.sparse.csgraph as csg
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
import ray
from itertools import combinations
import util_functions as util



def sample_neg_motif(net, test_ratio=0.1, motif='star', k=3, f=0.9, motif_edge_mode=1,dynamic_test_ratio = False, train_pos=None, test_pos=None, max_train_num=None, max_motifs_per_node=None,
               all_unknown_as_negative=False, no_parallel=False):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)

    # construct nx graph
    g = nx.from_scipy_sparse_array(net)
    ref_g = ray.put(g)
    print('Input graph: %d / %d (Nodes/Edges).'%(len(g.nodes),len(g.edges)))

    # init list of motifs and negative motifs
    motifs = []
    motifs_subgraph = []
    neg = []
    neg_subgraph = []
    
    #get number of nodes
    n = net.shape[0]
    
    start = time.perf_counter()

    nsets = n*[set()]
    def preprocess_neighborhoods():
        for node in list(nx.nodes(g)):
            nsets[node] = {x for x in list(nx.neighbors(g,node)) if x > node}
        
        
    if motif == 'star':
        k -= 1  

        neighbors = []
        nodes = list(nx.nodes(g))
        random.shuffle(nodes)  
        
        max_samples =6000000  
        
        for node in nodes:
            if len(motifs) >= max_samples:
                break

            neighbors = list(nx.neighbors(g, node))
            neighbors.sort()

            if max_motifs_per_node is not None and math.comb(len(neighbors), k) > 2 * max_motifs_per_node:
                temp_motifs = set()
                while len(temp_motifs) < max_motifs_per_node and len(motifs) < max_samples:
                    temp_motifs.add(tuple([node] + random.sample(neighbors, k)))
                motifs.extend([list(cand) for cand in temp_motifs])
            else:
                temp_motifs = [([node] + list(cands)) for cands in combinations(neighbors, k)]
                if max_motifs_per_node is not None and len(temp_motifs) > max_motifs_per_node:
                    random.shuffle(temp_motifs)
                    temp_motifs = temp_motifs[:max_motifs_per_node]
                motifs.extend(temp_motifs)
                if len(motifs) >= max_samples:
                    motifs = motifs[:max_samples]
                    break
                    
        mid = time.perf_counter()  
    
        print('Found', len(motifs), 'motifs.')
        
        if len(motifs) < 200:
            print('Not enough motifs found! Aborting Execution.')
            quit(0)

        print('Sampling negative motifs for train and test')
        
        ref_motifs = ray.put(motifs)
        
        remote_tasks = [star_negative.remote(ref_g, ref_motifs, k, count) for count in chunk_int(len(motifs))]
        for i, ref in enumerate(remote_tasks):
            neg.extend(ray.get(ref))
            remote_tasks[i] = None
            del ref

        presub = time.perf_counter()
        
        if motif_edge_mode == 1:
            for sample in neg:
                subgraph = []
                for i in range(1, k + 1):
                    if not g.has_edge(sample[0], sample[i]):
                        subgraph.append((0, i))
                neg_subgraph.append(subgraph)
            
            motifs_subgraph = neg_subgraph.copy()
            random.shuffle(motifs_subgraph)
        elif motif_edge_mode == 0:
            subgraph = [(0, x) for x in range(1, k + 1)]
            motifs_subgraph = len(motifs) * [subgraph]
            neg_subgraph = len(neg) * [subgraph]
        elif motif_edge_mode == 2:
            motifs_subgraph = len(motifs) * [[]]
            neg_subgraph = len(neg) * [[]]

        k += 1

    if motif == 'circle':
        
        max_circles_per_node = max_motifs_per_node

        
        preprocess_neighborhoods()
        def dfs(node, start_node, path, visited, seen):
            if len(path) == k and path[0] == start_node and g.has_edge(path[-1], path[0]):
                circle = tuple(path)
                if circle not in seen:  
                    motifs.append(list(path))  
                    seen.add(circle)  
                return
            if len(path) >= k or node in visited:
                return

            visited.add(node) 

            for neighbor in nsets[node]:
                dfs(neighbor, start_node, path + [neighbor], visited, seen)

            visited.remove(node)  

        
        seen = set()  
        for node in nx.nodes(g):
            dfs(node, node, [node], set(), seen)

        mid = time.perf_counter()

        
        print('Found', len(motifs), 'circle motifs.')
        

        
        if len(motifs) < 200:
            print('Not enough motifs found! Aborting Execution.')
            quit(0)

       
        print('Sampling negative motifs for train and test')

        
        ref_motifs = ray.put(motifs)

        
        remote_tasks = [circle_negative.remote(ref_g, ref_motifs, k, count) for count in chunk_int(len(motifs))]
        for i, ref in enumerate(remote_tasks):
            neg.extend(ray.get(ref))
            remote_tasks[i] = None
            del ref

        presub = time.perf_counter()

        
        print('Sampled', len(neg), 'negative motifs.')

        
        if motif_edge_mode == 1:
            for sample in neg:
                subgraph = []
                for i in range(k):
                    for j in range(i + 1, k):
                        if not g.has_edge(sample[i], sample[j]):
                            subgraph.append((i, j))
                neg_subgraph.append(subgraph)

            motifs_subgraph = neg_subgraph.copy()
            random.shuffle(motifs_subgraph)
        elif motif_edge_mode == 0:
            subgraph = []
            for a in range(0, k):
                for b in range(a + 1, k):
                    subgraph.append((a, b))
            motifs_subgraph = len(motifs) * [subgraph]
            neg_subgraph = len(neg) * [subgraph]
        elif motif_edge_mode == 2:
            motifs_subgraph = len(motifs) * [[]]
            neg_subgraph = len(neg) * [[]]


    if motif == 'path':

       
        preprocess_neighborhoods()
        
       
        def dfs_paths(node, path, visited, seen):
            if len(path) == k: 
                path_tuple = tuple(path)
                reverse_path_tuple = tuple(path[::-1]) 
                if path_tuple not in seen and reverse_path_tuple not in seen:
                    motifs.append(list(path))  
                    seen.add(path_tuple)
                    seen.add(reverse_path_tuple)
                return
            if len(path) > k or node in visited:  
                return

            visited.add(node)  

            for neighbor in nsets[node]:
                dfs_paths(neighbor, path + [neighbor], visited, seen)

            visited.remove(node)  

        
        for node in nx.nodes(g):
            dfs_paths(node, [node], set(), set())

        mid = time.perf_counter()

        if len(motifs) < 200:
            print('Not enough motifs found! Aborting Execution.')
            quit(0)

        print('Sampling negative motifs for train and test')

        ref_motifs = ray.put(motifs)

        remote_tasks = [path_negative.remote(ref_g, ref_motifs, k, count) for count in chunk_int(len(motifs))]
        for i, ref in enumerate(remote_tasks):
            neg.extend(ray.get(ref))
            remote_tasks[i] = None
            del ref

        presub = time.perf_counter()

        print('Sampled', len(neg), 'negative motifs.')

        if motif_edge_mode == 1:
            for sample in neg:
                subgraph = []
                for i in range(k):
                    for j in range(i + 1, k):
                        if not g.has_edge(sample[i], sample[j]):
                            subgraph.append((i, j))
                neg_subgraph.append(subgraph)

            motifs_subgraph = neg_subgraph.copy()
            random.shuffle(motifs_subgraph)
        elif motif_edge_mode == 0:
            subgraph = []
            for a in range(0, k):
                for b in range(a + 1, k):
                    subgraph.append((a, b))
            motifs_subgraph = len(motifs) * [subgraph]
            neg_subgraph = len(neg) * [subgraph]
        elif motif_edge_mode == 2:
            motifs_subgraph = len(motifs) * [[]]
            neg_subgraph = len(neg) * [[]]
        
    if motif == 'tailed_triangle':

        
        preprocess_neighborhoods()

        def dfs_tailed_triangle(node, path, visited, seen):
            if len(path) == 4:
                if len(set(path)) == 4: 
                    node1, node2, node3, node4 = path
                    
                    if node2 in g.neighbors(node1) and node3 in g.neighbors(node1) and node3 in g.neighbors(node2):
                        
                        if any(node4 in g.neighbors(node) for node in (node1, node2, node3)):
                            
                            if tuple(path) not in seen:  
                                motifs.append(path)  
                                seen.add(tuple(path))  
                return

            visited.add(node)  

            
            for neighbor in nsets[node]:
                if neighbor not in path:  
                    dfs_tailed_triangle(neighbor, path + [neighbor], visited, seen)
            visited.remove(node)  

       
        for node in nx.nodes(g):
            dfs_tailed_triangle(node, [node], set(), set())

        mid = time.perf_counter()
        print('Found', len(motifs), 'tailed triangle motifs:')
        


       
        if len(motifs) < 200:
            print('Not enough motifs found! Aborting Execution.')
            quit(0)

        
        print('Sampling negative motifs for train and test')

        
        ref_motifs = ray.put(motifs)

        
        remote_tasks = [tailed_triangle_negative.remote(ref_g, ref_motifs, k, count) for count in chunk_int(len(motifs))]
        for i, ref in enumerate(remote_tasks):
            neg.extend(ray.get(ref))
            remote_tasks[i] = None
            del ref

        presub = time.perf_counter()

        
        if motif_edge_mode == 1:
            for sample in neg:
                subgraph = []
                for i in range(k):
                    for j in range(i+1,k):
                        if not g.has_edge(sample[i],sample[j]):
                            subgraph.append((i,j))
                neg_subgraph.append(subgraph)
            
            motifs_subgraph = neg_subgraph.copy()
            random.shuffle(motifs_subgraph)
        elif motif_edge_mode == 0:
            subgraph = []
            for a in range(0,k):
                for b in range(a+1,k):
                    subgraph.append((a,b))
            motifs_subgraph = len(motifs)*[subgraph]
            neg_subgraph = len(neg)*[subgraph]
        elif motif_edge_mode == 2:
            motifs_subgraph = len(motifs)*[[]]
            neg_subgraph = len(neg)*[[]]
    

    #k-clique
    if motif=='clique':
        #search k-clique motif
        
        max_cliques_per_node = max_motifs_per_node 
        
        # create neighborhood datastructure for efficient querying
        preprocess_neighborhoods()
        
        # this helper function recursively finds cliques
        def helper(cands, nhood, count, threshold):
            if count == 1: # only one vertex is missing
                for node in nhood:
                    if (not threshold is None) and len(motifs) == threshold:
                        return
                    motifs.append(cands+[node]) # add each common neighbor to get a clique
            else: 
                if len(nhood) < count: # not enough common neighbors
                    return
                for node in nhood: # for each common neighbor, add it and recurse
                    if (not threshold is None) and len(motifs) == threshold:
                        return
                    new_nhood = nhood & nsets[node]
                    cands.append(node)
                    helper(cands,new_nhood,count-1,threshold)
                    cands.pop()
        
        # find all cliques starting where the lowest vertex is "node"
        for node in list(nx.nodes(g)):
            helper([node],nsets[node],k-1,len(motifs)+max_cliques_per_node if not max_cliques_per_node is None else None)
        
        mid = time.perf_counter()

        if len(motifs) < 200:
            print('Not enough motifs found! Aborting Execution.')
            quit(0)
        # sample negative motifs for train/test
        print('sampling negative motifs for train and test')
        
        ref_motifs = ray.put(motifs)
        
        # generate negative samples in parallel
        #4.3 negative sampling：clique_negative.remote() function
        remote_tasks = [clique_negative.remote(ref_g,ref_motifs,k,count) for count in chunk_int(len(motifs))]
        for i,ref in enumerate(remote_tasks):
            neg.extend(ray.get(ref))
            remote_tasks[i] = None
            del ref
        
        presub = time.perf_counter() 
        
        # generate subgraphs
        if motif_edge_mode == 1:
            for sample in neg:
                subgraph = []
                for i in range(k):
                    for j in range(i+1,k):
                        if not g.has_edge(sample[i],sample[j]):
                            subgraph.append((i,j))
                neg_subgraph.append(subgraph)
            
            motifs_subgraph = neg_subgraph.copy()
            random.shuffle(motifs_subgraph)
        elif motif_edge_mode == 0:
            subgraph = []
            for a in range(0,k):
                for b in range(a+1,k):
                    subgraph.append((a,b))
            motifs_subgraph = len(motifs)*[subgraph]
            neg_subgraph = len(neg)*[subgraph]
        elif motif_edge_mode == 2:
            motifs_subgraph = len(motifs)*[[]]
            neg_subgraph = len(neg)*[[]]
        
    if motif == 'chordal_cycle':
        preprocess_neighborhoods()
        
        def dfs_paths(node, start_node, path, visited, seen):
            if len(path) == k and path[0] == start_node and g.has_edge(path[-1], path[0]):

               
                diagonal_edges = 0
                if g.has_edge(path[0], path[2]):
                    diagonal_edges += 1
                if g.has_edge(path[1], path[3]):
                    diagonal_edges += 1

                if diagonal_edges == 1:
                    cycle = tuple(path)
                    if cycle not in seen:
                        motifs.append(list(path))  
                        seen.add(cycle)
                return

            if len(path) > 4 or node in visited:  
                return

            visited.add(node)  

            for neighbor in nsets[node]:
                dfs_paths(neighbor, start_node, path + [neighbor], visited, seen)

            visited.remove(node)  

        
        for node in g.nodes():
            dfs_paths(node, node, [node], set(), set())

        
        mid = time.perf_counter()

        
        if len(motifs) < 200:
            print('Not enough motifs found! Aborting Execution.')
            quit(0)

       
        print('Sampling negative motifs for train and test')

        
        ref_motifs = ray.put(motifs)

        
        remote_tasks = [circle_negative.remote(ref_g, ref_motifs, k, count) for count in chunk_int(len(motifs))]
        for i, ref in enumerate(remote_tasks):
            neg.extend(ray.get(ref))
            remote_tasks[i] = None
            del ref

        presub = time.perf_counter()

        
        if motif_edge_mode == 1:
            for sample in neg:
                subgraph = []
                for i in range(k):
                    for j in range(i+1,k):
                        if not g.has_edge(sample[i],sample[j]):
                            subgraph.append((i,j))
                neg_subgraph.append(subgraph)
            
            motifs_subgraph = neg_subgraph.copy()
            random.shuffle(motifs_subgraph)
        elif motif_edge_mode == 0:
            subgraph = []
            for a in range(0,k):
                for b in range(a+1,k):
                    subgraph.append((a,b))
            motifs_subgraph = len(motifs)*[subgraph]
            neg_subgraph = len(neg)*[subgraph]
        elif motif_edge_mode == 2:
            motifs_subgraph = len(motifs)*[[]]
            neg_subgraph = len(neg)*[[]]

            
    #f-dense graph with k vertices
    if motif=='dense':
    
        # fix a number of samples to generate
        sample_max = 1000000
        if not max_train_num is None:
            sample_max = 10*max_train_num

        print('cpus: ',util.max_cpu_count())    
        
        min_cnt = k*(k-1)//2*f
        
        # generate samples in parallel
        remote_tasks = [dense_sampling.remote(ref_g,k,min_cnt,count) for count in chunk_int(sample_max)]
        for i,(ref_pos,ref_neg) in enumerate(remote_tasks):
            motifs.extend(ray.get(ref_pos))
            neg.extend(ray.get(ref_neg))
            remote_tasks[i] = None
            del ref_pos
            del ref_neg
        del remote_tasks
        
        mid = time.perf_counter()
        
        # renmoving duplicates
        set_motifs = set()
        set_neg = set()
        
        for cnt,sample in motifs:
            set_motifs.add((cnt,tuple(sample)))
        motifs = []
        for cnt,sample in set_motifs:
            motifs.append((cnt,list(sample)))
        del set_motifs
        
        for cnt,sample in neg:
            set_neg.add((cnt,tuple(sample)))
        neg = []
        for cnt,sample in set_neg:
            neg.append((cnt,list(sample)))
        del set_neg
        
        # preferentially sample subgraphs with a density close to the threshold
        def poly_sample(arr,size):
            length = len(arr)-1
            k = max(2,int(len(arr)/size)//2)
            out = [arr[0]]
            _a = size/(length**k)
            step = 1
            for i in range(0,length):
                j = length - i
                val = _a*(i**k)
                if val >= step:
                    step += 1
                    out.append(arr[j])
                else:
                    pass
            return out
        
        # balancing dataset
        tot = min(len(motifs),len(neg))
        if len(motifs) > len(neg):
            motifs = sorted(motifs,key=lambda x:x[0])
            if len(motifs) < len(neg)*2:  
                random.shuffle(motifs)
                motifs = motifs[:tot]
            else:
                motifs = poly_sample(motifs,tot)
        else:
            neg = sorted(neg,key=lambda x:x[0],reverse=True)
            if len(neg) < len(motifs)*2: 
                random.shuffle(neg)
                neg = neg[:tot]
            else:
                neg = poly_sample(neg,tot)
        motifs = list(map(lambda x:x[1],motifs))
        neg = list(map(lambda x:x[1],neg))

        if len(motifs) < 200:
            print('No motifs found! Aborting Execution.')
            quit(0)
        
        presub = time.perf_counter()
        
        # generate subgraphs
        if motif_edge_mode == 1:
            counts = []
            for sample in neg:
                cnt = 0
                subgraph = []
                for i in range(k):
                    for j in range(i+1,k):
                        if g.has_edge(sample[i],sample[j]):
                            cnt += 1
                        else:
                            subgraph.append((i,j))
                counts.append(cnt)
                neg_subgraph.append(subgraph)
                
            random.shuffle(counts)
            for i,sample in enumerate(motifs):
                target = counts[i]
                cnt = 0
                options = []
                subgraph = []
                for i in range(k):
                    for j in range(i+1,k):
                        if g.has_edge(sample[i],sample[j]):
                            cnt += 1
                            options.append((i,j))
                        else:
                            subgraph.append((i,j))
                while cnt > target:
                    cnt -= 1
                    cand = random.choice(options)
                    options.remove(cand)
                    subgraph.append(cand)
                motifs_subgraph.append(subgraph)
        elif motif_edge_mode == 0:
            print('Warning: The subgraphs for masking is the complete graph induced by the motif vertices! This might produce irrelevant results with naive predictors.')
            subgraph = []
            for a in range(0,k):
                for b in range(a+1,k):
                    subgraph.append((a,b))
            motifs_subgraph = len(motifs)*[subgraph]
            neg_subgraph = len(neg)*[subgraph]
        elif motif_edge_mode == 2:
            motifs_subgraph = len(motifs)*[[]]
            neg_subgraph = len(neg)*[[]]
    


    print('Found %d motifs. Execution time: (%.2fs, positive motifs / %.2fs, negative motifs / %.2fs, subgraphs)'%(len(motifs),mid-start,presub-mid,time.perf_counter()-presub)) 
    if len(neg_subgraph) == len(neg):
        temp = list(zip(neg, neg_subgraph))
        random.shuffle(temp)
        neg, neg_subgraph = zip(*temp)
    else:
        print('Warning: Dangerous shuffle!')
        random.shuffle(neg)
        neg_subgraph = len(neg)*[]

    # randomly sample positive motifs and split them
    if train_pos is None and test_pos is None:
        if len(motifs) == len(motifs_subgraph):
            temp = list(zip(motifs, motifs_subgraph))
            random.shuffle(temp)
            motifs, motifs_subgraph = zip(*temp)
        else:
            print('Warning: Dangerous shuffle!')
            random.shuffle(motifs)
            motifs_subgraph = len(motifs)*[]
            
        # split positive motifs
        split = int(math.ceil(len(motifs) * (1 - test_ratio)))
        train_pos = motifs[:split]
        train_pos_subgraphs = motifs_subgraph[:split]
        test_pos = motifs[split:]
        test_pos_subgraphs = motifs_subgraph[split:]
        
        #split negative motifs
        train_neg  = neg[:split]
        train_neg_subgraphs = neg_subgraph[:split]
        test_neg = neg[split:]
        test_neg_subgraphs = neg_subgraph[split:]

    else:
        #split negative motifs like the provided training data
        train_neg  = neg[:len(train_pos)]
        train_neg_subgraphs = neg_subgraph[:len(train_pos)]
        test_neg = neg[len(train_pos):]
        test_neg_subgraphs = neg_subgraph[len(train_pos):]

    # if max_train_num is set, truncate the training data to max_train_num
    if max_train_num is not None:
        # ??? second permutation needed ???
        train_pos = train_pos[:max_train_num]
        train_pos_subgraphs = train_pos_subgraphs[:max_train_num]
        train_neg = train_neg[:max_train_num]
        train_neg_subgraphs = train_neg_subgraphs[:max_train_num]
        
        test_pos = test_pos[:int(max_train_num*test_ratio)]
        test_pos_subgraphs = test_pos_subgraphs[:int(max_train_num*test_ratio)]
        test_neg = test_neg[:int(max_train_num*test_ratio)]  
        test_neg_subgraphs = test_neg_subgraphs[:int(max_train_num*test_ratio)]
        
    print('#Train: %d, #Test: %d'%(len(train_pos)+len(train_neg), len(test_pos)+len(test_neg))) 
    # print("Extracted circle motifs:", motifs)

    
    return train_pos, train_pos_subgraphs, train_neg, train_neg_subgraphs, test_pos, test_pos_subgraphs, test_neg, test_neg_subgraphs

# split a list into chunks for parallel processing
def chunk_list(lst):
    n = len(lst)
    c = util.max_cpu_count()
    b = n//c+1
    res = []
    for i in range(c-1):
        res.append(lst[i*b:(i+1)*b])
    res.append(lst[(c-1)*b:])
    return res

def chunk_int(n):
    c = util.max_cpu_count()
    b = n//c+1
    res = (c-1)*[b]
    res.append(n-(c-1)*b)
    return res

# generates a random sample by sampling random vertices
def random_sample(g,n,size):
    cands = random.sample(range(n),size)
    star = random.choice(cands)
    cands.remove(star)
    cands.sort()
    return ([star]+cands)

# generates a (more) connected sample
def connected_sample(g,n,size):
    cand = random.randint(0,n-1)
    cands = {cand}
    neighbors = list(nx.neighbors(g,cand))
    while len(cands) < size:
        if len(neighbors) > 0:
            neighbor = random.choice(list(neighbors))
            if neighbor in cands:
                continue
            cands.add(neighbor)
            neighbors.extend(list(nx.neighbors(g,neighbor)))
        else:
            rnd = random.randint(0,n-1)
            if not rnd in cands:
                cands.add(rnd)
                neighbors.extend(list(nx.neighbors(g,neighbor)))
    cands = list(cands)
    star = random.choice(cands)
    cands.remove(star)
    cands.sort()
    return ([star]+cands)

@ray.remote
def star_negative(g, motifs, k, count):
    n = g.number_of_nodes()
    res = []
    split = [0.8,0.9]
    #4.3 three sampling strategies.8:1:1 
    # (1) select positive samples and then remove a few vertices, replacing them with other nearby vertices
    while len(res) < split[0]*count:
        kstar = random.choice(motifs)
        index = random.randint(1, k)
        outer = kstar[index]
        cands = list(nx.neighbors(g,outer))
        for cand in random.sample(cands,min(k,len(cands))):
            if (not g.has_edge(cand,kstar[0])) and not (cand in kstar):
                neighbors = kstar[1:]
                neighbors[index-1] = cand
                neighbors.sort()
                res.append([kstar[0]]+neighbors)
                break
    # (3) We select a random vertex 𝑟 into an empty set,and then we keep adding randomly selected vertices from the union over the neighborhoods of vertices already in the set, growing a subgraph until reaching the vettices size
    while len(res) <= split[1]*count:
        
        cands = connected_sample(g,n,k+1)

        for i in range(1,k+1):
            if not (g.has_edge(cands[0],cands[i])):
                res.append(cands)
                break
    
    # (2) randomly select VM vertices
    while len(res) < count:
        
        cands = random_sample(g,n,k+1)

        for i in range(1,k+1):
            if not (g.has_edge(cands[0],cands[i])):
                res.append(cands)
                break
    return res

@ray.remote
def circle_negative(g, motifs, k, count):
    res = []
    n = g.number_of_nodes()
    
   
    split = [0.8, 0.9]
    while len(res) < split[0] * count:
        circle = random.choice(motifs)
        index = random.randint(0, len(circle) - 1)   
        outer = circle[index]
        neighbors = list(nx.neighbors(g, outer))
        for cand in random.sample(neighbors, min(k, len(neighbors))):
            if not (cand in circle):
                new_neighbors = circle[:]
                new_neighbors[index] = cand
                
                res.append(new_neighbors)
                break
            
        
    def contains_4_circle(g, subgraph_nodes):
        subgraph = g.subgraph(subgraph_nodes)
        cycles = list(nx.cycle_basis(subgraph))
        for cycle in cycles:
            if len(cycle) == 4:
                return True
        return False

    
    while len(res) < split[1] * count:
        rn = connected_sample(g, n, k)
        
        if not contains_4_circle(g, rn):
            res.append(rn)

    
    while len(res) < count:
        rn = random_sample(g, n, k)
        
        if not contains_4_circle(g, rn):
            res.append(rn)



    return res



@ray.remote
def clique_negative(g, motifs, k, count):
    res = []
    n = g.number_of_nodes()
    
    
    split = [0.8, 0.9]
    # generates samples similar to a clique
    while len(res) < split[0]*count:
        clique = random.choice(motifs) # take a clique
            
        # select a random vertex and remove it
        sel = random.choice(clique)
        clique = set(clique)
        clique.remove(sel)
            
        neighbors = list(nx.neighbors(g,sel))
        
        # find a suitable vertex to add instead
        best = None
        best_val = 0
        for node in neighbors:
            if node in clique:
                continue
            val = 0
            for c in clique:
                if g.has_edge(node,c):
                    val += 1
            if val > best_val and val != k-1:
                best_val = val
                best = node
        if not best is None:
             clique.add(best)
             clique = list(clique)
             clique.sort()
             res.append(clique)

    # samples any random preferrably connected graphs
    while len(res) < split[1]*count:
        
        # sampling k preferrably connected nodes in sorted array
        rn = connected_sample(g,n,k)
        
        # test for motif
        accepted = False
        for i in range(k):
            for j in range(i+1,k):
                if not g.has_edge(rn[i],rn[j]):
                    res.append(rn)
                    accepted = True
                    break
            if accepted:
                break
    
    # samples any random non-cliques (mainly disconnected vertices)
    while len(res) < count:
        
        # sampling k random nodes in sorted array
        rn = random_sample(g,n,k)
    
        #test for motif
        #if (not (g.has_edge(rn[0],rn[1]) and g.has_edge(rn[0],rn[2]) and g.has_edge(rn[1],rn[2]))): 
        
        accepted = False
        for i in range(k):
            for j in range(i+1,k):
                if not g.has_edge(rn[i],rn[j]):
                    res.append(rn)
                    accepted = True
                    break
            if accepted:
                break
    
    return res

@ray.remote
def tailed_triangle_negative(g, motifs, k, count):
    res = []
    n = g.number_of_nodes()

    
    split = [0.8, 0.9]

    
    while len(res) < split[0] * count:
        triangle = random.choice(motifs)
        index = random.randint(0,len(triangle)-1)
        outer = triangle[index]
        neighbors = list(nx.neighbors(g, outer))
        
        for cand in random.sample(neighbors, min(k, len(neighbors))):
            if not (cand in triangle):
                
                new_path = triangle[:]
                new_path[index] = cand
                res.append(new_path)
                break

    
    while len(res) < split[1] * count:
        rn = connected_sample(g, n, k)
       
        has_tailed_triangle = False
        for i in range(k):
            for j in range(i + 1, k):
                for l in range(j + 1, k):
                   
                    if g.has_edge(rn[i], rn[j]) and g.has_edge(rn[j], rn[l]) and g.has_edge(rn[l], rn[i]):
                        
                        neighbors = set(nx.neighbors(g, rn[i])) | set(nx.neighbors(g, rn[j])) | set(nx.neighbors(g, rn[l]))
                        for neighbor in [rn[i], rn[j], rn[l]]:
                            neighbors.remove(neighbor)
                        if neighbors:
                            has_tailed_triangle = True
                            break
                if has_tailed_triangle:
                    break
            if has_tailed_triangle:
                break
        if not has_tailed_triangle:
            res.append(rn)


    
    while len(res) < count:
        rn = random_sample(g, n, k)
        has_tailed_triangle = False
        for i in range(k):
            for j in range(i + 1, k):
                for l in range(j + 1, k):
                   
                    if g.has_edge(rn[i], rn[j]) and g.has_edge(rn[j], rn[l]) and g.has_edge(rn[l], rn[i]):
                        
                        neighbors = set(nx.neighbors(g, rn[i])) | set(nx.neighbors(g, rn[j])) | set(nx.neighbors(g, rn[l]))
                        for neighbor in [rn[i], rn[j], rn[l]]:
                            neighbors.remove(neighbor)
                        if neighbors:
                            has_tailed_triangle = True
                            break
                if has_tailed_triangle:
                    break
            if has_tailed_triangle:
                break
        if not has_tailed_triangle:
            res.append(rn)

    return res


@ray.remote
def path_negative(g, motifs, k, count):
    res = []
    n = g.number_of_nodes()

    
    split = [0.8, 0.9]
    
    
    while len(res) < split[0] * count:
        path = random.choice(motifs)
        index = random.randint(0, len(path)-1)  
        middle_node = path[index]
        neighbors = list(nx.neighbors(g, middle_node))
        for cand in random.sample(neighbors, min(k, len(neighbors))):
            
            if not (cand in path):
                new_path = path[:]  
                new_path[index] = cand  
                
                res.append(new_path)
                break
    
    
    while len(res) < split[1] * count:
        rn = connected_sample(g, n, k)
        
        has_path = False
        for i in range(k):
            for j in range(i + 1, k):
                if nx.shortest_path_length(g, source=rn[i], target=rn[j]) == k:
                    has_path = True
                    break
            if has_path:
                break
        if not has_path:
            res.append(rn)

    
    
    while len(res) < count:
        rn = random_sample(g, n, k)
        
        has_path = False
        for i in range(k):
            for j in range(i + 1, k):
                if nx.shortest_path_length(g, source=rn[i], target=rn[j]) == k:
                    has_path = True
                    break
            if has_path:
                break
        if not has_path:
            res.append(rn)

    return res

@ray.remote
def chordal_cycle_negative(g, motifs, k, count):
    res = []
    n = g.number_of_nodes()
    
    
    split = [0.8, 0.9]

    
    while len(res) < split[0] * count:
        cycle = random.choice(motifs)
        index = random.randint(0, len(cycle) - 1)
        middle_node = cycle[index]
        neighbors = list(nx.neighbors(g, middle_node))
        for cand in random.sample(neighbors, min(k, len(neighbors))):
           
            if not (cand in cycle):
                new_cycle = cycle[:]  
                new_cycle[index] = cand  
                
                res.append(new_cycle)
                break
    
        
    while len(res) < split[1] * count:
        rn = connected_sample(g, n, k)
        
        
        diagonal_edges = 0
        for i in range(k):           
            if g.has_edge(rn[i], rn[(i + 2) % k]):
                diagonal_edges += 1
            
       
        if diagonal_edges != 1:
            res.append(rn)
   
    while len(res) < count:
        rn = random_sample(g, n, k)

       
        diagonal_edges = 0
        for i in range(k):           
            if g.has_edge(rn[i], rn[(i + 2) % k]):
                diagonal_edges += 1
            
        
        if diagonal_edges != 1:
            res.append(rn)

    
    return res





@ray.remote(num_returns=2)
def dense_sampling(g,k,min_cnt,count):
    res_pos = []
    res_neg = []
    n = g.number_of_nodes()
    
        
    # preprocess nodes to find good starting candidates
    
    cores = [node for node,deg in g.degree() if deg > 2*k]
    dyn_corelevel = 2
    if len(cores) < n//k:
        cores = [node for node,deg in g.degree() if deg > k]
        dyn_corelevel = 1
    if len(cores) < n//k:
        cores = list(nx.nodes(g))
        dyn_corelevel = 0
    
    dyn_p = 1/k
    dyn_avg = min_cnt
    dyn_sum = 0
    dyn_check = k*4
    
    for it in range(85*count//100): 
        # choose an initial vertex and initialize the neighborhood
        cand = random.choice(cores)
        cands = {cand}
        neighbors = set(nx.neighbors(g,cand))
        
        # add vertices until the subgraph has sufficient vertices
        while len(cands) < k:
            # if there are neighbors, take neighbors, else random
            if len(neighbors) > 0:
                # find a good vertex to add
                neighbor = 0
                opt_val = -1 # = number of edges to current candidates
                opt_val2 = -1 # = degree of new vertex
                for _ in range(k):
                    cand = random.choice(list(neighbors))
                    val = len(cands & set(nx.neighbors(g,cand)))
                    val2 = g.degree(cand) # check for common neighbors instead?
                    if (val > opt_val or (val == opt_val and val2 > opt_val2)):
                        neighbor = cand
                        opt_val = val
                        opt_val2 = val2
                        
                # add the candidate and update neighborhood if suitable
                cands.add(neighbor)
                neighbors.remove(neighbor)
                not_enough_neighbors = len(neighbors) < k # need more neighbors
                randomness = random.random() < dyn_p
                if not_enough_neighbors or randomness:
                    neighbors = (neighbors | set(nx.neighbors(g,neighbor))) - cands 
            else:
                # choose and add a random vertex
                rnd = random.randint(0,n-1)
                if not rnd in cands:
                    cands.add(rnd)
                    neighbors = (neighbors | set(nx.neighbors(g,rnd))) - cands
        cands = list(cands)
        cands.sort()
        
        # count the number of edges
        cnt = 0
        for i in range(0,k):
            for j in range(i+1,k):
                if g.has_edge(cands[i],cands[j]):
                    cnt += 1
        dyn_sum += cnt
        # append the subgraph to the positive or negative samples
        if cnt >= min_cnt:
            res_pos.append((cnt,cands))
        else:
            res_neg.append((cnt,cands))
        # update dynamic targeting
        
        if it % dyn_check == dyn_check-1: # do parameter update to adjust sampled density
            
            dyn_avg = 0.8*dyn_avg + 0.2*dyn_sum/dyn_check # updates average
            
            if dyn_avg > min_cnt + 1: # if current samples are too dense, then increase dyn_p
                if dyn_avg > min_cnt + k:
                    dyn_p += 0.1 
                else:
                    dyn_p += 0.02
                dyn_p = min(1.0,dyn_p)
                if dyn_p == 1.0: # if dyn_p reaches 1, allow lower degree initial candiates
                    if dyn_corelevel == 2:
                        cores = [node for node,deg in g.degree() if deg > k]
                        dyn_corelevel = 1
                        dyn_p = 1/k
                        dyn_avg = min_cnt
                    elif dyn_corelevel == 1:
                        cores = list(nx.nodes(g))
                        dyn_corelevel = 0
                        dyn_p = 1/k
                        dyn_avg = min_cnt
            if dyn_avg < min_cnt - 1: # if current samples are not dense enough, then reduce dyn_p
                dyn_p *= 0.95 
            dyn_sum = 0
    for _ in range(5*count//100): 
        
        # choose a "density-parameter" q
        q = random.randint(1,k)
        
        # choose an initial vertex and initialize the neighborhood
        cand = random.randint(0,n-1)
        cands = {cand}
        neighbors = set(nx.neighbors(g,cand))
        
        # add vertices until the subgraph has sufficient vertices
        while len(cands) < k:
            
            # if there are neighbors, take neighbors, else random
            if len(neighbors) > 0:
            
                # find a good vertex to add
                neighbor = 0
                opt_val = -1 # = number of edges to current candidates
                for _ in range(q):
                    cand = random.choice(list(neighbors))
                    val = len(cands & set(nx.neighbors(g,cand)))
                    if val > opt_val:
                        neighbor = cand
                        opt_val = val
                
                # add the candidate and update neighborhood
                cands.add(neighbor)
                neighbors = (neighbors | set(nx.neighbors(g,neighbor))) - cands
            else:
            
                # choose and add a random vertex
                rnd = random.randint(0,n-1)
                if not rnd in cands:
                    cands.add(rnd)
                    neighbors = (neighbors | set(nx.neighbors(g,rnd))) - cands
        
        cands = list(cands)
        cands.sort()
        
        # count the number of edges
        cnt = 0
        for i in range(0,k):
            for j in range(i+1,k):
                if g.has_edge(cands[i],cands[j]):
                    cnt += 1
        
        # append the subgraph to the positive or negative samples
        if cnt >= min_cnt:
            res_pos.append((cnt,cands))
        else:
            res_neg.append((cnt,cands))
    
    for _ in range(5*count//100):
        cands = random_sample(g,n,k)
        # count the number of edges
        cnt = 0
        for i in range(0,k):
            for j in range(i+1,k):
                if g.has_edge(cands[i],cands[j]):
                    cnt += 1
    
        # append the subgraph to the positive or negative samples
        if cnt >= min_cnt:
            res_pos.append((cnt,cands))
        else:
            res_neg.append((cnt,cands))

    for _ in range(5*count//100):
        cands = connected_sample(g,n,k)
        # count the number of edges
        cnt = 0
        for i in range(0,k):
            for j in range(i+1,k):
                if g.has_edge(cands[i],cands[j]):
                    cnt += 1
    
        # append the subgraph to the positive or negative samples
        if cnt >= min_cnt:
            res_pos.append((cnt,cands))
        else:
            res_neg.append((cnt,cands))
    return res_pos,res_neg

def process_motifs_chunk(g, node, k, max_motifs_per_node):
    neighbors = list(nx.neighbors(g, node))
    if max_motifs_per_node is not None and math.comb(len(neighbors), k) > 2 * max_motifs_per_node:
        temp_motifs = set()
        while len(temp_motifs) < max_motifs_per_node:
            sample = random.sample(neighbors, k)
            sample = [node] + sorted(sample)
            temp_motifs.add(tuple(sample))
        return temp_motifs
    else:
        temp_motifs = []
        for cands in combinations(neighbors, k):
            motif = [node] + list(cands)
            motif = [node] + sorted(motif[1:])
            temp_motifs.append(tuple(motif))
        if max_motifs_per_node is not None and len(temp_motifs) > max_motifs_per_node:
            random.shuffle(temp_motifs)
            temp_motifs = temp_motifs[:max_motifs_per_node]
        return set(temp_motifs)

def process_all_motifs(g, k, max_motifs_per_node):
    motifs = set()
    for node in nx.nodes(g):
        motifs.update(process_motifs_chunk(g, node, k, max_motifs_per_node))
    return motifs