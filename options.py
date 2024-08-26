import argparse


# This function can be used as a type in options
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def args_parser():

    parser = argparse.ArgumentParser(description='Motif prediction')

    # general settings
    parser.add_argument('--data-name', default=None, help='choose input graph')
    parser.add_argument('--prediction-method', type=str, default='lhsgnn',
                        help='methods: naive, seal, seam, lhsgnn')
    parser.add_argument('--prediction-threshold', type=float, default=0.5,
                        help='Threshold for motif prediction (default: 0.5)')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--max-train-num', type=int, default=100000,
                        help='set maximum number of train links (to fit into memory)')
    parser.add_argument('--max-motifs-node', type=int, default=0,
                        help='set maximum number of train links (to fit into memory)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--test-ratio',type = float, default=0.1,
                        help='Set test ratio, options: any fraction from 0 to 1')
    parser.add_argument('--no-parallel', action='store_true', default=False,
                        help='if True, use single thread for subgraph extraction; \
                        by default use all cpu cores to extract subgraphs in parallel')
    parser.add_argument('--scratch-mode', action='store_true', default=False,
                        help='Option for CSCS cluster to use the users scratch directory \
                        to store datasets')
    parser.add_argument('--results-file-prefix',type = str, default='',
                        help='Set a prefix for the acc_results and auc_results files')

    # model settings
    parser.add_argument('--learning-rate',type = float, default=0.002,
                        help='Set learning rate for the GNN')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--hop', default=1, metavar='S',
                        help='enclosing subgraph hop number, \
                        options: 1, 2,..., "auto"')
    parser.add_argument('--motif-edge-mode', type=int, default=1,
                        help='options: 0 = remove all edges, 1 = match negative samples, 2 = dont remove any edges')
    
    parser.add_argument('--distance-label-mode', type=int, default=1,
                        help='options: 0 = no distance labels, 1 = default distance labels, 2 = experimental distance labels, 3 = both labels')
    parser.add_argument('--max-nodes-per-hop', default=None,
                        help='if > 0, upper bound the # nodes per hop by subsampling')
    parser.add_argument('--use-embedding', action='store_true', default=True,
                        help='whether to use node2vec node embeddings')
    parser.add_argument('--use-attribute', action='store_true', default=False,
                        help='whether to use node attributes')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='save the final model')
    parser.add_argument('--save-datasets', action='store_true', default=False,
                        help='save subgraph datasets')
    parser.add_argument('--only-save-datasets', action='store_true', default=False,
                        help='only save subgraph datasets without training')
    parser.add_argument('--load-datasets', action='store_true', default=False,
                        help='load subgraph datasets')

    parser.add_argument('--motif', type=str, default='star',
                        help='select motif: star = k-star, dbstar = dealbreaker k-star,clique = k-clique, dense = dense subgraph, circle = 4-circle, path = 4-path, tailed_triangle, chordal_cycle')
    parser.add_argument('--motif-k', type=int, default=3,
                        help='select k for k-star and k-clique motif')
    parser.add_argument('--motif-f', type=float, default=0.9,
                        help='select f for f-dense subgraph with k vertices')

    parser.add_argument('--max-cores', type=int, default=None,
                        help='select the maximum amount of threads for parallel execution')

    parser.add_argument('--aggregation', type=str, default='mul',
                        help='Aggregation method in naive prediction [mul (default), avg]')
    parser.add_argument('--naive-method', type=str, default='jaccard',
                        help='Method for naive link prediction [jaccard (default), cn]')
    parser.add_argument('--naive-edge-mode', type=int, default=0,
                        help='options: 0 = keep edges, 1 = remove the masked edges')


    args = parser.parse_args()
    return args
