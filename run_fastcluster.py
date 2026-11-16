#!/usr/bin/env python

"""
Omnibenchmark-izes Marek Gagolewski's https://github.com/gagolews/clustering-results-v1/blob/eae7cc00e1f62f93bd1c3dc2ce112fda61e57b58/.devel/do_benchmark_fastcluster.py

Takes the true number of clusters into account and outputs a 2D matrix with as many columns as ks tested,
being true number of clusters `k` and tested range `k plusminus 2`
"""

import argparse
import os, sys
import fastcluster
import numpy as np
import scipy.cluster.hierarchy

VALID_LINKAGE = ['complete', 'average', 'weighted', 'median', 'ward', 'centroid']

def load_labels(data_file):
    data = np.loadtxt(data_file, ndmin=1)
    
    if data.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")
    
    return(data)

def load_dataset(data_file):
    data = np.loadtxt(data_file, ndmin=2)
    
    ##data.reset_index(drop=True,inplace=True)
    
    if data.ndim != 2:
        raise ValueError("Invalid data structure, not a 2D matrix?")
    
    return(data)

## this maps the ks to their true offset to the truth, e.g.:
# >>> generate_k_range(5)
# {'k-2': 3, 'k-1': 4, 'k': 5, 'k+1': 6, 'k+2': 7}
# >>> generate_k_range(1)
# {'k-2': 2, 'k-1': 2, 'k': 2, 'k+1': 2, 'k+2': 3}
# >>> generate_k_range(2)
# {'k-2': 2, 'k-1': 2, 'k': 2, 'k+1': 3, 'k+2': 4}
## k is the true k
def generate_k_range(k):
    Ks = [k-2, k-1, k, k+1, k+2] # ks tested, including the true number
    replace = lambda x: x if x >= 2 else 2 ## but we never run k < 2; those are replaced by a k=2 run (to not skip the calculation)
    Ks = list(map(replace, Ks))
    
    # ids = ['k-2', 'k-1', 'k', 'k+1', 'k+2']
    ids = list(range(0,5))
    assert(len(ids) == len(Ks))
    
    k_ids_dict = dict.fromkeys(ids, 0)
    for i in range(len(ids)):
        key = ids[i]
        
        k_ids_dict[key] = Ks[i]
    return(k_ids_dict)

# ## modified from
# ## author Marek Gagolewski
# ## https://github.com/gagolews/clustering-results-v1/blob/eae7cc00e1f62f93bd1c3dc2ce112fda61e57b58/.devel/do_benchmark_fastcluster.py#L33C1-L58C15
# def do_benchmark_fastcluster_single_k(X, ks, linkage):
#     res = dict()
    
#     if linkage in ["median", "ward", "centroid"]:
#         linkage_matrix = fastcluster.linkage_vector(X, method=linkage)
#     else: # these compute the whole distance matrix
#         linkage_matrix = fastcluster.linkage(X, method=linkage)

#     # correction for the departures from ultrametricity -- cut_tree needs this.
#     linkage_matrix[:,2] = np.maximum.accumulate(linkage_matrix[:,2])
#     labels_pred_matrix = scipy.cluster.hierarchy.\
#         cut_tree(linkage_matrix, n_clusters=k)+1 # 0-based -> 1-based!!!
    
#     res = labels_pred_matrix#[:k]
    
#     return res

## modified from
## author Marek Gagolewski
## https://github.com/gagolews/clustering-results-v1/blob/eae7cc00e1f62f93bd1c3dc2ce112fda61e57b58/.devel/do_benchmark_fastcluster.py#L33C1-L58C15
def do_benchmark_fastcluster_range_ks(X, Ks, linkage):
    res = dict()
    
    ## for K_id in Ks.keys(): res[K_id] = dict()
    
    if linkage in ["median", "ward", "centroid"]:
        linkage_matrix = fastcluster.linkage_vector(X, method=linkage)
    else: # these compute the whole distance matrix
        linkage_matrix = fastcluster.linkage(X, method=linkage)

    for item in Ks.keys():
        K_id = item  ## just an unique identifier
        K = Ks[K_id] ## the tested k perhaps repeated
        
        ## correction for the departures from ultrametricity -- cut_tree needs this.
        linkage_matrix[:,2] = np.maximum.accumulate(linkage_matrix[:,2])
        pred = scipy.cluster.hierarchy.\
            cut_tree(linkage_matrix, n_clusters=K)+1 # 0-based -> 1-based!!!
        # print(K)
        # print(pred.shape)
        res[K_id] = pred[:,-1]
    # print(res)
    return np.array([res[key] for key in res.keys()]).T


def main():
    parser = argparse.ArgumentParser(description='clustbench fastcluster runner')

    parser.add_argument('--data_matrix', type=str,
                        help='gz-compressed textfile containing the comma-separated data to be clustered.', required = True)
    parser.add_argument('--data_true_labels', type=str,
                        help='gz-compressed textfile with the true labels; used to select a range of ks.', required = True)
    parser.add_argument('--output', action='append', default=[],
                        help='id=PATH, one per output the stage declares (api 0.8.0).')
    parser.add_argument('--linkage', type=str,
                        help='fastcluster linkage',
                        required = True)

    # ob still passes --output_dir and --name until api 0.10; they are ignored.
    args, _ = parser.parse_known_args()
    outputs = dict(o.split('=', 1) for o in args.output)

    if args.linkage not in VALID_LINKAGE:
        raise ValueError(f"Invalid linkage `{args.linkage}`")

    truth = load_labels(args.data_true_labels)
    k = int(max(truth)) # true number of clusters
    Ks = generate_k_range(k)
    # print(Ks)
    
    data = args.data_matrix
    curr = do_benchmark_fastcluster_range_ks(X= load_dataset(data), Ks = Ks, linkage = args.linkage)

    # print(curr.shape)
    

    header=['k=%s'%s for s in Ks.values()]
    
    curr = np.append(np.array(header).reshape(1,5), curr.astype(str), axis=0)
    np.savetxt(outputs['clustering_predicted_ks_range'],
               curr, fmt='%s', delimiter=",")#,
               # header = ','.join(header)) 

if __name__ == "__main__":
    main()
