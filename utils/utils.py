import os 
import os.path as osp
import math
import random

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
# from torch.utils.data import Dataset
import torch_geometric.transforms as T
import torch.nn as nn

import numpy as np
import heapq
from heapq import heappush, heappop
from tqdm import tqdm

import numba
from numba.typed import List
from numba import njit, jit
from numpy import dot
from numpy.linalg import norm

from augmentation.transformation import RandomEdgeMask,subspace, subsequence
amino_acids = [ 'ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN',
                'CYS', 'SEC', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET',
                'PHE', 'TYR', 'TRP']
amino_acids_dict = {'ARG':0, 'HIS':1, 'LYS':2, 'ASP':3, 'GLU':4, 'SER':5, 'THR':6, 'ASN':7, 'GLN':8,
                'CYS':9, 'SEC':10, 'GLY':11, 'PRO':12, 'ALA':13, 'VAL':14, 'ILE':15, 'LEU':16, 'MET':17,
                'PHE':18, 'TYR':19, 'TRP':20, 'UNK':21} # 21 represents masked (unknown)

relation_type = {0:'seq_m2', 1:'seq_m1',2:'seq_0',3:'seq_p1',4:'seq_p2', 5:'radius', 6:'knn'}

cos = nn.CosineSimilarity(dim=0, eps=1e-6)
'''
:4 = ATOM
13:15 = atoms 
17:20 = residue name 
32:38 = x coordinate
41:46 = y coordinate
47:54 = z coordinate 
'''

def find_bin(value, bins):
    """ bins is a list of tuples, like [(0,20), (20, 40), (40, 60)],
        binning returns the smallest index i of bins so that
        bin[i][0] <= value < bin[i][1]
    """
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1

@njit
def calculate_distance(x,y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)


def task_data_load(data_list):
    # train_bs = args.batch_size
    data = [i.strip('\n').split('\t') for i in open(data_list)]
    return data


def data_load(data_list):
    data = open(data_list).read().splitlines()
    return data

def one_hot_encoding(ele,size):
  one_hot_vector = [0]*size
  index = amino_acids_dict[ele[17:20]]
  one_hot_vector[index] = 1
  return np.array([one_hot_vector])


def generate_node_feature(ca_list):
    start = True
    for ele in ca_list:
        test = one_hot_encoding(ele,22)
        if start:
            node_feature = test
            start = False
        else:
            node_feature = np.concatenate((node_feature,test), axis=0)
    return torch.from_numpy(node_feature).float()

@njit(cache=True)
def generate_seq_edges(num_residue):
    seq_edge_dict = List()
    relation_type = List()
    for i_node in range(num_residue):
        for j_node in range(num_residue):
            seq_dis = j_node - i_node
            if seq_dis == -2: # seq -2 type
                seq_edge_dict.append([i_node,j_node])
                relation_type.append(0)

            elif seq_dis == -1: # seq -1 type
                seq_edge_dict.append([i_node,j_node])
                relation_type.append(1)

            elif seq_dis == 0: # seq 0 type unnecessary diagonal itself.
                seq_edge_dict.append([i_node,j_node])
                relation_type.append(2)

            elif seq_dis == 1: # seq 1 type
                seq_edge_dict.append([i_node,j_node])
                relation_type.append(3)

            elif seq_dis == 2: # seq 2 type
                seq_edge_dict.append([i_node,j_node])
                relation_type.append(4)
            else:
                pass
    return seq_edge_dict, relation_type

@njit
def generate_radius_edges(graph_3d):
    radius_edge = List()
    relation_type = List()
    for i_node,i_node_coord in enumerate(graph_3d):
        for j_node,j_node_coord in enumerate(graph_3d):
            dis_i_j = calculate_distance(i_node_coord,j_node_coord)
            if dis_i_j < 10 and abs(i_node - j_node) < 5:
                radius_edge.append([i_node,j_node])
                relation_type.append(5)

    return radius_edge, relation_type

def generate_knn_edges(graph_3d):
    knn_edge = []
    relation_type = []
    for i_node, i_node_coord in enumerate(graph_3d):
        heap = []    
        for j_node, j_node_coord in enumerate(graph_3d):
            # print(i_node, j_node)
            dis_i_j = calculate_distance(i_node_coord,j_node_coord)
            heappush(heap, (dis_i_j, j_node))

        # for _ in range(10):
        while len(heap) > 0:
            node = heappop(heap)
            knn_edge.append([i_node,node[1]])
        # print(knn_edge)

    knn_edges = np.array(knn_edge).transpose()
    
    new_knn_edges = []
    # print(knn_edges)
    for idx,_ in enumerate(knn_edges[0]):
        if abs(knn_edges[0][idx] - knn_edges[1][idx]) < 5:
            new_knn_edges.append([knn_edges[0][idx],knn_edges[1][idx]])
            relation_type.append(6)
          
    new_knn_edges = np.array(new_knn_edges).transpose()
    relation_type = np.array(relation_type)
    return new_knn_edges, relation_type



'''
 Boosting speed by numba 
'''
@njit
def generate_edge_graph(A,feature_A):
    bins = [(0,22.5),(22.5,45.0),(45.0,67.5), (67.5, 90.0), (90.0,112.5), (135,157.5), (157.5, 180.0)]
    edge_message_relation= List()
    edge_edge = List()

    for idx_source, (i, j) in enumerate(zip(A[0],A[1])):
        for idx_target, (w,k) in enumerate(zip(A[0],A[1])):
            if idx_source == idx_target:
                pass
            # elif j == w and i != k and i != j:
            elif j == k and i != w and i != j:
                # make edge
                edge_edge.append([w,k])
                
                a = feature_A[idx_source]
                b = feature_A[idx_target]


                cos_sim = dot(a, b)/(norm(a)*norm(b))
                
                value = math.degrees(math.acos(cos_sim))

                for i in range(0, len(bins)):
                    if bins[i][0] <= value < bins[i][1]:
                        degree_bin = i
                
                edge_message_relation.append(degree_bin)
    
    return edge_edge, edge_message_relation

@njit
def get_indexes(edges, edge_indicator):
    mask_idx = List()
    # print(edge_indicator)
    for target in edges:
        i,j = target[0], target[1]
        for idx, element in enumerate(edge_indicator):
            m,n = element[0], element[1]
    #     for m,n in zip(edge_indicator[0], edge_indicator[1]):
            if i == m and j == n:
                mask_idx.append(idx)
    #         count+=1
    return mask_idx
