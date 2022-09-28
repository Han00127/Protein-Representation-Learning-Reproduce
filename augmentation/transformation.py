import random
import torch
from itertools import islice, cycle
import numpy as np
from torch_geometric.transforms import BaseTransform
from numba import njit, jit
from numba.typed import List
import math

@njit
def calculate_distance(x,y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)

class RandomEdgeMask(BaseTransform):
    """Randomly masked edges in the data.

    Args:
        axis (int): The axis along the position of nodes being flipped.
        p (float, optional): Probability that node positions will be flipped.
            (default: :obj:`0.5`)
    """
    def __init__(self, mask_rate=0.15, p=0.5):
        self.mask_rate = mask_rate
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            _,num_edge  = data.edge_index.size()
            num_mask = int(0.15 * num_edge)
            idx_mask = np.random.choice(num_edge, (num_edge-num_mask), replace=False)

            data.edge_index = data.edge_index[:, idx_mask]
            data.edge_attr = data.edge_attr[idx_mask]
            data.edge_type = data.edge_type[idx_mask]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(axis={self.axis}, p={self.p})'
        
def subsequence(num_nodes, crop_length):
    '''
        Multiview Contrast, we set the cropping length of subsequence operation as 50
    '''
    crop_legnth = 50
    seq = np.arange(num_nodes)

    start = random.randint(0, num_nodes - 1)
    mask_idx = List(islice(cycle(seq), start, start + crop_length))

    return mask_idx

@njit
def subspace(num_nodes,pos):
    '''
    the radius of subspace operation as 15, 3d 좌표 정보 활용. 
    '''
    center_node = np.random.randint(num_nodes)
    center_pos = pos[center_node]    
    mask_idx = List()
    for idx, coord in enumerate(pos):
        dis = calculate_distance(center_pos,coord)
        if dis < 15:
            mask_idx.append(idx)
    return mask_idx


'''
https://github.com/Shen-Lab/GraphCL/blob/fe93387ca251950000014ff96c3868a567c85813/transferLearning_MoleculeNet_PPI/bio/util.py

modified to node attr by Tak
'''
class MaskNode:
    def __init__(self, mask_rate):
        """
        Assume node_attr is of the form:
        w0 represents type of residue
        [w0, w2, w3, w4, w5, w6, w7,...,w20,mask]
        :param mask_rate: % of nodes to be masked
        """
        self.mask_rate = mask_rate

    def __call__(self, data, masked_node_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_edge_indices: If None, then randomly sample num_edges * mask_rate + 1
        number of edge indices. Otherwise should correspond to the 1st
        direction of an edge pair. ie all indices should be an even number
        :return: None, creates new attributes in the original data object:
        data.mask_edge_idx: indices of masked edges
        data.mask_edge_labels: corresponding ground truth edge feature for
        each masked edge
        data.edge_attr: modified in place: the edge features (
        both directions) that correspond to the masked edges have the masked
        edge feature
        """
        if masked_node_indices == None:
            # sample x distinct edges to be masked, based on mask rate. But
            # will sample at least 1 edge
            num_node = int(data.x.size()[0])  # num unique edges
            sample_size = int(num_node * self.mask_rate + 1)
            # during sampling, we only pick the 1st direction of a particular
            # edge pair
            masked_node_indices = [i for i in random.sample(range(num_node), sample_size)]

        data.masked_node_idx = torch.tensor(np.array(masked_node_indices))

        # create ground truth edge features for the edges that correspond to
        # the masked indices
        mask_node_labels_list = []
        for idx in masked_node_indices:
            mask_node_labels_list.append(data.x[idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)

        # created new masked edge_attr, where both directions of the masked
        # edges have masked edge type. For message passing in gcn

        # append the 2nd direction of the masked edges
        # all_masked_node_indices = masked_node_indices + [i + 1 for i in masked_node_indices]
        for idx in masked_node_indices:
            data.x[idx] = torch.tensor(np.array([0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 1]),dtype=torch.float)
        return data
