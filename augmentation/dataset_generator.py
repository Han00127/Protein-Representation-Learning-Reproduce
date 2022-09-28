import os 
import os.path as osp
import math
import random
import time
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
# import torch_geometric.transforms as T
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

from utils import utils
from augmentation.transformation import RandomEdgeMask,subspace, subsequence,MaskNode
import re
class Contrastive_ProteinDataset(Dataset):
    def __init__(self,data_list, data_dir):
        self.data_list = data_list  # 단백질 이름 리스트 
        self.data_dir = data_dir    # 단백질 파일 위치 
        self.selected_transform = RandomEdgeMask(p=0.5)

    def __getitem__(self,index): # 상속 클래스 - 단백질 graph 생성 후 return
        protein_path = os.path.join(self.data_dir,self.data_list[index]) # absolute path
        # processed_file = os.path.join(self.graph_dir,self.data_list[index].split('.pdb')[0]+'_GearNet.pt') # absolute path
        
        ca_list = []
        with open('{}'.format(protein_path)) as pdbfile:
            start = True
            for line in pdbfile:
                if line[:4] == 'ATOM' and line[13:15] == 'CA':
                    ca_list.append(line) # 
                    coordinate = [0] * 3
                    coordinate[0], coordinate[1], coordinate[2] = float(line[31:38]),float(line[40:46]),float(line[47:54])
                    if start:
                        graph_3d = np.array([coordinate])
                        start = False
                    else:
                        coordinate = np.array([coordinate])
                        graph_3d = np.concatenate((graph_3d,coordinate), axis=0)

        '''
        pre_transformation
        0. subsequence, 1 subspace
        '''
        cropping = np.random.randint(2) # randomly select cropping function 
        if cropping == 0:
            mask_idx = subsequence(len(ca_list),crop_length=50)
            mask_idx = np.array(mask_idx)
        else: 
            mask_idx = subspace(len(ca_list), graph_3d)
            mask_idx = np.array(mask_idx)
            

        # get node features
        node_feature = utils.generate_node_feature(ca_list)
        # print(node_feature[idx_mask])
        
        node_feature = node_feature[mask_idx]
        graph_3d = graph_3d[mask_idx]

        num_residues = node_feature.size()[0]
        # print("num residues ", num_residues)

        # get sequential edges

        seq_edge,relation_type = utils.generate_seq_edges(num_residues)

        seq_edge = torch.from_numpy(np.array(seq_edge).transpose()).long()
        relation_type = torch.from_numpy(np.array(relation_type)).long()

        # get radius edges
        radius_edge, radius_type = utils.generate_radius_edges(graph_3d)

        radius_type = torch.from_numpy(np.array(radius_type)).long()
        radius_edge = torch.from_numpy(np.array(radius_edge).transpose()).long()

        # get knn edges 
        new_knn_edges, knn_type = utils.generate_knn_edges(graph_3d)
        new_knn_edges = torch.from_numpy(new_knn_edges).long()
        knn_type = torch.from_numpy(knn_type).long()

        # generate edge_index_dict 
        total_edge_index_dict = {}
        total_edge_index_dict[0] = seq_edge
        total_edge_index_dict[1] = radius_edge
        total_edge_index_dict[2] = new_knn_edges

        total_relation_dict = {}
        total_relation_dict[0] = relation_type
        total_relation_dict[1] = radius_type
        total_relation_dict[2] = knn_type


        # Generate edge features [node x 51 dim]


        edge_features_dict = {}
        for relation in range(0,3):    
            start = True
            sub_edge_feature = []
            for i,j in zip(total_edge_index_dict[relation][0], total_edge_index_dict[relation][1]):
                temp = np.array([])
                ## node feature 
                temp = np.hstack((temp, node_feature[i]))
                temp = np.hstack((temp, node_feature[j]))

                ## relation type 
                one_hot_relation_type = [0]* 7
                one_hot_relation_type[relation] = 1
                temp = np.hstack((temp, one_hot_relation_type))
                
                ## sequential distance 
                seq = [abs(i-j)]
                temp = np.hstack((temp, seq))
                
                ## Eculidean Distance 
                dis = [utils.calculate_distance(graph_3d[i],graph_3d[j])]
                temp = np.hstack((temp, dis))
                temp = np.array([temp])
                
                if start:
                    sub_edge_feature = temp
                    start = False
                else:
                    sub_edge_feature = np.concatenate((sub_edge_feature,temp), axis=0)
            start = True
            edge_features_dict[relation] = torch.from_numpy(sub_edge_feature).float()
        
        # Integrating all graph information
        edge_index = []
        edge_feature = []
        relation = []
        start = True
        for i in range(3):
            if start:
                edge_index = total_edge_index_dict[i] # 2 x 447
                edge_feature = edge_features_dict[i] # 447 x 51
                relation = total_relation_dict[i] # 449
                start = False
            else:
                edge_index_tmp = total_edge_index_dict[i] # 2 x 447
                edge_feature_tmp = edge_features_dict[i] # 447 x 51
                relation_tmp = total_relation_dict[i] # 449
                edge_index = torch.cat((edge_index,edge_index_tmp), axis=1)
                edge_feature = torch.cat((edge_feature, edge_feature_tmp))
                relation = torch.cat((relation, relation_tmp))


        data = Data(x = node_feature, edge_index = edge_index, edge_type=relation, edge_attr= edge_feature)
        # torch.save(data, processed_file)

        '''
        Post transformation
        '''       
        data = self.selected_transform(data)
        return data

    def shuffle(self):
        prev = self.data_list.copy()
        random.shuffle(prev)
        self.data_list = prev

    def __len__(self):
        return len(self.data_list)



class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_type_s=None, num_nodes_s = None,edge_index_t=None, x_t=None,edge_type_t=None,num_nodes_t = None):
        super().__init__()
        # For protein grpah representation
        self.x_s = x_s
        self.edge_index_s = edge_index_s
        self.edge_type_s = edge_type_s
        self.num_nodes = num_nodes_s

        # For Protein graph edge to node graph.
        self.x_t = x_t
        self.edge_index_t = edge_index_t
        self.edge_type_t = edge_type_t
        self.num_nodes_t = num_nodes_t
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def cat_dim(self, key, value):
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # `*index*` and `*face*` should be concatenated in the last dimension,
        # everything else in the first dimension.
        return -1 if bool(re.search('(index|face)', key)) else 0


class Contrastive_ProteinDataset_GearNet_Edge(Dataset):
    def __init__(self,data_list, data_dir):
        self.data_list = data_list  # 단백질 이름 리스트 
        self.data_dir = data_dir    # 단백질 파일 위치 
        self.selected_transform = RandomEdgeMask(p=0.5)
        
    def __getitem__(self,index): # 상속 클래스 - 단백질 graph 생성 후 return
        protein_path = os.path.join(self.data_dir,self.data_list[index]) # absolute path
        # if os.path.exists(processed_file):
        #    data = torch.load(processed_file)
        #    return data
        # start_time = time.time()
        ca_list = []
        # print(processed_path)₩
        with open('{}'.format(protein_path)) as pdbfile:
            start = True
            for line in pdbfile:
                if line[:4] == 'ATOM' and line[13:15] == 'CA':
                    ca_list.append(line) # 
                    coordinate = [0] * 3
                    coordinate[0], coordinate[1], coordinate[2] = float(line[31:38]),float(line[40:46]),float(line[47:54])
                    if start:
                        graph_3d = np.array([coordinate])
                        start = False
                    else:
                        coordinate = np.array([coordinate])
                        graph_3d = np.concatenate((graph_3d,coordinate), axis=0)
        '''
        pre_transform for cropping mask idx
        0. subsequence 1. subspace 
        '''
        cropping = np.random.randint(2) # randomly select cropping function 
        if cropping == 0:
            mask_idx = subsequence(len(ca_list),crop_length=50)
            mask_idx = np.array(mask_idx)
        else: 
            mask_idx = subspace(len(ca_list), graph_3d)
            mask_idx = np.array(mask_idx)


        ################### get node features and cropping ## ###################
        node_feature = utils.generate_node_feature(ca_list)
        node_feature = node_feature[mask_idx] # masking node_features 
        graph_3d = graph_3d[mask_idx] # masking 3d coordinates

        num_residues = node_feature.size()[0]
        #########################################################################
        
        ################### get edege index and its relations ## ###################
        # get sequential edges
        seq_edge,relation_type = utils.generate_seq_edges(num_residues)

        seq_edge = torch.from_numpy(np.array(seq_edge).transpose()).long()
        relation_type = torch.from_numpy(np.array(relation_type)).long()

        # get radius edges
        radius_edge, radius_type = utils.generate_radius_edges(graph_3d)

        radius_type = torch.from_numpy(np.array(radius_type)).long()
        radius_edge = torch.from_numpy(np.array(radius_edge).transpose()).long()

        # get knn edges 
        new_knn_edges, knn_type = utils.generate_knn_edges(graph_3d)
        new_knn_edges = torch.from_numpy(new_knn_edges).long()
        knn_type = torch.from_numpy(knn_type).long()

        # generate edge_index_dict 
        total_edge_index_dict = {}
        total_edge_index_dict[0] = seq_edge
        total_edge_index_dict[1] = radius_edge
        total_edge_index_dict[2] = new_knn_edges

        total_relation_dict = {}
        total_relation_dict[0] = relation_type
        total_relation_dict[1] = radius_type
        total_relation_dict[2] = knn_type
        ##################################################################

        ################### # Generate edge features #####################
        edge_features_dict = {}
        for relation in range(0,3):    
            start = True
            sub_edge_feature = []
            for i,j in zip(total_edge_index_dict[relation][0], total_edge_index_dict[relation][1]):
                temp = np.array([])
                temp = np.hstack((temp, node_feature[i]))
                # print(node_feature[j])
                temp = np.hstack((temp, node_feature[j]))
                one_hot_relation_type = [0]* 7
                one_hot_relation_type[relation] = 1
                # one_hot_relation_type = np.array(one_hot_relation_type)
                temp = np.hstack((temp, one_hot_relation_type))
                # print(one_hot_relation_type)
                seq = [abs(i-j)]
                temp = np.hstack((temp, seq))
                # print(seq)
                dis = [utils.calculate_distance(graph_3d[i],graph_3d[j])]
                temp = np.hstack((temp, dis))
                temp = np.array([temp])
                # print(temp.shape)
                if start:
                    sub_edge_feature = temp
                    start = False
                else:
                    sub_edge_feature = np.concatenate((sub_edge_feature,temp), axis=0)
            start = True
            edge_features_dict[relation] = torch.from_numpy(sub_edge_feature).float()
        ##################################################################

        ############## Integrate all graph information ############
        edge_index = []
        edge_feature = []
        relation = []
        start = True
        for i in range(3):
            if start:
                edge_index = total_edge_index_dict[i] # 2 x 447
                edge_feature = edge_features_dict[i] # 447 x 51
                relation = total_relation_dict[i] # 449
                start = False
            else:
                edge_index_tmp = total_edge_index_dict[i] # 2 x 447
                edge_feature_tmp = edge_features_dict[i] # 447 x 51
                relation_tmp = total_relation_dict[i] # 449
                edge_index = torch.cat((edge_index,edge_index_tmp), axis=1)
                edge_feature = torch.cat((edge_feature, edge_feature_tmp))
                relation = torch.cat((relation, relation_tmp))
        ##################################################################

        ######################## Transformation ##########################################
        data = Data(x = node_feature, edge_index = edge_index, edge_type=relation, edge_attr= edge_feature, pos=graph_3d)
        data = self.selected_transform(data)
        ###################################################################################

        ########################Generate edge message graph ##########################################
        edge_message_index,edge_message_relation = utils.generate_edge_graph(data.edge_index.numpy(), data.edge_attr.numpy())
        edge_message_index = torch.from_numpy(np.array(edge_message_index).transpose()).long()
        edge_message_relation = torch.from_numpy(np.array(edge_message_relation)).long()
        ##################################################################
        
        data.edge_message_index = edge_message_index
        data.edge_message_relation = edge_message_relation
        # torch.save(data,osp.join(self.processed , self.data_list[index][0]+'.pt'))

        # data = PairData(data.edge_index, data.x, data.edge_type, data.x.size()[0],edge_message_index, data.edge_attr, edge_message_relation,data.edge_attr.size()[0])
        # data = Data(x = node_feature, edge_index = edge_index, edge_type=relation, edge_attr= edge_feature,num_nodes=num_residues)
        # torch.save(data, processed_file)
        # print("one graph data generated in {}s seconds---".format(time.time()-start_time))
        return data


    def shuffle(self):
        prev = self.data_list.copy()
        random.shuffle(prev)
        self.data_list = prev

    def __len__(self):
        return len(self.data_list)


class Self_ProteinDataset_GearNet_Edge(Dataset):
    '''
        Further developed to improve speed 
        save original file, if file exists, load and transformation.
    '''
    def __init__(self,data_list, data_dir,transform=None):
        self.data_list = data_list  # 단백질 이름 리스트 
        self.data_dir = data_dir    # 단백질 파일 위치 
        self.transform = transform  # Transformation
        self.processed = data_dir + 'whole/'
        if not osp.exists(self.processed):
            os.mkdir(self.processed)
    def __getitem__(self,index): # 상속 클래스 - 단백질 graph 생성 후 return
        
        if osp.exists(osp.join(self.processed , self.data_list[index]+'.pt')):
            data = torch.load(osp.join(self.processed , self.data_list[index]+'.pt'))
            data = self.transform(data)
            masked_node_idx = data.masked_node_idx
            masked_node_label = data.mask_node_label

            edge_message_index,edge_message_relation = utils.generate_edge_graph(data.edge_index.numpy(), data.edge_attr.numpy())
            edge_message_index = torch.from_numpy(np.array(edge_message_index).transpose()).long()
            edge_message_relation = torch.from_numpy(np.array(edge_message_relation)).long()

            data.edge_message_index = edge_message_index
            data.edge_message_relation = edge_message_relation
        
            data.masked_node_idx = masked_node_idx
            data.masked_node_label = masked_node_label
            return data

        protein_path = os.path.join(self.data_dir,self.data_list[index]) # absolute path
        ca_list = []
        with open('{}'.format(protein_path)) as pdbfile:
            start = True
            for line in pdbfile:
                if line[:4] == 'ATOM' and line[13:15] == 'CA':
                    ca_list.append(line) # 
                    coordinate = [0] * 3
                    coordinate[0], coordinate[1], coordinate[2] = float(line[31:38]),float(line[40:46]),float(line[47:54])
                    if start:
                        graph_3d = np.array([coordinate])
                        start = False
                    else:
                        coordinate = np.array([coordinate])
                        graph_3d = np.concatenate((graph_3d,coordinate), axis=0)

        # get node features
        node_feature = utils.generate_node_feature(ca_list)
        num_residues = node_feature.size()[0]

        ### relational edge generation ################################
        ## get sequential edges
        seq_edge,relation_type = utils.generate_seq_edges(num_residues)

        seq_edge = torch.from_numpy(np.array(seq_edge).transpose()).long()
        relation_type = torch.from_numpy(np.array(relation_type)).long()

        ## get radius edges
        radius_edge, radius_type = utils.generate_radius_edges(graph_3d)

        radius_type = torch.from_numpy(np.array(radius_type)).long()
        radius_edge = torch.from_numpy(np.array(radius_edge).transpose()).long()

        ## get knn edges 
        new_knn_edges, knn_type = utils.generate_knn_edges(graph_3d)
        new_knn_edges = torch.from_numpy(new_knn_edges).long()
        knn_type = torch.from_numpy(knn_type).long()

        # generate edge_index_dict 
        total_edge_index_dict = {}
        total_edge_index_dict[0] = seq_edge
        total_edge_index_dict[1] = radius_edge
        total_edge_index_dict[2] = new_knn_edges

        total_relation_dict = {}
        total_relation_dict[0] = relation_type
        total_relation_dict[1] = radius_type
        total_relation_dict[2] = knn_type


        # Generate edge features [node x 51 dim]
        edge_features_dict = {}
        for relation in range(0,3):    
            start = True
            sub_edge_feature = []
            for i,j in zip(total_edge_index_dict[relation][0], total_edge_index_dict[relation][1]):
                temp = np.array([])
                temp = np.hstack((temp, node_feature[i]))
                # print(node_feature[j])
                temp = np.hstack((temp, node_feature[j]))
                one_hot_relation_type = [0]* 7
                one_hot_relation_type[relation] = 1
                # one_hot_relation_type = np.array(one_hot_relation_type)
                temp = np.hstack((temp, one_hot_relation_type))
                # print(one_hot_relation_type)
                seq = [abs(i-j)]
                temp = np.hstack((temp, seq))
                # print(seq)
                dis = [utils.calculate_distance(graph_3d[i],graph_3d[j])]
                temp = np.hstack((temp, dis))
                temp = np.array([temp])
                # print(temp.shape)
                if start:
                    sub_edge_feature = temp
                    start = False
                else:
                    sub_edge_feature = np.concatenate((sub_edge_feature,temp), axis=0)
            start = True
            edge_features_dict[relation] = torch.from_numpy(sub_edge_feature).float()
        
        ## Integrating all graph information
        edge_index = []
        edge_feature = []
        relation = []
        start = True
        for i in range(3):
            if start:
                edge_index = total_edge_index_dict[i] # 2 x 447
                edge_feature = edge_features_dict[i] # 447 x 51
                relation = total_relation_dict[i] # 449
                start = False
            else:
                edge_index_tmp = total_edge_index_dict[i] # 2 x 447
                edge_feature_tmp = edge_features_dict[i] # 447 x 51
                relation_tmp = total_relation_dict[i] # 449
                edge_index = torch.cat((edge_index,edge_index_tmp), axis=1)
                edge_feature = torch.cat((edge_feature, edge_feature_tmp))
                relation = torch.cat((relation, relation_tmp))

        data = Data(x = node_feature, edge_index = edge_index, edge_type=relation, edge_attr= edge_feature, pos=graph_3d)
        torch.save(data,osp.join(self.processed , self.data_list[index]+'.pt'))

        '''
        selected transformation MaskNode
        data.masked_node_idx
        data.mask_node_label
        '''
        data = self.transform(data)
        masked_node_idx = data.masked_node_idx
        masked_node_label = data.mask_node_label

        edge_message_index,edge_message_relation = utils.generate_edge_graph(data.edge_index.numpy(), data.edge_attr.numpy())
        edge_message_index = torch.from_numpy(np.array(edge_message_index).transpose()).long()
        edge_message_relation = torch.from_numpy(np.array(edge_message_relation)).long()

        data.edge_message_index = edge_message_index
        data.edge_message_relation = edge_message_relation
    
        data.masked_node_idx = masked_node_idx
        data.masked_node_label = masked_node_label
 
        return data


    def shuffle(self):
        prev = self.data_list.copy()
        random.shuffle(prev)
        self.data_list = prev

    def __len__(self):
        return len(self.data_list)