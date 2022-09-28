import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_scatter import scatter
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import FastRGCNConv, RGCNConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np
from models.v2_message_emp import EdgeMessagePassingLayer,CustomRGCNConv


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class GearNet(torch.nn.Module):
    def __init__(self, num_relations, dropout, pretrain_level = 0):
        super(GearNet, self).__init__()
        self.dropout_ratio = dropout

        
        # self.input_node_embeddings = torch.nn.Embedding(21, 512)
        # torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)


        self.bn = BatchNorm(512)
        self.conv1 = RGCNConv(22, 512, num_relations)
        self.conv2 = RGCNConv(512, 512, num_relations)
        self.conv3 = RGCNConv(512, 512, num_relations)
        self.conv4 = RGCNConv(512, 512, num_relations)
        self.conv5 = RGCNConv(512, 512, num_relations)
        self.conv6 = RGCNConv(512, 512, num_relations)

        # self.conv1 = FastRGCNConv(21, 512, num_relations)
        # self.conv2 = FastRGCNConv(512, 512, num_relations)
        # self.conv3 = FastRGCNConv(512, 512, num_relations)
        # self.conv4 = FastRGCNConv(512, 512, num_relations)
        # self.conv5 = FastRGCNConv(512, 512, num_relations)
        # self.conv6 = FastRGCNConv(512, 512, num_relations)

        self.pool = gap
        # if pretrain_level == 0: # constrastive learning
        self.projection_head = nn.Sequential(nn.Linear(512, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward(self,data):
        # '''
        #   GPU 별 batch 분배 
        # '''
        # print(data)
        batch = data.batch 
        x,edge_index,edge_type = data.x, data.edge_index,data.edge_type
        # Acutual message passing   
        # x =  self.input_node_embeddings(x)
        x = F.relu(self.bn(self.conv1(x, edge_index,edge_type)))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = F.relu(self.bn(self.conv2(x, edge_index,edge_type)))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = F.relu(self.bn(self.conv3(x, edge_index,edge_type)))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = F.relu(self.bn(self.conv4(x, edge_index,edge_type)))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = F.relu(self.bn(self.conv5(x, edge_index,edge_type)))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = self.bn(self.conv6(x, edge_index,edge_type))
        # emb = x 
        x = self.pool(x,batch)
        x = self.projection_head(x)
        return x

    def constastive_loss(self, x1, x2):
        tow = 0.07
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / tow)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

'''
GearNet-Edge model
'''

class GearNet_Edge(torch.nn.Module):
    def __init__(self, num_relations, dropout, pre_train_level, task = False,):
        super(GearNet_Edge, self).__init__()
        self.dropout_ratio = dropout

        self.bn1 = BatchNorm(512)
        self.bn2 = BatchNorm(512)
        self.bn3 = BatchNorm(512)
        self.conv1 = CustomRGCNConv(22, 512, num_relations)
        self.conv2 = CustomRGCNConv(512, 512, num_relations)
        self.conv3 = CustomRGCNConv(512, 512, num_relations)
        self.conv4 = CustomRGCNConv(512, 512, num_relations)
        self.conv5 = CustomRGCNConv(512, 512, num_relations)
        self.conv6 = CustomRGCNConv(512, 512, num_relations)

          # emb_dim, hidden_dim, num_relations,input_layer
        self.emp1 = EdgeMessagePassingLayer(53,512,8,True)
        self.emp2 = EdgeMessagePassingLayer(512,512,8)
        self.emp3 = EdgeMessagePassingLayer(512,512,8)
        self.emp4 = EdgeMessagePassingLayer(512,512,8)
        self.emp5 = EdgeMessagePassingLayer(512,512,8)
        self.emp6 = EdgeMessagePassingLayer(512,512,8)

        self.dim_ = 3072
        self.pool = gap

        self.pre_train_level = pre_train_level
        self.task = task
        if self.pre_train_level == 0: # constrastive learning
            # self.projection_head = nn.Sequential(nn.Linear((self.dim_ * 2), int(self.dim_)), nn.ReLU(inplace=True), nn.Linear(int(self.dim_), int(self.dim_ / 2)), nn.Linear(int(self.dim_ / 2), 300))
            self.projection_head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, 300), nn.Linear(300, 300))
        elif self.pre_train_level == 1: #residue type prediction
            self.classification_head = nn.Sequential(nn.Linear(512,300), nn.ReLU(inplace=True), nn.Linear(300,22))
    

    def forward(self,data):

        for_downstream = []
        batch = data.batch 

        x,edge_index,edge_type = data.x, data.edge_index, data.edge_type
        edge_message_x, edge_message_index, edge_message_type = data.edge_attr, data.edge_message_index, data.edge_message_relation

        edge_message_x = self.emp1(edge_message_x, edge_message_index, edge_message_type)
        x = F.relu(self.bn1(self.conv1(x,edge_index,edge_type,edge_message_x)))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        for_downstream.append(x)

        edge_message_x = self.emp2(edge_message_x, edge_message_index, edge_message_type)
        x = F.relu(self.bn2(self.conv2(x,edge_index,edge_type,edge_message_x)))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        for_downstream.append(x)

        edge_message_x = self.emp3(edge_message_x, edge_message_index, edge_message_type)
        x = F.relu(self.bn3(self.conv3(x,edge_index,edge_type,edge_message_x)))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        for_downstream.append(x)

        edge_message_x = self.emp4(edge_message_x, edge_message_index, edge_message_type)
        x = F.relu(self.bn3(self.conv4(x,edge_index,edge_type,edge_message_x)))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        for_downstream.append(x)

        edge_message_x = self.emp5(edge_message_x, edge_message_index, edge_message_type)
        x = F.relu(self.bn3(self.conv5(x,edge_index,edge_type,edge_message_x)))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        for_downstream.append(x)

        edge_message_x = self.emp6(edge_message_x, edge_message_index, edge_message_type)
        x = self.bn2(self.conv6(x,edge_index,edge_type,edge_message_x))
        for_downstream.append(x)
        '''
            pretrain_level
        '''
        if self.task:
            return x, for_downstream, edge_index,edge_message_x
        else:
            if self.pre_train_level == 0: # constrastive learning
                x = self.pool(x,batch)
                x = self.projection_head(x)
            elif self.pre_train_level == 1: #residue type prediction
                ### predict the edge types.
                # masked_node_index = data.x_s[data.masked_node_idx]
                node_rep = x[data.masked_node_idx]
                x = self.classification_head(node_rep)
            return x


class GearNet_Decoder(torch.nn.Module):
    """
    GNN w/ dropout
    Output:
        node representations
    """ 
    def __init__(self, num_layer, emb_dim, drop_ratio = 0, task_id = 0):
        '''
            task: 0 - Enzyme Commision number prediction 
            task: 1 - Gene Ontology
            task: 2 - Fold Classification
            task  3 - Reaction 
        '''
        super(GearNet_Decoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.task_id = task_id
        self.start = True
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.") 
        self.dim_ = 3072 
        '''
            class specific module
            task: 0 - Enzyme Commision number prediction 
            task: 1 - Gene Ontology
            task: 2 - Fold Classification
            task  3 - Reaction 
        '''
        if self.task_id == 0: # 
            pass 
        elif self.task_id== 1: #
            pass
        elif self.task_id == 2: # 
            self.fc = nn.Linear(int(self.dim_ * 2),self.dim_)
            self.fc2 = nn.Linear(self.dim_, int(self.dim_ / 2))
            self.fc3 = nn.Linear(int(self.dim_ / 2),1195)
        elif self.task_id == 3: # 
            self.fc = nn.Linear(int(self.dim_ * 2),self.dim_)
            self.fc2 = nn.Linear(self.dim_, int(self.dim_ / 2))
            self.fc3 = nn.Linear(int(self.dim_ / 2),384)

        self.fc.apply(init_weights)
        self.fc2.apply(init_weights)
        self.fc3.apply(init_weights)

        
    #def forward(self, x, edge_index, edge_attr):
    def forward(self, h_list, edge_index,batch, edge_attr):        
        # print(torch.cat(h_list[:], dim = 1).size()) #2422, 4352 
        # # print( torch.sum(torch.cat(h_list[:], dim = 1), dim = 0).size())
        # node_representation = torch.sum(torch.cat(h_list[:], dim = 1), dim = 0)[0]

        node_representation = torch.cat(h_list[:], dim = 1)
        x = torch.cat([gmp(node_representation, batch), gap(node_representation, batch)], dim=1)

        if self.task_id == 0: # 
            pass 
        elif self.task_id == 1: #
            pass
        elif self.task_id == 2: # 
            x = F.relu(self.fc(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        elif self.task_id == 3: # 
            x = F.relu(self.fc(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x) 

        # if self.JK == "last":
        #     node_representation = h_list[-1]
        # elif self.JK == "sum":
        #     h_list = [h.unsqueeze_(0) for h in h_list]
        #     node_representation = torch.sum(torch.cat(h_list[1:], dim = 0), dim = 0)[0]

        return x


class GearNet_Edge_model(torch.nn.Module):
    """
    GNN w/ dropout
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0, task = 0, bases = 0):
        '''
            task: 0 - Enzyme Commision number prediction 
            task: 1 - Gene Ontology
            task: 2 - Fold Classification
            task  3 - Reaction 
        '''
        super(GearNet_Edge_model, self).__init__()
        self.encoder = GearNet_Edge(num_relations=7, dropout=drop_ratio, pre_train_level=0, task=True)
        self.decoder = GearNet_Decoder(num_layer, emb_dim, drop_ratio, task)


    def forward(self,data):
        batch = data.batch
        #print(batch)
        x, features, edge_index, edge_attr = self.encoder(data)
        out = self.decoder(features,edge_index,batch,edge_attr)
        return out



