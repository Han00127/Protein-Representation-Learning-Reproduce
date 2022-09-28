import numpy as np 
from copy import deepcopy
from tqdm import tqdm
import os 
import os.path as osp
import sys 
import random
import math
import time
import argparse

import torch 
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader,DataListLoader
from torch_geometric.nn import DataParallel
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from utils import utils
from utils.loss import constastive_loss

from augmentation.transformation import MaskNode
from augmentation.dataset_generator import Self_ProteinDataset_GearNet_Edge

def main():
    parser = argparse.ArgumentParser(description='PyTorch base residue type prediction training')
    # parser.add_argument('--device', type=int, default=0,help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=96,help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0001,help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=1e-5,help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=6,help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.1,help='dropout ratio (default: 0.1)')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")

    parser.add_argument('--model_type', type=str, default='v1', help='refered to model type [v1, v2]')

    parser.add_argument('--save_path', type=str, default = '/project/rw/codingtest/saved_info/', help='directory training info')
    parser.add_argument('--filename', type=str, default = 'multi_constrastive', help='output filename')
    
    parser.add_argument('--resume', type=bool, default = False, help='resume')
    parser.add_argument('--resume_path', type=str, default = '/project/rw/codingtest/saved_info/', help='evaluating training or not')
    args = parser.parse_args()

    ########### set SEED ################
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    #####################################

    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()

    ########## Tensorbord add#########################
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.save_path)
    ##################################################
    '''
        swiss_protein_520k temp data 
        data = "/data/project/rw/codingtest/swiss_prot_520k_temp/raw/prot_temp_list.txt"
        data_dir = "/data/project/rw/codingtest/swiss_prot_520k_temp/pdb_files/"
        
        swiss_protein_520k
        data = "/data/project/rw/codingtest/swiss_prot_520k/raw/protein_list.txt"
        data_dir = "/data/project/rw/codingtest/swiss_prot_520k/protein/"
    '''

    # swiss_protein_520k
    data = "/data/project/rw/codingtest/swiss_prot_520k/raw/protein_list.txt"
    data_dir = "/data/project/rw/codingtest/swiss_prot_520k/protein/"


    # data = "/data/project/rw/codingtest/swiss_prot_520k_temp/raw/prot_temp_list.txt"
    # data_dir = "/data/project/rw/codingtest/swiss_prot_520k_temp/pdb_files/"
    data_list = utils.data_load(data)

    dataset1 = Self_ProteinDataset_GearNet_Edge(data_list=data_list, data_dir=data_dir, transform = MaskNode(0.25))
    loader1 = DataLoader(dataset1,batch_size=32, shuffle=False, pin_memory=True, num_workers=2)

    start_epoch = 0
    iteration = 0
    ## Generated model, optimizer, and training base setting. 
    if args.model_type == 'v1':
        from models.v1_message_model_d1 import GearNet_Edge
        model = GearNet_Edge(num_relations=7, dropout=args.dropout_ratio, pre_train_level=1, task = False)
    else:
        from models.v2_message_model import GearNet_Edge
        model = GearNet_Edge(num_relations=7, dropout=args.dropout_ratio, pre_train_level=1, task = False)
        
    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    if args.resume:
        saved_model = torch.load(args.resume_path,map_location=device)
        loaded_state_dict = saved_model['model_state_dict']
        from collections import OrderedDict
        pretrained_weight = OrderedDict()

        for pre_trained_param, model_param in zip(loaded_state_dict, model.state_dict()):
            if pre_trained_param[7:] == model_param:
                pretrained_weight[model_param] = loaded_state_dict[pre_trained_param]

        model.load_state_dict(pretrained_weight)

        optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        start_epoch = saved_model['epoch']+1
        iteration = saved_model['count']+1

        # optimizer state to gpu
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    model = DataParallel(model,device_ids=[0,1,2,3])
    model.to(device)
    model.train()


    # Criterion
    criterion = torch.nn.CrossEntropyLoss().to(device)
    for e in range(start_epoch,args.epochs):
        print("==== Epoch: {}====".format(e))
        train_loss_accum = 0
        start_time = time.time()
        total = 0
        accuracy = 0 
        with tqdm(loader1, unit='batch') as tepoch:
            for step, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {e}")
                batch_list = batch.to_data_list()
                node_label = torch.argmax(batch.masked_node_label, dim = 1).to(device)

                optimizer.zero_grad()
                with autocast():
                    x1 = model(batch_list)
                    logit = F.softmax(x1, dim=1)
                    loss = criterion(logit, node_label)
                
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                scaler.step(optimizer)
                scaler.update()
                
                _, predicted = torch.max(logit.data, 1)
                total += node_label.size()[0]
                accuracy += (predicted == node_label).sum().item()

                writer.add_scalar('Training loss Overall',loss,iteration)
                writer.add_scalar('Training Acc Overall', 100 * accuracy / total,iteration)
                tepoch.set_postfix(loss=loss.item())
                iteration+=1
        torch.save({
                'epoch': e,
                'count': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_accum,
                }, osp.join(args.save_path,args.filename+'_ckps_{}.pt'.format(e)))

if __name__ == "__main__":
    main()