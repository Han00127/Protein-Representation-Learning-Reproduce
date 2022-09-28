import torch 
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader,DataListLoader
from torch_geometric.nn import DataParallel
import torch.optim as optim
import numpy as np 
from copy import deepcopy
from tqdm import tqdm
import os 
import os.path as osp
import sys 
import random
import math
import argparse

from utils import utils
from utils.loss import constastive_loss

from augmentation.fine_tune_dataset import Fold_Dataset_GearNet_Edge
from utils.dataloader import DataLoaderFinetune
from torch.cuda.amp import GradScaler, autocast

import time


from augmentation.fine_tune_dataset import Reaction_Dataset_GearNet_Edge

criterion = torch.nn.CrossEntropyLoss()
scaler = GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, device, loader, optimizer,count,writer):
    model.train()
    t_total = 0
    t_accuracy = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        label = batch.label.squeeze(1)
        # batch_size = label.size()[0]
        batch = batch.to_data_list()

        with autocast():
            out = model(batch)
            # out = torch.nn.functional.softmax(out, dim=1)
            loss = criterion(out,label.to(device))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(),10)
        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()
        
        _, predicted = torch.max(out.data, 1)
        t_total += label.size()[0]
        t_accuracy += (predicted == label.to(device)).sum().item()

        writer.add_scalar('Training loss Overall',loss,count)
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate',lr,count)
        
        train_loss_accum += float(loss.detach().cpu().item())
        count +=1
    train_acc_accum = (100 * t_accuracy / t_total)
    train_loss = train_loss_accum / (step+1)

    return train_acc_accum, train_loss,count


def eval(model, device, loader):
    model.eval()
    v_total = 0
    v_accuracy = 0
    valid_loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        v_label = batch.label.squeeze(1)
        """
            y_one_hot [0,0,0,1,0,0,0,0]    batch * num_label
                      [0,0,0,0,1,0,0,0]
                             ...
                      [0,0,0,0,0,0,0,1]
            v_label.squeeze(1) = [1,2,3,1,2,6] 1 x batch
            v_predicted = [6,1,5,4,3,2,4] 1 x batch
        """
        # y_one_hot = torch.zeros(v_label.size()[0],1195).scatter_(1,v_label,1)
        v_batch = batch.to_data_list()
        
        with torch.no_grad():
            with autocast():
                pred = model(v_batch)
                pred = torch.nn.functional.softmax(pred, dim=1)
                v_loss = criterion(pred,v_label.to(device))

        _, v_predicted = torch.max(pred.data, 1)
        v_total += v_label.size()[0]
        v_accuracy += (v_predicted == v_label.to(device)).sum().item()
        valid_loss_accum += float(v_loss.detach().cpu().item())

        ## AUC and ROC calculation#############
        # y_true.append(y_one_hot.detach().cpu())
        # y_scores.append(pred.detach().cpu())
        ########################################

    valid_acc_accum = (100 * v_accuracy / v_total)
    valid_loss = valid_loss_accum / (step+1)
    return valid_acc_accum, valid_loss 

def model_generation(args):
    if args.model_type == 'v1':
        if not args.decoder_pooling:
                from models.v1_message_model_d1 import GearNet_Edge_model
                model = GearNet_Edge_model(6,512, 0.2, 3)
        else:
                from models.v1_message_model_d2 import GearNet_Edge_model
                model = GearNet_Edge_model(6,512, 0.2, 3)
    else:
        from models.v2_message_model import GearNet_Edge_model
        model = GearNet_Edge_model(6,512, 0.2, 3)
    return model

def load_model_weight(args, model):
    saved_model = torch.load(args.resume_path,map_location=device)
    loaded_state_dict = saved_model['model_state_dict']
    from collections import OrderedDict
    pretrained_weight = OrderedDict()
    for pre_trained_param, model_param in zip(loaded_state_dict, model.state_dict()):
        if pre_trained_param[7:] == model_param:
            pretrained_weight[model_param] = loaded_state_dict[pre_trained_param]
    model.load_state_dict(pretrained_weight)
    return model,saved_model


def main():
    parser = argparse.ArgumentParser(description='PyTorch base fine_tuning')
    # parser.add_argument('--device', type=int, default=0,help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=8,help='input batch size 2 per GPU for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=300,help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,help='learning rate (default: 1e-3)')
    parser.add_argument('--decay', type=float, default=5e-4,help='weight decay (default: 5e-4)')
    parser.add_argument('--num_layer', type=int, default=6,help='number of GNN message passing layers (default: 6).')
    parser.add_argument('--emb_dim', type=int, default=512,help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,help='dropout ratio (default: 0.2)')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")

    parser.add_argument('--model_type', type=str, default='v1', help='refered to model type [v1, v2]')

    parser.add_argument('--save_path', type=str, default = '/project/rw/codingtest/saved_info/', help='directory training info')
    
    parser.add_argument('--resume', type=bool, default = False, help='resume')
    parser.add_argument('--resume_path', type=str, default = '/project/rw/codingtest/saved_info/', help='resume path')

    parser.add_argument('--test', type=bool, default = False, help='only test ')
    parser.add_argument('--decoder_pooling', type=bool, default = False, help= 'False: gap, True: cat(gap, gmp)')
    args = parser.parse_args()


    # ########### set SEED ################
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # #####################################

    # ######## Test dataset ##########################################################################
    test_data = '/data/project/rw/codingtest/downstream_data/reaction_data/testing.txt'
    test_data_dir = '/data/project/rw/codingtest/downstream_data/reaction_data/'
    test_data_list = utils.data_load(test_data)
    test_dataset = Reaction_Dataset_GearNet_Edge(test_data_list, test_data_dir, phase='testing')
    test_loader = DataLoader(test_dataset,batch_size=12, shuffle=False, pin_memory=True, num_workers=8)
    ##################################################################################################

    if args.test: 
        ## get model
        model = model_generation(args)
        model,_ = load_model_weight(args, model)
        model = DataParallel(model,device_ids=[0,1,2,3])
        model.to(device)
        test_acc,test_loss = eval(model, device, test_loader)
        print("==== Accuracy {} ====".format(test_acc))

    else:

        ######### Tensorbord add#########################
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.save_path)
        #################################################


        ######### Train dataset ##########################################################
        train_data = '/data/project/rw/codingtest/downstream_data/reaction_data/training.txt'
        train_data_dir = '/data/project/rw/codingtest/downstream_data/reaction_data/'
        valid_data = '/data/project/rw/codingtest/downstream_data/reaction_data/validation.txt'
        ##################################################################################

        train_data_list = utils.data_load(train_data)
        valid_data_list = utils.data_load(valid_data)
        
        train_dataset = Reaction_Dataset_GearNet_Edge(train_data_list, train_data_dir, phase='training')
        valid_dataset = Reaction_Dataset_GearNet_Edge(valid_data_list, train_data_dir, phase='validation')
        
        train_loader = DataLoader(train_dataset,batch_size=8, shuffle=True, pin_memory=True, num_workers=8,drop_last = True)
        val_loader = DataLoader(valid_dataset,batch_size=8, shuffle=False, pin_memory=True, num_workers=8)

        # ################################################################################################
        ## Generated model, optimizer, and training base setting. 
        model = model_generation(args)
        optimizer = optim.SGD(model.parameters(), lr=args.lr,weight_decay=args.decay)
        ###################################################################################################

        start_epoch = 0
        iteration = 0
        if args.resume:
            model, saved_model = load_model_weight(args, model)
            optimizer.load_state_dict(saved_model['optimizer_state_dict'])
            start_epoch = saved_model['epoch']+1
            lr = saved_model['learning_rate']
            optimizer.lr = lr
            iteration = saved_model['count']+1

        else:
            # load pretrained model
            if args.model_type != 'v2':
                print("Load pretrained GearNet-Edge")
                saved_model = torch.load('/data/project/rw/codingtest/saved_info/0919/multi_constrastive_pretrain_model_batch128_lr0.0001/ckps_epoch32.pt',map_location=device)
                loaded_state_dict = saved_model['model_state_dict']
                from collections import OrderedDict
                pretrained_weight = {}
                for pre_trained_param, model_param in zip(loaded_state_dict, model.encoder.state_dict()):
                    if pre_trained_param[7:] == model_param:
                        pretrained_weight[model_param] = loaded_state_dict[pre_trained_param]
                model.encoder.load_state_dict(pretrained_weight)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5, last_epoch=- 1, verbose=False)
    model = DataParallel(model,device_ids=[0,1,2,3])
    model.to(device)
    model.train()
    
    train_acc_list = []
    val_acc_list = []
    
    

    best_valid_acc = 0
    best_test_acc = 0

    for e in range(start_epoch, args.epochs):
        print("====epoch " + str(e))
        train_acc, train_loss,iteration = train(model, device, train_loader, optimizer,iteration,writer)
        lr_scheduler.step() 

        print("====Evaluation validation")
        valid_acc, valid_loss = eval(model, device, val_loader)
        
        writer.add_scalars('Training Info loss', {'train_loss':train_loss,
                                                  'valid_loss':valid_loss}, e)
        
        writer.add_scalars('Training Info ACC', {'train_acc': train_acc,
                                                 'valid_acc': valid_acc}, e)
        if best_valid_acc <= valid_acc:
            best_valid_acc = valid_acc
            torch.save({
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'learning_rate': optimizer.param_groups[0]['lr']
                        }, osp.join(args.save_path,"ckps_epoch{}_best.pt".format(e)))

        print("====Evaluation Test")
        test_acc,test_loss = eval(model, device, test_loader)
        print("==== Accuracy {} ====".format(test_acc))
        writer.add_scalars('Test ACC', {'test_acc': test_acc}, e)
        if best_test_acc <= test_acc:
            best_test_acc = test_acc
            torch.save({
                        'epoch': e,
                        'count': iteration,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'learning_rate': optimizer.param_groups[0]['lr']
                        }, osp.join(args.save_path,"test","ckps_epoch{}_best.pt".format(e)))

if __name__ == "__main__":
    main()

