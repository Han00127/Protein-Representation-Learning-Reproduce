import torch 
'''
https://github.com/Shen-Lab/GraphCL/blob/fe93387ca251950000014ff96c3868a567c85813/transferLearning_MoleculeNet_PPI/bio/pretrain_graphcl.py
'''
def constastive_loss( x1, x2):
    tow = 0.07
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / tow)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim) + 1e-6
    loss = - torch.log(loss).mean()
    return loss