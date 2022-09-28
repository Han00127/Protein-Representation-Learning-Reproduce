import os 
import sys 
import gzip 

path = "/root/datasets/prot_data/protein_swissprot_520k/"
file_list = os.listdir(path)

# # files = os.listdir("/swissprot_pdb_vs")

path2 = "/root/datasets/prot_data/protein_swissprot_520k/"
count = 0
for file in file_list:
    f_name = file.split('.gz')[0]
    if f_name == '.DS_Store':
        pass
    else:
        os.system("gunzip -c {}.gz > /root/datasets/protein_dataset/protein_swissprot_520k/{}".format(path2+f_name,f_name))
        count+=1

print("All of thing are done, {}", count)


########################################################################################
## write protein_list###################################################################
########################################################################################

# path = "/root/datasets/prot_temp_data/"
# file_list = os.listdir(path)
# with open("/root/datasets/prot_temp_list.txt", 'w') as f:
#     for file in file_list:
#         f_name = file.split('.gz')[0]
#         f.write(f_name + '\n')
# f.close()