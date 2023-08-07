import os
import argparse
import logging
from logging.handlers import QueueHandler, QueueListener
import resource
import json
import copy
import pickle as pkl


from collections import OrderedDict
import ml_collections
import torch
import torch.multiprocessing as mp
from einops import rearrange
import numpy as np


#from abfold.data import dataset
from abfold.trainer import dataset
from abfold.data.utils import save_ig_pdb, save_general_pdb

import sys
sys.path.append("/home/liushiwei/rebuttal/riemannian-score-sde")


ressymb_to_resindex = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,
}

resindex_to_ressymb = {v: k for k, v in ressymb_to_resindex.items()}
resindex_to_ressymb[21] = 'X'

from operator import itemgetter


folder_path = "./DiffAffinity_sample_results"
root, dirs, files = list(os.walk(folder_path))[0]

work_dir = "./pdb"

for pdbid in dirs:

    if not os.path.exists(f"{work_dir}/{pdbid}"):
        os.makedirs(f"{work_dir}/{pdbid}")
    
    name = pdbid
    with open(f"{folder_path}/{pdbid}/data.pickle", "rb") as f:
        data = pkl.load(f)
    
    aasymbl = itemgetter(*data["aa"].numpy()[0])(resindex_to_ressymb)
    aasymbl = "".join(aasymbl)

    with open(f"{folder_path}/{pdbid}/atom14.pickle", "rb") as f:
        atom14 = pkl.load(f)
    
    atom14_true = data['pos_heavyatom'][...,:14,:]
    atom14 = data['chi_masked_flag'][:,:,None,None].numpy() * atom14 + (1-data['chi_masked_flag'][:,:,None,None].numpy()) * atom14_true.numpy()

    nX = aasymbl.count('X')
    if nX > 0:
        print(aasymbl)

        pdb_file_pred = f"{work_dir}/{pdbid}/pred.pdb"
        save_general_pdb(aasymbl[:-nX], atom14[0][:-nX], pdb_file_pred)

        pdb_file_true = f"{work_dir}/{pdbid}/true.pdb"
        save_general_pdb(aasymbl[:-nX], atom14_true[0][:-nX], pdb_file_true)

    else:

        pdb_file_pred = f"{work_dir}/{pdbid}/pred.pdb"
        save_general_pdb(aasymbl, atom14[0], pdb_file_pred)

        pdb_file_true = f"{work_dir}/{pdbid}/true.pdb"
        save_general_pdb(aasymbl, atom14_true[0], pdb_file_true)        
    # break

    
