# permutation test.......
import numpy as np
import scipy.io as scio
import h5py
import torch
import itertools as itools
from speechmodel.ridge_simple import bootstrap_ridge
from speechmodel.ridge_simple_node_count import bootstrap_ridge_node_count
from speechmodel.ridge_simple_cpu import bootstrap_ridge_cpu
from tqdm import tqdm
from statsmodels.stats import multitest
import pickle
import math
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

# logging.basicConfig(level=logging.DEBUG)
# import threading

def block_permute(time_size, window_size):
    offset = random.randint(0, time_size-1)
    x = [i+offset for i in range(time_size)]
    x[time_size-offset:] = [i for i in range(offset)]
    indchunks = list(zip(*[iter(x)]*window_size))
    if time_size%window_size != 0:
        start = time_size%window_size
        indchunks.append(x[-start:])
    random.shuffle(indchunks)
    res = list(itools.chain(*indchunks[:]))
    return torch.tensor(res)

def permutation_test(permute_times, window, train_feature, train_fmri, true_corrs, cuda0, cuda_device, save_file):
    print('doing regression.....')
    # alphas = np.logspace(1, 3, 20) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    alphas = np.array(10)
    nboots = 10 # Number of cross-validation runs.
    chunklen = 30 # 
    nresp, nvox = train_fmri.shape
    nchunks = int(nresp/(30*nboots))
    alarm_num_record = torch.zeros(train_fmri.shape[-1]).cuda(cuda0)

    for i in tqdm(range(permute_times), desc='doing permutation test:'):
        perm = block_permute(nresp, window)
        perm = perm.long()
        train_feature_perm = train_feature[perm, :]
        bscorrs, _ = bootstrap_ridge(train_feature_perm, train_fmri, alphas, nboots, chunklen, nchunks, 
                                                singcutoff=1e-6, cuda0=cuda0, cuda_device=cuda_device)
        corrs = bscorrs.mean(1)
        alarm_num_record += (corrs>true_corrs).float()
        if i%50 == 0:
            pickle.dump([i, alarm_num_record], open(save_file, 'wb'))
        
    return alarm_num_record

def permutation_test_cpu(permute_times, window, train_feature, train_fmri, true_corrs, save_file):
    print('doing regression.....')
    # alphas = np.logspace(1, 3, 20) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    alphas = np.array(10)
    nboots = 10 # Number of cross-validation runs.
    chunklen = 30 # 
    nresp, nvox = train_fmri.shape
    nchunks = int(nresp/(30*nboots))
    alarm_num_record = torch.zeros(train_fmri.shape[-1])

    for i in tqdm(range(1, permute_times+1), desc='doing permutation test:'):
        perm = block_permute(nresp, window)
        perm = perm.long()
        train_feature_perm = train_feature[perm, :]
        bscorrs, _ = bootstrap_ridge_cpu(train_feature_perm, train_fmri, alphas, nboots, chunklen, nchunks, 
                                                singcutoff=1e-6)
        corrs = bscorrs.mean(1)
        alarm_num_record += (corrs>true_corrs).float()
        if i%50 == 0:
            pickle.dump([i, alarm_num_record], open(save_file, 'wb'))
    return alarm_num_record

def permutation_test_node_count(permute_times, window, train_feature, train_fmri, true_corrs, save_file):
    print('doing regression.....')
    # alphas = np.logspace(1, 3, 20) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    alphas = np.array(10)
    nboots = 10 # Number of cross-validation runs.
    chunklen = 30 # 
    nresp, nvox = train_fmri.shape
    nchunks = int(nresp/(30*nboots))
    alarm_num_record = torch.zeros(train_fmri.shape[-1])

    for i in tqdm(range(1, permute_times+1), desc='doing permutation test:'):
        perm = block_permute(nresp, window)
        perm = perm.long()
        train_feature_perm = train_feature[perm, :]
        bscorrs, _ = bootstrap_ridge_node_count(train_feature_perm, train_fmri, alphas, nboots, chunklen, nchunks, 
                                                singcutoff=1e-6)
        corrs = bscorrs.mean(1)
        alarm_num_record += (corrs>true_corrs).float()
        if i%50 == 0:
            pickle.dump([i, alarm_num_record], open(save_file, 'wb'))
    return alarm_num_record

def fdr_bh(p_vals, fdr_q):
    rej, q=multitest.fdrcorrection(p_vals, fdr_q)
    num = rej.sum()
    return rej, num, q

def run_permutation(train_feature, train_fmri, true_corrs, permute_times, window,  cuda0, cuda1, save_file):
    alarm_num_record = permutation_test(permute_times, window, train_feature, train_fmri, true_corrs, cuda0, cuda1, save_file)
    pickle.dump([permute_times, alarm_num_record], open(save_file, 'wb'))

def run_permutation_cpu(train_feature, train_fmri, true_corrs, permute_times, window, save_file):
    alarm_num_record = permutation_test_cpu(permute_times, window, train_feature, train_fmri, true_corrs, save_file)
    pickle.dump([permute_times, alarm_num_record], open(save_file, 'wb'))

def run_permutation_node_count(train_feature, train_fmri, true_corrs, permute_times, window, save_file):
    alarm_num_record = permutation_test_node_count(permute_times, window, train_feature, train_fmri, true_corrs, save_file)
    pickle.dump([permute_times, alarm_num_record], open(save_file, 'wb'))

def compute_pval(alarm_num_record_path, fdr_q=0.05):
    alarm_num_record = pickle.load(open(alarm_num_record_path, 'rb'))
    permute_time = alarm_num_record[0]
    pdata = alarm_num_record[1].cpu()
    pvals = (pdata+1)/(permute_time+1)
    mask, num, q = fdr_bh(pvals, fdr_q)
    return mask, num, q
