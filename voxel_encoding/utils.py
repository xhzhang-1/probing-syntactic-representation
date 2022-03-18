import numpy as np
import pickle
import h5py
import torch
from tqdm import tqdm
from statsmodels.stats import multitest
import statsmodels.stats.weightstats as st

def mult_diag(d, mtx, left=True):
    """
    the code is adapted from https://github.com/HuthLab/speechmodeltutorial
    Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.
    Input:
        d -- 1D (N,) array (contains the diagonal elements)
        mtx -- 2D (N,N) array

    Output:
        mult_diag(d, mts, left=True) == dot(diag(d), mtx)
        mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    
    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        # return (d*mtx.T).T
        return (d*mtx.transpose()).transpose()
    else:
        return d*mtx

def extract_block_boot_data(data_root, task, start=0, end=10001, times=10000):
    task_data = []
    for i in tqdm(range(start, end), desc='[loading data for task '+task+'... ]'):
        if i%500 == 0:
            corrs = pickle.load(open(data_root+'permute_corrs_'+str(i)+'.pkl', 'rb'))
            for j in corrs[task]:
                task_data.append(j.cpu())
    task_data = torch.stack(task_data)
    assert(task_data.shape[0] == times)

    return task_data

def z_test_pair(original, changed):
    p_val = st.ztest(x1=original, x2=changed, alternative='larger')
    return p_val

def z_test(data):
    p_val = st.ztest(data, alternative='larger')
    return p_val

def batch_ztest(data1, data2=None, method='single'):
    results = []
    mask = np.zeros(59412)
    count = 0
    zeros = 0
    for i in tqdm(range(data1.shape[1]), desc='doing pair z-test...'):
        if method == 'single':
            p_val = z_test(data1[:,i])
        else:
            p_val = z_test_pair(data1[:,i], data2[:,i])
        results.append(p_val[1])
        if p_val[1]<0.01:
            count += 1
            mask[i] = 1
        if p_val[1] == 0:
            zeros += 1
    print(zeros)
    return results, count, mask

def fdr_bh(p_vals, fdr_q):
    rej, q=multitest.fdrcorrection(p_vals, fdr_q)
    num = rej.sum()
    return rej, num, q

def load_fmri(fmri_path):
    train_fmri = torch.tensor([])
    for i in tqdm(range(1, 52), desc='loading fmri...'):
        fmri_file = fmri_path + '/encoding_dataset_single'+str(i)+'.mat'
        data = h5py.File(fmri_file, 'r')
        train_fmri = torch.cat([train_fmri, torch.tensor(np.array(data['fmri_response']))])

    return train_fmri

def load_feature(feature_path):
    train_feature = torch.tensor([])
    for i in tqdm(range(1, 52), desc='loading stimuli...'):
        feature_file = feature_path + '/story_%d.mat'%i
        data = h5py.File(feature_file, 'r')
        single_feature = torch.tensor(np.array(data['word_feature'])).transpose(0, 1)
        train_feature = torch.cat([train_feature, single_feature])
    return train_feature