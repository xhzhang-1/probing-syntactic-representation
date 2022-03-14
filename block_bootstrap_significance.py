# 64
import pickle
import numpy as np
from scipy import stats
from scipy.stats import norm
import statsmodels.stats.weightstats as st
import scipy.io as scio
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from statsmodels.stats import multitest
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


def extract_data(data_root, task, start=0, end=10000, times=10000):
    task_data = []
    for i in tqdm(range(start, end), desc='[loading data for task '+task+'... ]'):
        if i%500 == 0:
            corrs = pickle.load(open(data_root+'permute_corrs_'+str(i)+'.pkl', 'rb'))
            for j in corrs[task]:
                task_data.append(j.cpu())
    corrs = pickle.load(open(data_root+'permute_corrs.pkl', 'rb'))
    for j in corrs[task]:
        task_data.append(j.cpu())
    '''corrs = pickle.load(open(data_root+'permute_corrs_99.pkl', 'rb'))
    for j in corrs[task]:
        task_data.append(j.cpu())
    corrs = pickle.load(open(data_root+'permute_corrs_0_1.pkl', 'rb'))
    for j in corrs[task]:
        task_data.append(j.cpu())'''
    task_data = torch.stack(task_data)
    assert(task_data.shape[0] == times)
    #import pdb
    #pdb.set_trace()
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

def independent_ttest(original, changed):
    if stats.levene(original, changed).pvalue > 0.01:
        p_val = stats.ttest_ind(original, changed)
    else:
        p_val = stats.ttest_ind(original, changed, equal_var=False)
    return p_val

def pair_ttest(original, changed):
    p_val = stats.ttest_rel(original, changed)
    return p_val

def t_interval(data):
    data = np.array(data)
    mean=data.mean()
    std=data.std()
    interval=stats.t.interval(0.95,data.shape[0]-1,mean,std)
    return interval

def fdr_bh_old(p_vals, fdr_q):
    vals, indices = p_vals.sort()
    mask =np.zeros(p_vals.shape)
    m = p_vals.shape[0]
    num = 0
    q = np.zeros(m)
    for count in range(0, m):
        q[indices[count]] = vals[count]*m/(count+1)
        if vals[count] > fdr_q*(count+1)/m:
            continue
        else:
            mask[indices[count]] = 1
            num += 1
    return mask, num, q

def fdr_bh(p_vals, fdr_q):
    rej, q=multitest.fdrcorrection(p_vals, fdr_q)
    num = rej.sum()
    return rej, num, q

def batch_ttest(data1, data2):
    results = []
    mask = np.zeros(59412)
    count = 0
    for i in tqdm(range(data1.shape[1]), desc='doing pair t-test...'):
        p_val = independent_ttest(data1[:,i], data2[:,i])
        results.append(p_val.pvalue)
        if p_val.pvalue<0.01:
            count += 1
            mask[i] = 1
    return results, count, mask

def compute_t_interval(data):
    intervals = []
    #import pdb
    #pdb.set_trace()
    for i in range(data.shape[0]):
        intervals.append(t_interval(data[:, i]))
        # p_vals.append(single_ttest(original, changed[i]))
    return intervals


intervals = []
p_vals = []
q_vals = []
tasks = ['ori', 'pos', 'ner', 'srl', 'dep']
data_root = 'results/results_bert/encoding_results_spe_zscore/'
# data_root = 'results/results_bert8/results_analysis/'
'''for i in tasks:
    data = extract_data(data_root, i, 0, 1, 500)
    pickle.dump(data, open(data_root+'corrs_for_'+i+'_500.pkl', 'wb'))
'''
ori = pickle.load(open(data_root+'corrs_for_ori_500.pkl', 'rb'))

for i in tasks[1:]:
    # the results of one sample ztest is worse than two-sample.
    changed = pickle.load(open(data_root+'corrs_for_'+i+'_500.pkl', 'rb'))
    match = ori-changed
    # p_val, count, mask = batch_ztest(match)
    p_val, count, mask = batch_ztest(ori, changed, 'pair')
    p_val = torch.tensor(p_val)
    mask, num, q_val = fdr_bh(p_val, 0.01)
    print(i+' significantly different from ori: p\n '+str(count))
    print(i+' significantly different from ori: q\n '+str(num))
    #pickle.dump(mask, open('results_ontonotes/useful_results/corrs_for_'+i+'_10000.pkl', 'wb'))
    # scio.savemat(data_root+'ztest_'+i+'_smaller_than_ori_pair_new.mat', {'mask':mask})
    # scio.savemat(data_root+'ztest_'+i+'_smaller_qvals_pair_new.mat', {'qval':q_val})
    p_vals.append(p_val)
    q_vals.append(q_val)

import pdb
pdb.set_trace()
#pickle.dump(intervals, open('results_ontonotes/useful_results/voxel_correlation_interval.pkl', 'wb'))
pickle.dump(p_vals, open(data_root+'ztest_voxel_difference_pvals_pair.pkl', 'wb'))
pickle.dump(q_vals, open(data_root+'ztest_voxel_difference_qvals_pair.pkl', 'wb'))