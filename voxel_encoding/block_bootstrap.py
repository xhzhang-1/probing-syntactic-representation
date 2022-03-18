''' Testing whether the correlation difference between 
the original and one-feature-removed embeddings is significant.
Compute the correlation for all five types of embeddings, 
and store them for the use of significance test.'''

import numpy as np
from ridge_for_multi_model import bootstrap_ridge_multi_model
from argparse import ArgumentParser
from utils import load_fmri, load_feature, extract_block_boot_data, batch_ztest, fdr_bh
import pickle
import scipy.io as scio
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score function

def feature_block_bootstrap(train_feature, train_fmri, res_path, nboots):
    print('doing regression with all features.....')
    alphas = np.array(10)
    chunklen = 140 # 
    nchunks = 20
    bootstrap_ridge_multi_model(train_feature, train_fmri, alphas, nboots, \
                                chunklen, nchunks, res_path, cuda0=2, cuda1=3)

def significance_test(boot_corr_root, nboots, start=0, end=10001):
    p_vals = []
    q_vals = []
    tasks = ['ori', 'pos', 'ner', 'srl', 'dep']
    for i in tasks:
        data = extract_block_boot_data(boot_corr_root, i, start, end, nboots)
        pickle.dump(data, open(boot_corr_root+'corrs_for_'+i+'_'+str(nboots)+'.pkl', 'wb'))
    
    ori = pickle.load(open(boot_corr_root+'corrs_for_ori_'+str(nboots)+'.pkl', 'rb'))
    for i in tasks[1:]:
        changed = pickle.load(open(boot_corr_root+'corrs_for_'+i+'_500.pkl', 'rb'))
        p_val, count, mask = batch_ztest(ori, changed, 'pair')
        p_val = torch.tensor(p_val)
        mask, num, q_val = fdr_bh(p_val, 0.01)
        print(i+' significantly larger than ori: q\n '+str(num))
        scio.savemat(boot_corr_root+'ztest_'+i+'_larger_than_ori_pair_new.mat', {'mask':mask})
        scio.savemat(boot_corr_root+'ztest_'+i+'_larger_qvals_pair_new.mat', {'qval':q_val})
        p_vals.append(p_val)
        q_vals.append(q_val)

    pickle.dump(p_vals, open(boot_corr_root+'ztest_voxel_difference_pvals_pair.pkl', 'wb'))
    pickle.dump(q_vals, open(boot_corr_root+'ztest_voxel_difference_qvals_pair.pkl', 'wb'))

if __name__ == "__main__":
    argp = ArgumentParser()
    argp.add_argument('--fmri_path', help='fmri path')
    argp.add_argument('--feature_path', help='feature representation path')
    argp.add_argument('--res_path', help='result path')
    argp.add_argument('--nboots', default=10000, help='times to run block bootstrap')
    
    args = argp.parse_args()

    train_fmri = load_fmri(args.fmri_path)
    train_feature = load_fmri(args.fmri_path)

    weights = {}
    train_features = {}
    train_features['ori'] = train_feature
    tasks = ['ori', 'pos', 'ner', 'srl', 'dep']
    for task in tasks[1:]:
        spec_feature = load_feature(args.feature_path+'/null_'+task)
        train_features[task] = spec_feature
    feature_block_bootstrap(train_features, train_fmri, args.res_path)
    significance_test(args.res_path, args.nboots, 0, args.nboots)

