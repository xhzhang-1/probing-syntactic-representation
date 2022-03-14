# 64
''' Testing whether the correlation difference between 
the original and one-feature-removed embeddings is significant.
Compute the correlation for all five types of embeddings, 
and store them for the use of significance test.'''

import numpy as np
import scipy.io as scio
import h5py
import torch
from speechmodel.utils import mult_diag, counter
import itertools as itools
from speechmodel.pearson import pearson
from speechmodel.ridge_torch import bootstrap_ridge
from speechmodel.ridge_for_multi_model import bootstrap_ridge_multi_model
from tqdm import tqdm
import pickle
import math
import random
from functools import reduce
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score function

def load_data():
    train_fmri = torch.tensor([])
    train_feature = torch.tensor([])
    # filepath = 'data/encoding_data/fmri_encoding_dataset/encoding_dataset_test_single.mat'
    # data = h5py.File(filepath, 'r')
    # test_fmri = torch.tensor(np.array(data['fmri_response']))
    # test_feature = torch.tensor(np.array(data['word_feature'])).transpose(0, 1)
    
    for i in tqdm(range(1, 52), desc='loading fmri and stimulus...'):
        filepath = 'data/encoding_data/fmri_encoding_dataset/encoding_dataset_single'+str(i)+'.mat'
        data = h5py.File(filepath, 'r')
        # single_feature = torch.matmul(single_feature, projection_matrix)
        train_fmri = torch.cat([train_fmri, torch.tensor(np.array(data['fmri_response']))])
        # filepath = 'data/encoding_data/story_glove/glove_story_%d.mat'%i
        # data = h5py.File(filepath, 'r')
        # filepath = 'data/encoding_data/story_bert_layer8_encoding/story_%d_layer7_zscored.mat'%i
        # filepath = 'data/encoding_data/story_bert_layer8_encoding/story_%d.mat'%i
        filepath = 'data/encoding_data/story_bert_feature_layer8/zscore/story_%d_layer8_zscored.mat'%i
        data = h5py.File(filepath, 'r')
        single_feature = torch.tensor(np.array(data['word_feature'])).transpose(0, 1)
        train_feature = torch.cat([train_feature, single_feature])
        #train_fmri.append(torch.tensor(np.array(data['fmri_response'])).cuda())
        #train_feature.append(torch.tensor(np.array(data['word_feature'])).transpose(0, 1).cuda())
    return train_fmri, train_feature


def compute_cor(prefix, train_feature, train_fmri, test_feature, test_fmri):
    print('doing regression.....')
    alphas = np.logspace(1, 3, 20) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    nboots = 10 # Number of cross-validation runs.
    chunklen = 140 # 
    nchunks = 20
    #import pdb
    #pdb.set_trace()
    wt, corr, alphas, _, _ = bootstrap_ridge(train_feature, train_fmri, test_feature, test_fmri,
                                                        alphas, nboots, chunklen, nchunks,
                                                        singcutoff=1e-10, single_alpha=False)

    # voxcorrs = pearson(pred, test_fmri)
    save_fn = 'results_ontonotes/encoding_weights/4weights_null'+prefix+'.mat'
    scio.savemat(save_fn, {'correlations': np.array(corr), 'weights': np.array(wt)})
    pickle.dump(wt, open('results_ontonotes/encoding_weights/4weights_null'+prefix+'.pkl', 'wb'))
    pickle.dump(corr, open('results_ontonotes/encoding_weights/4corr_null'+prefix+'.pkl', 'wb'))

# zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score function
def ridge_corr(oRstim, oRresp, heldinds, notheldinds, cuda_device, singcutoff=1e-10):
    Rstim = oRstim[notheldinds,:].cuda()
    Pstim = oRstim[heldinds,:].cuda()
    Presp = oRresp[heldinds,:].cuda()
    Rresp = oRresp[notheldinds,:].cuda(cuda_device)
        
    ## Run ridge regression using this test set
    U,S,V = torch.svd(Rstim) #cuda 1
    
    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = torch.sum(S>singcutoff)
    nbad = origsize-ngoodS
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    V = V[:,:ngoodS]
    # logger.info("Dropped %d tiny singular values.. (U is now %s)"%(nbad, str(U.shape)))

    UR = torch.matmul(U.transpose(0, 1).cuda(cuda_device), Rresp).cuda()
    PVh = torch.matmul(Pstim, V)
        
    S = S/(S**2+alphas**2) ## Reweight singular vectors by the (normalized?) ridge parameter
    pred = torch.matmul(mult_diag(S, PVh, left=False), UR)
    zPresp = zs(Presp)
    Rcorr = (zPresp*zs(pred)).mean(0)
  
    Rcorr[torch.isnan(Rcorr)] = 0
    return Rcorr

def bootstrap_ridge_comparison(oRstim, oRresp, alphas=10.0, nboots=10, chunklen=140, nchunks=20, 
                    dtype=torch.tensor, corrmin=0.2, joined=None, singcutoff=1e-10, normalpha=False,
                    use_corr=True, cuda_device = 1):
    
    nresp, nvox = oRresp.shape
    valinds = [] ## Will hold the indices into the validation data for each bootstrap
    
    Rcmats = []

    for bi in counter(range(nboots), countevery=1, total=nboots):
        # logger.info("Selecting held-out test set..")
        # start = time.time()
        allinds = range(nresp)
        indchunks = list(zip(*[iter(allinds)]*chunklen))
        random.shuffle(indchunks)
        heldinds = list(itools.chain(*indchunks[:nchunks]))
        notheldinds = list(set(allinds)-set(heldinds))

        valinds.append(heldinds)
        Rcorr = ridge_corr(oRstim, oRresp, heldinds, notheldinds, cuda_device)
        Rcmats.append(Rcorr)

    allRcorrs = torch.stack(Rcmats, 1)

    return allRcorrs

def compute_weights(train_fmri, train_feature, test_fmri, test_feature, task):
    if task == 'ori':
        compute_cor(task, train_feature, train_fmri, test_feature, test_fmri)
        return train_feature
    else:
        # projection_matrix = pickle.load(open('results_ontonotes/null_Ps/null_'+task+'.pkl', 'rb'))
        projection_matrix = pickle.load(open('results/results_ontonotes/all_layer_of_bert/null_Ps_layer8/'+task+'_mean_P.pkl', 'rb'))
        projection_matrix = torch.tensor(projection_matrix).float()
        if projection_matrix.shape[0] == 2048:
            # projection_matrix = torch.matmul(projection_matrix[0:1024, 0:1024], projection_matrix[1024:, 1024:])
            projection_matrix = projection_matrix[0:1024, 0:1024]
        nulled_train_feature = torch.matmul(train_feature, projection_matrix)
        # nulled_test_feature = torch.matmul(test_feature, projection_matrix)
        # compute_cor(task, nulled_train_feature, train_fmri, nulled_test_feature, test_fmri)
        return nulled_train_feature

def compute_feature(task):
    # projection_matrix = pickle.load(open(p_path+task+'_mean_P.pkl', 'rb'))
    # projection_matrix = torch.tensor(projection_matrix).float()
    # if projection_matrix.shape[0] == 2048:
        # projection_matrix = torch.matmul(projection_matrix[0:1024, 0:1024], projection_matrix[1024:, 1024:])
    #     projection_matrix = projection_matrix[0:1024, 0:1024]
    # spec_feature = torch.matmul(train_feature, projection_matrix)
    feature = torch.tensor([])
    for i in tqdm(range(1, 52), desc='loading '+task+' stimulus...'):
        filepath = 'data/encoding_data/story_bert_feature_layer8/'+task+'_z/story_%d_layer8_zscored.mat'%i
        data = h5py.File(filepath, 'r')
        single_feature = torch.tensor(np.array(data['word_feature'])).transpose(0, 1)
        feature = torch.cat([feature, single_feature])
        #train_fmri.append(torch.tensor(np.array(data['fmri_response'])).cuda())
        #train_feature.append(torch.tensor(np.array(data['word_feature'])).transpose(0, 1).cuda())
    return feature

def feature_t_test(train_feature, train_fmri, res_path):
    print('doing regression with all features.....')
    #alphas = np.logspace(1, 3, 20) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    alphas = np.array(10)
    nboots = 10000 # Number of cross-validation runs.
    chunklen = 140 # 
    nchunks = 20
    Rcorrs = bootstrap_ridge_multi_model(train_feature, train_fmri, alphas, nboots, chunklen, nchunks, res_path,
                                                        singcutoff=1e-10, single_alpha=True, cuda0=2, cuda_device=3)
    import pdb
    pdb.set_trace()
    pickle.dump(Rcorrs, open(res_path+'corrs_for_all_features.pkl', 'wb'))
    # pickle.dump(valinds, open('results_ontonotes/significant_voxels/valinds_for_all_features.pkl', 'wb'))


train_fmri, train_feature = load_data()
# train_feature = zs(train_feature)
# train_fmri = zs(train_fmri)

weights = {}
train_features = {}
train_features['ori'] = train_feature
p_path = 'results/results_bert8/results_analysis/'
tasks = ['ori', 'pos', 'ner', 'srl', 'dep']
for task in tasks[1:]:
    # spec_feature = compute_weights(train_fmri, train_feature, test_fmri, test_feature, task)
    # wt = pickle.load(open('results_ontonotes/encoding_weights/weights_null'+task+'.pkl', 'rb'))
    # weights[task] = wt
    spec_feature = compute_feature(task)
    # spec_feature = compute_weights(train_fmri, train_feature, [], [], task)
    train_features[task] = spec_feature
# res_path = 'results/results_bert/encoding_results_spe_zscore/'
res_path = 'results/results_bert8/results_analysis/'
feature_t_test(train_features, train_fmri, res_path)

