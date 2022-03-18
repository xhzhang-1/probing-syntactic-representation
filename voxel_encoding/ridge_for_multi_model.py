"""
the code is adapted from https://github.com/HuthLab/speechmodeltutorial
"""
import numpy as np
from utils import mult_diag
import random
import itertools as itools
import torch
from tqdm import tqdm
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score function

def bootstrap_ridge_multi_model(oRstim, oRresp, alphas, nboots, chunklen, nchunks, res_path,
                    singcutoff=1e-10, cuda0=0, cuda1=1):
    nresp = len(oRresp) 

    Rcorrs = {'ori':[], 'pos':[], 'ner':[], 'srl':[], 'dep':[]}
    
    for bi in tqdm(range(1, nboots+1), desc=''):
        allinds = range(nresp)
        indchunks = list(zip(*[iter(allinds)]*chunklen))
        random.shuffle(indchunks)
        heldinds = list(itools.chain(*indchunks[:nchunks]))
        notheldinds = list(set(allinds)-set(heldinds))
        Rresp = oRresp[notheldinds,:].cuda(cuda1)
        Presp = oRresp[heldinds,:].cuda(cuda0)
        for feature in oRstim.keys():
            Rstim = oRstim[feature][notheldinds,:].cuda(cuda0)
            Pstim = oRstim[feature][heldinds,:].cuda(cuda0)
            
            ## Run ridge regression using this test set
            try:
                U,S,V = torch.svd(Rstim) #cuda 1
            except:
                U, S, V = np.linalg.svd(np.array(Rstim.cpu()), full_matrices=False)
                U = torch.tensor(U).cuda(cuda0)
                S = torch.tensor(S).cuda(cuda0)
                V = torch.tensor(V.T).cuda(cuda0)
            ngoodS = torch.sum(S>singcutoff)
            U = U[:,:ngoodS]
            S = S[:ngoodS]
            V = V[:,:ngoodS]

            UR = torch.matmul(U.transpose(0, 1).cuda(3), Rresp.cuda(3)).cuda(cuda0)
            PVh = torch.matmul(Pstim, V)

            S = S/(S**2+alphas**2) ## Reweight singular vectors by the (normalized?) ridge parameter
            pred = torch.matmul(mult_diag(S, PVh, left=False), UR)
            zPresp = zs(Presp)
            Rcorr = (zPresp*zs(pred)).mean(0)
    
            Rcorr[torch.isnan(Rcorr)] = 0
            Rcorrs[feature].append(Rcorr)
        # save the results every 500 bootstraps
        if bi % 500 == 0:
            pickle.dump(Rcorrs, open(res_path+'permute_corrs_'+str(bi)+'.pkl', 'wb'))
            Rcorrs = {'ori':[], 'pos':[], 'ner':[], 'srl':[], 'dep':[]}

