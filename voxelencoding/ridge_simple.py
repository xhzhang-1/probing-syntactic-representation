from functools import reduce
import numpy as np
from voxelencoding.utils import mult_diag
import random
import itertools as itools
import torch
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score function

def bootstrap_ridge(oRstim, oRresp, alphas=10.0, nboots=10, chunklen=140, nchunks=20,
                    singcutoff=1e-5, cuda0=0, cuda1 = 1):
    
    nresp, _ = oRresp.shape
    valinds = [] ## Will hold the indices into the validation data for each bootstrap
    
    Rcmats = []
    for bi in tqdm(range(nboots)):
        allinds = range(nresp)
        indchunks = list(zip(*[iter(allinds)]*chunklen))
        random.shuffle(indchunks)
        heldinds = list(itools.chain(*indchunks[:nchunks]))
        notheldinds = list(set(allinds)-set(heldinds))

        valinds.append(heldinds)
        
        Rstim = oRstim[notheldinds,:].cuda(cuda0)
        Pstim = oRstim[heldinds,:].cuda(cuda0)
        Presp = oRresp[heldinds,:].cuda(cuda0)
        Rresp = oRresp[notheldinds,:].cuda(cuda1)
        
        ## Run ridge regression using this test set
        # start = time.time()
        try:
            U,S,V = torch.svd(Rstim) #cuda 1
        except:
            U, S, V = np.linalg.svd(np.array(Rstim.cpu()), full_matrices=False)
            U = torch.tensor(U).cuda(cuda0)
            S = torch.tensor(S).cuda(cuda0)
            V = torch.tensor(V.T).cuda(cuda0)
    
        ## Truncate tiny singular values for speed
        ngoodS = torch.sum(S>singcutoff)
        U = U[:,:ngoodS]
        S = S[:ngoodS]
        V = V[:,:ngoodS]

        UR = torch.matmul(U.transpose(0, 1).cuda(cuda1), Rresp).cuda(cuda0)
        PVh = torch.matmul(Pstim, V)
        
        S = S/(S**2+alphas**2) ## Reweight singular vectors by the (normalized?) ridge parameter
        pred = torch.matmul(mult_diag(S, PVh, left=False), UR)
        zPresp = zs(Presp)
        Rcorr = (zPresp*zs(pred)).mean(0)
  
        Rcorr[torch.isnan(Rcorr)] = 0
        Rcmats.append(Rcorr)

    allRcorrs = torch.stack(Rcmats, 1)
    wt = reduce(torch.matmul, [V, torch.diag(S/(S**2+alphas**2)), UR])

    return allRcorrs, wt
