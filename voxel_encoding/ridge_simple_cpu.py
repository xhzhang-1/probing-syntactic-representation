from voxelencoding.utils import mult_diag
import random
import itertools as itools
import torch

zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score function

def bootstrap_ridge_cpu(oRstim, oRresp, alphas=10.0, nboots=10, chunklen=30,
                    nchunks=10, singcutoff=1e-10):
    
    nresp, nvox = oRresp.shape
    valinds = [] ## Will hold the indices into the validation data for each bootstrap
    
    Rcmats = []
    for bi in tqdm(range(nboots), desc='doing regression...'):
        allinds = range(nresp)
        indchunks = list(zip(*[iter(allinds)]*chunklen))
        random.shuffle(indchunks)
        heldinds = list(itools.chain(*indchunks[:nchunks]))
        notheldinds = list(set(allinds)-set(heldinds))

        valinds.append(heldinds)
        
        Rstim = oRstim[notheldinds,:]
        Pstim = oRstim[heldinds,:]
        Presp = oRresp[heldinds,:]
        Rresp = oRresp[notheldinds,:]
        
        ## Run ridge regression using this test set
        # start = time.time()
        U,S,V = torch.svd(Rstim) #cuda 1
    
        ## Truncate tiny singular values for speed
        ngoodS = torch.sum(S>singcutoff)
        U = U[:,:ngoodS]
        S = S[:ngoodS]
        V = V[:,:ngoodS]

        UR = torch.matmul(U.transpose(0, 1), Rresp).cpu()
        PVh = torch.matmul(Pstim, V)
        
        S = S/(S**2+alphas**2) ## Reweight singular vectors by the (normalized?) ridge parameter
        pred = torch.matmul(mult_diag(S, PVh, left=False), UR)
        zPresp = zs(Presp)
        Rcorr = (zPresp*zs(pred)).mean(0)
  
        Rcorr[torch.isnan(Rcorr)] = 0
        Rcmats.append(Rcorr)

    allRcorrs = torch.stack(Rcmats, 1)

    return allRcorrs
