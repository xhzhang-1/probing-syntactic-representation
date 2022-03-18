"""
permutation test to compute the brain mask that predictable to original word vectors 
"""
import pickle
from argparse import ArgumentParser
from utils import load_fmri, load_feature
from validation_permutation import run_permutation
from ridge_simple import bootstrap_ridge
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score function

if __name__ == "__main__":
    argp = ArgumentParser()
    argp.add_argument('--cuda0', type=int, default=0, help='cuda_device 0')
    argp.add_argument('--cuda1', type=int, default=1, help='cuda_device 1')
    argp.add_argument('--permute_times', type=int, default=10000, help='permutation times')
    argp.add_argument('--compute_true_corr', type=bool, default=False, help='true if need to compute true corrs')
    argp.add_argument('--fmri_path', type=str, default='data/encoding_dataset/', help='path root to load fmri')
    argp.add_argument('--feature_path', type=str, default='data/encoding_dataset/', help='path to load feature')
    argp.add_argument('--res_path', type=str, default='results/significant_voxels/', help='path to save results')
    argp.add_argument('--true_corr_root', type=str, default='results/significant_voxels/', help='path to load or save true corrs')
    args = argp.parse_args()

    train_fmri = load_fmri(args.fmri_root)
    train_feature = load_feature(args.feature_path)

    if args.compute_true_corr:
        bscorrs, _ = bootstrap_ridge(train_feature, train_fmri, chunklen=30, singcutoff=1e-10, cuda0=args.cuda0, cuda_device=args.cuda1)
        true_corrs = bscorrs.mean(1)
        print('compute true corrs...done')
        pickle.dump(true_corrs, open(args.true_corr_root+'true_corrs_'+args.sub+'_'+args.feature_type+'.pkl', 'wb'))
        
    true_corrs = pickle.load(open(args.true_corr_root+'true_corrs_'+args.sub+'_'+args.feature_type+'.pkl', 'rb'))
    true_corrs = true_corrs.cuda(args.cuda0)
    window = 30
    alarm_path = args.res_root+args.feature_type+'_alarm_num_record_'+args.sub+'_'+str(args.permute_times)+'_'+str(args.cuda1)+'.pkl'
        
    run_permutation(train_feature, train_fmri, true_corrs, \
        args.permute_times, window, args.cuda0, args.cuda1, alarm_path)