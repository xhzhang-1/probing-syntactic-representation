# from speechmodel.ridge_simple import bootstrap_ridge
from speechmodel.ridge_torch import bootstrap_ridge
from speechmodel.ridge_story import bootstrap_ridge_story
import scipy.io as scio
import torch
import numpy as np
from tqdm import tqdm
import h5py
import pickle
from scipy import signal
from speechmodel.pearson import pearson
from validation_permutation import run_permutation

zs = lambda v: (v-v.mean(0))/v.std(0)

def load_fmri(fmri_path, language='zh'):
    train_fmri = torch.tensor([])
    # train_fmri = []

    if language == 'zh':
        for i in tqdm(range(1, 61), desc='loading fmri...'):
            # if i == test_story:
            #     continue
            fmri_file = fmri_path+'/story_'+str(i)+'.mat'
            data = h5py.File(fmri_file, 'r')
            single_fmri = np.array(data['fmri_response'])
            # train_fmri.append(torch.FloatTensor(single_fmri))
            train_fmri = torch.cat([train_fmri, torch.FloatTensor(single_fmri)])
    
    elif language == 'en':
        train_fmri = torch.tensor([])
        # train_feature = torch.tensor([])
        for i in tqdm(range(1, 52), desc='loading fmri...'):
            # filepath = 'encoding_data/fmri_encoding_dataset/encoding_dataset_single'+str(i)+'.mat'
            fmri_file = fmri_path + 'encoding_dataset_single'+str(i)+'.mat'
            data = h5py.File(fmri_file, 'r')
            # single_feature = torch.matmul(single_feature, projection_matrix)
            train_fmri = torch.cat([train_fmri, torch.tensor(np.array(data['fmri_response']))])
            
        '''filepath = 'data/encoding_data/story_bert_feature_zscored/story_test_layer7_zscored.mat'
        data = h5py.File(filepath, 'r')
        test_feature = torch.tensor(np.array(data['word_feature'])).transpose(0, 1)
        filepath = 'data/encoding_data/fmri_encoding_dataset/encoding_dataset_test_single.mat'
        data = h5py.File(filepath, 'r')
        test_fmri = torch.tensor(np.array(data['fmri_response']))'''
        # return train_fmri, train_feature, test_fmri, test_feature

    else:
        raise('Unknown language!')
        
    # return train_fmri, test_fmri
    return train_fmri

def load_feature(feature_path, language='zh', zscore=False):
    train_feature = torch.tensor([])
    # train_feature = []
    if language == 'zh':
        for i in tqdm(range(1, 61), desc='loading stimuli...'):
            # if i == test_story:
            #     continue
            feature_file = feature_path + '/story_'+str(i)+'.mat'
            data = h5py.File(feature_file, 'r')
            # single_feature = torch.tensor(np.array(data['word_feature']))
            single_feature = torch.tensor(np.array(data['word_feature'])).transpose(0, 1)
            if zscore:
                single_feature = zs(single_feature)
            # single_feature = torch.matmul(single_feature, projection_matrix)
            train_feature = torch.cat([train_feature, single_feature])
            # train_feature.append(single_feature)
    elif language == 'en':
        for i in tqdm(range(1, 52), desc='loading stimuli...'):
            feature_file = feature_path + 'story_%d.mat'%i
            data = h5py.File(feature_file, 'r')
            single_feature = torch.tensor(np.array(data['word_feature'])).transpose(0, 1)
            if zscore:
                single_feature = zs(single_feature)
            train_feature = torch.cat([train_feature, single_feature])
    else:
        raise('Unknown language!')
        
    # return train_feature, test_feature
    return train_feature

def fmri_filt(fmri_data, Wn, filt_type='bandpass', axis=0):
    # b, a = signal.butter(N, Wn, 'lowpass')
    # N:滤波器阶数
    # Wn：归一化截止频率。计算公式Wn=2*截止频率/采样频率。
    #（注意：根据采样定理，采样频率要大于两倍的信号本身最大的频率，才能还原信号。fMRI的采样频率是1.40845，那真实的信号频率小于0.7？
    # 截止频率一定小于信号本身最大的频率，所以Wn一定在0和1之间）。当构造带通滤波器或者带阻滤波器时，Wn为长度为2的列表。
    # 带通，0.01~0.08
    Wn = [0.0142, 0.1135]

    b, a = signal.butter(4, Wn, filt_type)
    filted_fmri = signal.filtfilt(b, a, fmri_data, axis)
    # temp = single_fmri.copy()
    # single_fmri = temp
    return filted_fmri

def compute_p(P):
    val, vec = np.linalg.eig(P)
    S = np.eye(1024)
    for i in range(1024):
        S[i][i] = 1 - val[i]
    return np.matmul(np.matmul(vec, S), vec.T)

def label_response(label_vector_path, weight):
    label_vectors = pickle.load(open(label_vector_path, 'rb'))
    count = 0
    for i in label_vectors:
        res = np.matmul(i, weight)
        scio.savemat('results/results_bert/label_fmri_responses/label_%d.mat'%count, {'response':res})
        count += 1

def block_permutation(train_feature, train_fmri, true_corrs_path, permute_time, block_size, feature_type, res_path, cuda0, cuda1):
    true_corrs = scio.loadmat(true_corrs_path)
    true_corrs = true_corrs['corrs']
    true_corrs = torch.tensor(true_corrs).cuda().reshape(59412)
    run_permutation(train_feature, train_fmri, true_corrs, permute_time, block_size, feature_type, res_path, cuda0, cuda1)

def leave_one_out_encoding(fmri_root, feature_path, sub):
    fmri_path = fmri_root + sub
    total_fmri = load_data(fmri_path, 'zh')
    total_feature = load_feature(feature_path, 'zh')
    alphas = np.logspace(1, 3, 10)
    nstories = 60
    chunklen = 140
    corrs = bootstrap_ridge_story(total_feature, total_fmri, alphas, nstories,
                                chunklen, traincor_only=True)
    import pdb
    pdb.set_trace()
    print('subject '+sub+' leave-one-out cross-validation, mean corrs: '+str(corrs.mean()))
    scio.savemat('results_zh/ligang_final/results_elmo/09/leave_one_out_both_zscore.mat', {'corrs':np.array(corrs.cpu())})
    
