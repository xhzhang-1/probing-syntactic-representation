import torch
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from sklearn.manifold import TSNE

def plot_line(x):
    plt.figure()  
    plt.plot(x) 
    plt.show()

def plot_heatmap(data, data_type):
    data=np.array(data)
    plt.subplots(figsize=(9, 9))
    # sns.heatmap(data, annot=False, vmax=1, square=True, cmap="Blues")
    
    ax = sns.heatmap(data, annot=False, vmax=1, square=True, cmap="Blues", xticklabels=['Null_POS', 'Null_NE', 'Null_SR', 'Null_DEP'],\
        yticklabels=['POS', 'NE', 'SR', 'DEP'])
    # ax.set(font_scale=2.0)
    # plt.rc('font',family='Times New Roman',size=12)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()
    plt.savefig(data_type+'.png')

def plot_words_similarity(word_embeds):
    cos = []
    count = 0
    for i in word_embeds:
        cos.append([])
        for j in word_embeds:
            cos[count].append(torch.cosine_similarity(i.view(1,768),j.view(1,768)).cpu())
        count += 1
    cos = np.array(cos)
    plt.subplots(figsize=(9, 9))
    sns.heatmap(cos, annot=False, vmax=1, square=True, cmap="Blues")
    plt.show()

def pjanalysis(P):
    maxs = []
    mins = []
    for i in P:
        maxs.append(max(i))
        mins.append(min(i))
    return maxs, mins

def plot_precision_heatmap(intervals=None):
    tasks = ['pos', 'ner', 'srl', 'dep']
    if intervals == None:
        intervals = pickle.load(open('results_ontonotes/null_tests_final/5t_intervals.pkl', 'rb'))
    #baselines = {'pos':0.11186443511846043, 'ner':0.9013735390994666, 'srl':0.2639221858243508, 'dep':0.1174882930588212}
    #baselines = {'pos':0.1177, 'ner':0.1954, 'srl':0.4244, 'dep':0.1230, 'freq':0.3097}
    baselines = {'pos':0.1177, 'ner':0.1954, 'srl':0.4244, 'dep':0.1230}
    # ner f1: 0.1954
    means = []
    drops = []
    percent_drop = []
    fluct = []
    task_mean = []

    for i in tasks:
        means.append([])
        drops.append([])
        percent_drop.append([])
        fluct.append([])
        
        # means[-1].append(baselines[i])
        count = 0
        task_mean.append((intervals[i][0][0]+intervals[i][0][1])/2)
        for j in intervals[i][1:5]:
            mean = (j[0]+j[1])/2
            fluctuation = (j[1]-j[0])/2
            means[-1].append(mean)
            fluct[-1].append(fluctuation)
            drops[-1].append(task_mean[-1]-mean)
            temp = (task_mean[-1]-mean)/(task_mean[-1]-baselines[i])
            percent_drop[-1].append(temp)
            count += 1
    plot_heatmap(means, 'mean')
    plot_heatmap(drops, 'drop')
    # plot percent of drops, (task_mean[-1]-mean)/(task_mean[-1]-baseline)
    plot_heatmap(percent_drop, 'percent of drops')

def load_labels(file_path, p, name_lists, tag_dict=None):
    # count sample numbers of pos, ner, wf
    label_list = []
    idx = []
    count = 0
    with open(file_path, 'r') as rf:
        for line in rf:
            line = line.strip('\n').split('\t')
            if len(line) > 1:
                if line[p] in name_lists:
                    label_list.append(line[p])
                    idx.append(count)
                count += 1
    return label_list, idx

def get_nulled_list(input_hdf5_filepath, layer_index, projection_matrix):
    single_layer_features_list = []
    hf = h5py.File(input_hdf5_filepath, 'r') 
    indices = filter(lambda x: x != 'sentence_to_index', list(hf.keys()))
    single_layer_features_list = []
    #count = 0
    for index in tqdm(sorted([int(x) for x in indices]), desc='[computing word embeddings]'):
        feature_stack = hf[str(index)]
        single_layer_features = feature_stack[layer_index]
        #assert single_layer_features.shape[0] == len(observation.sentence)
        for i in single_layer_features:
            single_layer_features_list.append(np.matmul(i, projection_matrix))
            
    return single_layer_features_list

def load_elmo(input_hdf5_filepath, layer_index):
    single_layer_features_list = []
    hf = h5py.File(input_hdf5_filepath, 'r') 
    indices = filter(lambda x: x != 'sentence_to_index', list(hf.keys()))
    single_layer_features_list = []
    # count = 0
    for index in tqdm(sorted([int(x) for x in indices]), desc='[computing word embeddings]'):
        feature_stack = hf[str(index)]
        single_layer_features = feature_stack[layer_index]
        #assert single_layer_features.shape[0] == len(observation.sentence)
        for i in single_layer_features:
            single_layer_features_list.append(i)
    return np.vstack(single_layer_features_list)

def load_elmo_list(input_hdf5_filepath, layer_index):
    single_layer_features_list = []
    hf = h5py.File(input_hdf5_filepath, 'r') 
    indices = filter(lambda x: x != 'sentence_to_index', list(hf.keys()))
    single_layer_features_list = []
    # count = 0
    for index in tqdm(sorted([int(x) for x in indices]), desc='[computing word embeddings]'):
        feature_stack = hf[str(index)]
        single_layer_features = feature_stack[layer_index]
        #assert single_layer_features.shape[0] == len(observation.sentence)
        single_layer_features_list.append(single_layer_features)
    return single_layer_features_list

def get_dep_concats(filepath):
    if 'train' in filepath:
        deps = pickle.load(open('ud-english/train-dep.pkl', 'rb'))
    elif 'test' in filepath:
        deps = pickle.load(open('ud-english/test-dep.pkl', 'rb'))
    if 'dev' in filepath:
        deps = pickle.load(open('ud-english/dev-dep.pkl', 'rb'))
    sent_num = len(deps)
    concats = [[] for i in range(sent_num)]
    labels = [[] for i in range(sent_num)]
    p = 0
    discard = ['iobj', 'det:predet', 'csubj', 'remnant', 'csubjpass', 'vocative', 'dislocated', 'cc:preconj', 'reparandum', 'foreign', 'dep', 'goeswith']
    for i in deps:
        t = 0
        for j in i:
            if j[1] in discard:
                t += 1
                continue
            if j[0] != -1:
                concats[p].append([j[0], t])
                labels[p].append(j[1])
                if j[1] == 'root':
                    import pdb
                    pdb.set_trace()
            else:
                r = np.random.randint(len(deps[p]))
                concats[p].append([r, t])
                labels[p].append('none')
            t += 1
        p += 1
    return labels, concats

def load_dep_elmo(labels, concats, embeds):
    embed_list = []
    label_list = []
    import pdb
    pdb.set_trace()
    for concat, label, embed in zip(concats, labels, embeds):
        for i in range(len(label)):
            embed_list.append(np.append(embed[concat[i][0]], embed[concat[i][1]]))
            label_list.append(label[i])
    return embed_list, label_list

def get_srl_concats(filepath, sent_num=2211):
    srls = open(filepath+'srls_processed.txt', 'r')
    concats = [[] for i in range(sent_num)]
    labels = [[] for i in range(sent_num)]
    p = 0
    discard = ['ARGM-REC', 'ARGM-GOL', 'ARGM-DSP', 'ARGM-PRR', 'ARGM-COM', 'ARGM-PRX', 'ARGA', 'ARGM-LVB']
    for line in srls:
        if line == '\n':
            assert len(labels[p]) == len(concats[p])
            p += 1
            if p == sent_num:
            # print(concats[-1])
                break
            continue
        line = line.strip().split('\t')
        
        if 'V' in line:
            pv = line.index('V')
            stars = []
            for i in range(len(line)):
                if line[i] != '*' and line[i] != 'V':
                    if line[i] not in discard:
                        concats[p].append([pv, i])
                        labels[p].append(line[i])
                elif line[i] == '*':
                    stars.append(i)
            if len(stars) == len(line)-1:
                for i in range(2):
                    if i > len(stars)-1:
                        break
                    concats[p].append([pv, stars[i]])
                    labels[p].append('*')

        else:
            for i in range(len(line)-1):
                concats[p].append([i, i+1])
                labels[p].append('_')
            concats[p].append([len(line)-1, 0])
            labels[p].append('_')
    return labels, concats

def plot_all_layer_acc():
    layers = [str(i) for i in range(1, 13)]
    tasks = ['pos', 'ner', 'srl', 'dep']
    acc = {}
    layer_sum = [0 for i in range(12)]
    for t in tasks:
        acc[t] = []
    for l in layers:
        a = pickle.load(open('results/results_ontonotes/all_layer_of_bert/null_Ps_layer'+l+'/t_intervals.pkl', 'rb'))
        for t in tasks:
            temp = (a[t][0][0] + a[t][0][1])/2
            acc[t].append(temp)
            layer_sum[int(l)-1] += temp
    for t in tasks:
        plt.plot(acc[t], label=t)

    plt.plot(layer_sum)

if __name__ == "__main__":
    intervals = pickle.load(open('results/results_ontonotes/all_layer_of_bert/null_Ps_layer8/t_intervals.pkl', 'rb'))
    plot_precision_heatmap(intervals)