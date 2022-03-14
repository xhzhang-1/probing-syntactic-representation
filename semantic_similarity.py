'''Computing whether bnlp would affect the semantic similarity.
Method: compute the correlation between human-rating and cosine similarity of embeddings.
'''

import pickle
import numpy as np
import torch
from tqdm import tqdm
from match_story_bert import match_tokenized_to_untokenized
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer
subword_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')
'''from allennlp.commands.elmo import ElmoEmbedder
options_file = "../elmo_files/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "../elmo_files/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = ElmoEmbedder(options_file, weight_file)'''

def compute_elmo(text, words):
    res = []
    for i in range(2):
        res.append([])
        line = text[i].strip('\n').split(' ')
        elmo_embed = elmo.embed_sentence(line)
        for j in range(len(line)):
            if words[i] == line[j]:
                temp = elmo_embed[:,j,:]
                if temp.shape != (3, 1024):
                    import pdb
                    pdb.set_trace()
                res[-1].append(elmo_embed[:,j,:])
    return res

def compute_bert(text, words, layer):
    res = []
    for i in range(2):
        res.append([])
        tokenized_sent = subword_tokenizer.wordpiece_tokenizer.tokenize('[CLS] ' + text[i] + ' [SEP]')
        line = text[i].strip('\n').split(' ')
        indexed_tokens = subword_tokenizer.convert_tokens_to_ids(tokenized_sent)
        segment_ids = [1 for x in tokenized_sent]
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segment_ids])
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)
        untokenized_sent = line
        untok_tok_mapping = match_tokenized_to_untokenized(tokenized_sent, untokenized_sent)
        bert_embed = encoded_layers[layer][0]
        assert bert_embed.shape[0] == len(tokenized_sent)
        # single_layer_features = torch.tensor([np.mean(single_layer_features[untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], axis=0) for i in range(len(untokenized_sent))])
        bert_embed =[torch.mean(bert_embed[untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], 0) for i in range(len(untokenized_sent))]
        assert len(bert_embed) == len(untokenized_sent)
        for j in range(len(line)):
            if words[i] == line[j]:
                temp = bert_embed[j]
                res[-1].append(temp)
    return res

def load_data(data_path, bert_layer, embed=None):
    text = []
    rating = []
    word_embed = []
    with open(data_path, 'r') as rf:
        for line in rf:
            line = line.strip().split('\t')
            text = line[5:7]
            rating.append(line[7])
            words = [line[1], line[3]]
            word_embed.append(compute_bert(text, words, bert_layer))

    return rating, word_embed

def compute_pearson(rating, embed_list, p_matrix, task_dim, layer):
    e_rating = []
    k = np.zeros([1, task_dim])
    l = np.zeros([1, task_dim])
    valid = []
    if p_matrix.shape[0] == 2*task_dim:
        p_matrix = p_matrix[0:task_dim, 0:task_dim]
    p_matrix = torch.FloatTensor(p_matrix)
    for i in range(len(embed_list)):
        try:
            if embed_list[i][0]==[] or embed_list[i][1] == []:
                continue
            # k = np.matmul(p_matrix, ((np.sum(embed_list[i][0], 0))/len(embed_list[i][0]))[layer])
            # l = np.matmul(p_matrix, ((np.sum(embed_list[i][1], 0))/len(embed_list[i][1]))[layer])
            k = torch.matmul(p_matrix, ((torch.sum(torch.stack(embed_list[i][0]), 0))/len(embed_list[i][0])))
            l = torch.matmul(p_matrix, ((torch.sum(torch.stack(embed_list[i][1]), 0))/len(embed_list[i][1])))
            e_rating.append(torch.cosine_similarity(k,l, 0))
            # e_rating.append(np.dot(k,l)/(torch.norm(k)*(np.linalg.norm(l))))
            valid.append(i)
        except:
            print('error')
            import pdb
            pdb.set_trace()
    return np.corrcoef(e_rating, rating[valid])


if __name__ == "__main__":

    # projection_path = 'results/results_bert/results_analysis/'
    # changed on 2021.9.7
    projection_path = 'results/results_ontonotes/all_layer_of_bert/null_Ps_layer7/'
    tasks = ['pos', 'ner', 'srl', 'dep']

    '''srl = pickle.load(open('data/ontonotes/new_srl_labels.pkl', 'rb'))
    dep = pickle.load(open('data/ontonotes/new_dep_labels.pkl', 'rb'))
    pos = pickle.load(open('ontonotes/new_pos_labels.pkl', 'rb'))
    ner = pickle.load(open('ontonotes/new_ner_labels.pkl', 'rb'))'''

    # tag_dict = {}

    projection_matrix = {}
    for i in tasks[0:4]:
        # tag_dict[i] = pickle.load(open('ontonotes/new_'+i+'_labels.pkl', 'rb'))
        projection_matrix[i] = pickle.load(open(projection_path+i+'_mean_P.pkl', 'rb'))
        # print(projection_matrix[i].shape)
    
    data_path = 'semantic_similarity/SCWS/ratings.txt'
    # rating, word_embed = load_data(data_path, bert_layer=6)
    # pickle.dump([rating, word_embed], open('semantic_similarity/SCWS/word_embed_bert7.pkl', 'wb'))
    rating, word_embed = pickle.load(open('semantic_similarity/SCWS/word_embed_bert7.pkl', 'rb'))
    for i in range(len(rating)):
        rating[i] = float(rating[i])
    rating = np.array(rating)
    p0 = np.eye(768)
    corrs = {}
    task_dim = 768
    corrs['ori'] = compute_pearson(rating, word_embed, p0, task_dim, 1)
    
    for i in tasks:
        corrs[i] = compute_pearson(rating, word_embed, projection_matrix[i], task_dim, 1)
    print(corrs)
    import pdb
    pdb.set_trace()
