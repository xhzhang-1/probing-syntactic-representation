from collections import namedtuple, defaultdict
import h5py
from tqdm import tqdm
import torch
import numpy as np
import scipy.io as scio

def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1
    while (untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(tokenized_sent)):
        while (tokenized_sent_index + 1 < len(tokenized_sent) and tokenized_sent[tokenized_sent_index + 1].startswith('##')):
            mapping[untokenized_sent_index].append(tokenized_sent_index)
            tokenized_sent_index += 1
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        untokenized_sent_index += 1
        tokenized_sent_index += 1
    return mapping

def get_untok_lines(token_file):
    f = open(token_file, 'r', encoding='utf-8')
    untok_lines = [[]]
    for line in f:
        if len(line)>1:
            line = line.strip().split('\t')
            untok_lines[-1].append(line[1])
        else:
            untok_lines.append([])
    if untok_lines[-1] == []:
        untok_lines = untok_lines[0:-1]
    return untok_lines

def read_hdf5(filepath, bert_layer, script, token_file, subword_tokenizer=None):
    if subword_tokenizer == None:
        from pytorch_pretrained_bert import BertTokenizer
        subword_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        print('Using BERT-base-cased tokenizer to align embeddings with PTB tokens')
    hf = h5py.File(filepath, 'r')
    indices = list(hf.keys())
    single_layer_features_list = []
    f = open(script, 'r', encoding='utf-8')
    lines = []
    for line in f:
        if line[-2] != ' ':
            line = line[0:-2]
            line = line+' .' 
        lines.append(line)
    
    untok_lines = get_untok_lines(token_file)
    for index in tqdm(sorted([int(x) for x in indices]), desc='[aligning embeddings]'):
      # observation = observations[index]
      sent = lines[index]
      feature_stack = hf[str(index)]
      single_layer_features = feature_stack[bert_layer]
      tokenized_sent = subword_tokenizer.wordpiece_tokenizer.tokenize('[CLS] ' + sent + ' [SEP]')
      untokenized_sent = untok_lines[index]
      # tokenized_sent = subword_tokenizer.wordpiece_tokenizer.tokenize('[CLS] ' + ' '.join(observation.sentence) + ' [SEP]')
      # untokenized_sent = observation.sentence
      untok_tok_mapping = match_tokenized_to_untokenized(tokenized_sent, untokenized_sent)
      assert single_layer_features.shape[0] == len(tokenized_sent)
      single_layer_features = torch.tensor([np.mean(single_layer_features[untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], axis=0) for i in range(len(untokenized_sent))])
      assert single_layer_features.shape[0] == len(untokenized_sent)
      for i in single_layer_features:
          single_layer_features_list.append(np.array(i))
    return single_layer_features_list

'''root = 'data/encoding_data/'
in_file = []
scripts = []
import pdb
pdb.set_trace()
from pytorch_pretrained_bert import BertTokenizer
subword_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
print('Using BERT-base-cased tokenizer to align embeddings with PTB tokens')
for i in range(1, 2):
    script = root + 'story_script/story_test.txt'
    token_file = root + 'story_script/story_test_out.txt'
    filepath = root + 'story_bert/story_test.txt.bert-layers-7-12-768'
    embeds = read_hdf5(filepath, 0, script, token_file, subword_tokenizer)
    embeds = np.stack(embeds)
    scio.savemat(root+'story_bert_matched/story_test_layer7.mat', {'data':embeds})
    script = root + 'story_script/story_%02d.txt'%i
    token_file = root + 'story_script/story_%02d_out.txt'%i
    filepath = root + 'story_bert/story_%02d.txt.bert-layers-7-12-768'%i
    embeds = read_hdf5(filepath, 0, script, token_file, subword_tokenizer)
    embeds = np.stack(embeds)
    scio.savemat(root+'story_bert_matched/story_%02d_layer7.mat'%i, {'data':embeds})'''