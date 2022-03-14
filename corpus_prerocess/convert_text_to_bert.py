'''
Takes raw text and saves BERT-cased features for that text to disk

Adapted from the BERT readme (and using the corresponding package) at

https://github.com/huggingface/pytorch-pretrained-BERT

###
John Hewitt, johnhew@stanford.edu
Feb 2019

'''
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer
from argparse import ArgumentParser
import h5py
import numpy as np
import os

argp = ArgumentParser()
'''argp.add_argument('input_path')
argp.add_argument('output_path')'''
argp.add_argument('bert_model', help='base or large')
args = argp.parse_args()

# Load pre-trained model tokenizer (vocabulary)
# Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
if args.bert_model == 'base':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')
    LAYER_COUNT = 12
    FEATURE_COUNT = 768
elif args.bert_model == 'large':
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = BertModel.from_pretrained('bert-large-cased')
    LAYER_COUNT = 24
    FEATURE_COUNT = 1024
else:
    raise ValueError("BERT model must be base or large")

model.eval()

def get_bert_embed(input_path, output_path, LAYER_COUNT, FEATURE_COUNT):
    with h5py.File(output_path, 'w') as fout:
        for index, line in enumerate(open(input_path)):
            '''if line[-2] != ' ':
                line = line[0:-2]
                line = line + ' .'''
            '''line = line.strip() # Remove trailing characters
            # add space between characters
            line_space = ''
            for i in line:
                line_space += i
                line_space += ' '
            line = '[CLS] ' + line_space + ' [SEP]' '''
            line = '[CLS] ' + line + ' [SEP]'
            #import pdb
            #pdb.set_trace()
            tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(line)
            if len(tokenized_text) >300:
                print(line)
                continue
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segment_ids = [1 for x in tokenized_text]
  
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segment_ids])
  
            with torch.no_grad():
                encoded_layers, _ = model(tokens_tensor, segments_tensors)
            dset = fout.create_dataset(str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT))
            dset[:,:,:] = np.vstack([np.array(x) for x in encoded_layers])

def convert_story_to_bert(file_folder):
    # discard the '.' in the end of the sentence, and add an all-zero vector in 
    files = os.listdir(file_folder+'story_script/')
    # exits = os.listdir('story_elmos')
    for i in files:
        print("computing "+i+" embeddings")
        input_path = file_folder + 'story_script/' + i
        output_path = file_folder + 'story_bert/' + i + ".bert-layers-7-12-768"
        get_bert_embed(input_path, output_path, 6, FEATURE_COUNT)

def rewrite(in_file, out_file):
    with open(in_file, 'r', encoding='utf-8') as rf:
        with open(out_file, 'w') as wf:
            for line in rf:
                if line[0] == '#':
                    sent = ''
                    continue
                line = line.strip().split('\t')
                if len(line)>1:
                    sent += line[1]
                    sent += ' '
                else:
                    wf.write(sent[0:-1]+'\n')

# text = ["test_text", "train_text", "development_text"]

'''text = ["test", "train", "development"]

for i in text:
    rf = open('data/ontonotes/bc/'+i+'_labels_old.txt', 'r', encoding='utf-8')
    wf = open('data/ontonotes/bc/'+i+'_labels.txt', 'w', encoding='utf-8')
    sent = ''
    for line in rf:
        if len(line) > 1:
            line = line.strip().split('\t')
            if len(line[1]) == 2:
                line[1] = line[1].strip('/')
            for i in line:
                wf.write(i+'\t')
            wf.write('\n')
        else:
            wf.write('\n')'''

'''text = ["test", "train", "development"]

for i in text:
    rf = open('data/ontonotes/bc/'+i+'_labels.txt', 'r', encoding='utf-8')
    wf = open('data/ontonotes/bc/'+i+'_text_new.txt', 'w', encoding='utf-8')
    sent = ''
    for line in rf:
        if len(line) > 1:
            line = line.strip().split('\t')
            if len(line[1]) == 2:
                line[1] = line[1].strip('/')
            sent += line[1]
            sent += ' '
        else:
            wf.write(sent[0:-1]+'\n')
            sent = '' '''
# text = ['en-ud-test', 'en-ud-train', 'en-ud-dev']
file_folder = 'data/encoding_data/'
# convert_story_to_bert(file_folder)
'''i = 'story_test.txt'
input_path = file_folder + 'story_script/' + i
output_path = file_folder + 'story_bert/' + i + ".bert-layers-7-12-768"
get_bert_embed(input_path, output_path, 6, 768)'''

text = ['en-ud-test', 'en-ud-train', 'en-ud-dev']
# text = ["test", "train", "development"]
for i in text:
    # input_path = 'data/ontonotes/bc/' + i + "_text_new.txt"
    # output_path = 'data/ontonotes/bc/' + i + ".bert-layers-1-12-768.hdf5"
    input_path = '../structural-probes/UD_English/' + i + "_new.txt"
    output_path = '../structural-probes/UD_English/' + i + ".bert-layers-1-12-768"
    get_bert_embed(input_path, output_path, 12, FEATURE_COUNT)

'''for i in text:
    input_path = args.input_path + i + "_new.txt"
    # out = args.input_path + i + "_new.txt"
    output_path = args.output_path + i + ".bert-layers-7-12-768"
    get_bert_embed(input_path, output_path, 6, FEATURE_COUNT)
    # rewrite(input_path, out)
    # get_bert_embed(input_path, output_path, LAYER_COUNT, FEATURE_COUNT)'''