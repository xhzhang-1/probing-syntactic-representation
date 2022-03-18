"""
the code is adapted from https://github.com/john-hewitt/structural-probes
This module handles the reading of [text, label, srls] files and hdf5 embeddings.

Specifies Dataset classes, which offer PyTorch Dataloaders for the
train/dev/test splits.
"""
import os
from collections import namedtuple, defaultdict

from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import h5py
import pickle

Debug = False
class SimpleDataset:
  def __init__(self, args, task, model_type, all_labels, target_label=None, vocab={}):
    self.args = args
    self.batch_size = args['dataset']['batch_size']
    self.use_disk_embeddings = args['model']['use_disk']
    self.vocab = vocab
    self.task = task
    self.model_type = model_type
    if Debug:
      import pdb
      pdb.set_trace()
    self.observation_class = self.get_observation_class(self.args['dataset']['observation_fieldnames'])
    self.train_obs, self.dev_obs, self.test_obs = self.read_from_disk()
    self.train_dataset = ObservationIterator(self.train_obs, task, all_labels, target_label)
    self.dev_dataset = ObservationIterator(self.dev_obs, task, all_labels, target_label)
    self.test_dataset = ObservationIterator(self.test_obs, task, all_labels, target_label)

  def read_from_disk(self):
    train_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['train_path'])
    dev_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['dev_path'])
    test_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['test_path'])
    train_observations = self.load_conll_dataset(train_corpus_path)
    dev_observations = self.load_conll_dataset(dev_corpus_path)
    test_observations = self.load_conll_dataset(test_corpus_path)

    train_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['train_path'])
    dev_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['dev_path'])
    test_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['test_path'])
    train_observations = self.optionally_add_embeddings(train_observations, train_embeddings_path, train_corpus_path)
    dev_observations = self.optionally_add_embeddings(dev_observations, dev_embeddings_path, dev_corpus_path)
    test_observations = self.optionally_add_embeddings(test_observations, test_embeddings_path, test_corpus_path)
    return train_observations, dev_observations, test_observations

  def get_observation_class(self, fieldnames):
    return namedtuple('Observation', fieldnames)

  def generate_lines_for_sent(self, lines):
    buf = []
    for line in lines:
      if line.startswith('#'):
        continue
      if not line.strip():
        if buf:
          yield buf
          buf = []
        else:
          continue
      else:
        buf.append(line.strip())
    if buf:
      yield buf

  def load_conll_dataset(self, filepath):
    observations = []
    if self.model_type == 'dep':
      lines = (x for x in open(filepath))
    else:
      lines = (x for x in open(filepath+'labels.txt'))
    for buf in self.generate_lines_for_sent(lines):
      conll_lines = []
      for line in buf:
        conll_lines.append(line.strip().split('\t'))
      embeddings = [None for x in range(len(conll_lines))]
      # if self.model_type == 'srl': pairwise means predicate-argument pair;
      # elif self.model_type == 'dep': pairwise means head-dependent pair.
      pairwise_rels = []
      # import pdb
      # pdb.set_trace()
      pairwise_rels = [None for x in range(len(conll_lines))]
      try:
        observation = self.observation_class(*zip(*conll_lines), pairwise_rels, embeddings)
      except:
        import pdb
        pdb.set_trace()
      observations.append(observation)
    return observations

  def add_embeddings_to_observations(self, observations, embeddings, concats, labels):
    embedded_observations = []
    if concats != None:
      p = 0
      for observation, label, embed in zip(observations, labels, embeddings):
        c_embed = np.zeros([len(label), embed.shape[1]*2], dtype='float32')
        for i in range(len(label)):
          c_embed[i] = np.append(embed[concats[p][i][0]], embed[concats[p][i][1]])
        embedded_observation = self.observation_class(*(observation[:-2]), label, c_embed)
        embedded_observations.append(embedded_observation)
        p += 1
    else:
      for observation, embedding in zip(observations, embeddings):
        embedded_observation = self.observation_class(*(observation[:-1]), embedding)
        embedded_observations.append(embedded_observation)
    return embedded_observations

  def generate_token_embeddings_from_hdf5(self, args, observations, filepath, layer_index=None):
    if Debug:
      import pdb
      pdb.set_trace()
    hf = h5py.File(filepath, 'r') 
    indices = filter(lambda x: x != 'sentence_to_index', list(hf.keys()))
    single_layer_features_list = []
    for index in sorted([int(x) for x in indices]):
      observation = observations[index]
      feature_stack = hf[str(index)]
      if layer_index != None:
        single_layer_features = feature_stack[layer_index]
      else:
        single_layer_features = feature_stack[:]

      if single_layer_features.shape[0] != len(observation.sentence):
        import pdb
        pdb.set_trace()
      # assert single_layer_features.shape[0] == len(observation.sentence)
      single_layer_features_list.append(single_layer_features)
    return single_layer_features_list

  def get_train_dataloader(self, shuffle=True):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=shuffle)

  def get_dev_dataloader(self):
    return DataLoader(self.dev_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

  def get_test_dataloader(self, use_embeddings=True):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

  def optionally_add_embeddings(self, observations):
    return observations

  def get_srl_concats(self, filepath, sent_num):
    srls = open(filepath+'srls_less.txt', 'r')
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
  
  def get_dep_concats(self, filepath, sent_num):
    if 'train' in filepath:
      deps = pickle.load(open('data/ud-english/train-dep.pkl', 'rb'))
    elif 'test' in filepath:
      deps = pickle.load(open('data/ud-english/test-dep.pkl', 'rb'))
    if 'dev' in filepath:
      deps = pickle.load(open('data/ud-english/dev-dep.pkl', 'rb'))
    #import pdb
    #pdb.set_trace()
    assert(len(deps) == sent_num)
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

  def custom_pad(self, batch_observations):
    if self.use_disk_embeddings:
      seqs = [torch.tensor(x[0].embeddings, device=self.args['device']) for x in batch_observations]
    else:
      seqs = [torch.tensor(x[0].sentence, device=self.args['device']) for x in batch_observations]
    lengths = torch.tensor([len(x) for x in seqs], device=self.args['device'])
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    label_shape = batch_observations[0][1].shape
    maxlen = int(max(lengths))
    label_maxshape = [maxlen for x in label_shape]
    labels = [-torch.ones(*label_maxshape, device=self.args['device']) for x in seqs]
    for index, x in enumerate(batch_observations):
      length = x[1].shape[0]
      if len(label_shape) == 1:
        labels[index][:length] = x[1]
      elif len(label_shape) == 2:
        labels[index][:length,:length] = x[1]
      else:
        raise ValueError("Labels must be either 1D or 2D right now; got either 0D or >3D")
    labels = torch.stack(labels)
    return seqs, labels, lengths, batch_observations

class ELMoDataset(SimpleDataset):
  
  def optionally_add_embeddings(self, observations, pretrained_embeddings_path, corpus_path):
    layer_index = self.args['model'][self.model_type]['model_layer']
    print('Loading ELMo Pretrained Embeddings from {}; using layer {}'.format(pretrained_embeddings_path, layer_index))
    embeddings = self.generate_token_embeddings_from_hdf5(self.args, observations, pretrained_embeddings_path, layer_index)
    concats = None
    labels = None
    if self.model_type == 'srl':
      # labels are predicate_arguments, embeddings are predicate-argument pair embeddings.
      sent_num = len(observations)
      labels, concats = self.get_srl_concats(corpus_path, sent_num)
      ########
    elif self.model_type == 'dep':
      sent_num = len(observations)
      labels, concats = self.get_dep_concats(corpus_path, sent_num)
    observations = self.add_embeddings_to_observations(observations, embeddings, concats, labels)
    return observations

class SubwordDataset(SimpleDataset):
  @staticmethod
  def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1
    while (untokenized_sent_index < len(untokenized_sent) and
        tokenized_sent_index < len(tokenized_sent)):
      while (tokenized_sent_index + 1 < len(tokenized_sent) and
          tokenized_sent[tokenized_sent_index + 1].startswith('##')):
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        tokenized_sent_index += 1
      mapping[untokenized_sent_index].append(tokenized_sent_index)
      untokenized_sent_index += 1
      tokenized_sent_index += 1
    return mapping

  def generate_subword_embeddings_from_hdf5(self, observations, filepath, bert_layer, subword_tokenizer=None):
    raise NotImplementedError("Instead of making a SubwordDataset, make one of the implementing classes")

class BERTDataset(SubwordDataset):
  def generate_subword_embeddings_from_hdf5(self, observations, filepath, bert_layer, subword_tokenizer=None):
    
    if subword_tokenizer == None:
      try:
        from pytorch_pretrained_bert import BertTokenizer
        if self.args['model'][self.model_type]['hidden_dim'] == 768 or self.args['model'][self.model_type]['hidden_dim'] == 1536:
          subword_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
          print('Using BERT-base-cased tokenizer to align embeddings with PTB tokens')
        elif self.args['model'][self.model_type]['hidden_dim'] == 1024:
          subword_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
          print('Using BERT-large-cased tokenizer to align embeddings with PTB tokens')
        else:
          print("The heuristic used to choose BERT tokenizers has failed...")
          exit()
      except:
        print('Couldn\'t import pytorch-pretrained-bert. Exiting...')
        exit()
    hf = h5py.File(filepath, 'r')
    indices = list(hf.keys())
    single_layer_features_list = []
    for index in tqdm(sorted([int(x) for x in indices]), desc='[aligning embeddings]'):
      observation = observations[index]
      feature_stack = hf[str(index)]
      # single_layer_features = np.mean(feature_stack[bert_layer], 0)
      single_layer_features = feature_stack[bert_layer]
      tokenized_sent = subword_tokenizer.wordpiece_tokenizer.tokenize('[CLS] ' + ' '.join(observation.sentence) + ' [SEP]')
      untokenized_sent = observation.sentence
      untok_tok_mapping = self.match_tokenized_to_untokenized(tokenized_sent, untokenized_sent)
      try:
        assert single_layer_features.shape[0] == len(tokenized_sent)
        single_layer_features = torch.tensor([np.mean(single_layer_features[untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], axis=0) for i in range(len(untokenized_sent))])
        assert single_layer_features.shape[0] == len(observation.sentence)
      except:
        import pdb
        pdb.set_trace()
      single_layer_features_list.append(single_layer_features)
    return single_layer_features_list

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path, corpus_path):
    """Adds pre-computed BERT embeddings from disk to Observations."""
    layer_index = self.args['model'][self.model_type]['model_layer']
    print('Loading BERT Pretrained Embeddings from {}; using layer {}'.format(pretrained_embeddings_path, layer_index))
    embeddings = self.generate_subword_embeddings_from_hdf5(observations, pretrained_embeddings_path, layer_index)
    concats = None
    labels = None
    if self.model_type == 'srl':
      # labels are predicate_arguments, embeddings are predicate-argument pair embeddings.
      sent_num = len(observations)
      labels, concats = self.get_srl_concats(corpus_path, sent_num)
      ########
    elif self.model_type == 'dep':
      sent_num = len(observations)
      labels, concats = self.get_dep_concats(corpus_path, sent_num)
    observations = self.add_embeddings_to_observations(observations, embeddings, concats, labels)
    return observations

class ObservationIterator(Dataset):
  def __init__(self, observations, task, all_labels, target_label=None):
    self.observations = observations
    self.set_labels(observations, task, all_labels, target_label)

  def set_labels(self, observations, task, all_labels, target_label=None):
    self.labels = []
    for observation in tqdm(observations, desc='[computing labels]'):
      self.labels.append(task.labels(observation, all_labels, target_label))

  def __len__(self):
    return len(self.observations)

  def __getitem__(self, idx):
    return self.observations[idx], self.labels[idx]

