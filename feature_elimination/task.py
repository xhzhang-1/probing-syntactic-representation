"""
the code is adapted from https://github.com/john-hewitt/structural-probes
Contains classes describing linguistic tasks of interest on annotated data.
"""

import numpy as np
import torch

class Task:
  """Abstract class representing a linguistic task mapping texts to labels."""
  @staticmethod
  def labels(observation):
    """Maps an observation to a matrix of labels.
    
    Should be overriden in implementing classes.
    """
    raise NotImplementedError

class POSTask(Task):
  #pos = ['PUNCT', 'CONJ', 'AUX', 'X', 'PROPN', 'DET', 'ADP', 'SCONJ', 'PART',
  #          'ADJ', 'NOUN', 'NUM', 'SYM', 'INTJ', 'PRON', 'ADV', 'VERB']
  @staticmethod
  def labels(observation, pos, target_label=None):
    # pos is a dict
    sentence_length = len(observation[0])
    word_pos = observation[2]
    word_labels = torch.zeros(sentence_length)
    out = 0
    for i in range(sentence_length):
      p = word_pos[i]
      if p not in pos.keys():
        word_labels[i] = 21
        out += 1
      else:
        word_labels[i] = pos[p]
    return word_labels

class NERTask:
  """Maps observations to a depth in the parse tree for each word"""

  @staticmethod
  def labels(observation, tag, target_label=None):
    # pos is a list
    sentence_length = len(observation[0])
    word_ner = observation[6]
    #import pdb
    #pdb.set_trace()
    word_labels = torch.zeros(sentence_length)
    for i in range(sentence_length):
      if word_ner[i] not in tag.keys():
        import pdb
        pdb.set_trace()
      word_labels[i] = tag[word_ner[i]]
    #print("total labels: "+str(len(pos)))
    #print(pos)
    return word_labels

class SRLTask:
  """word pairs that have predicate-argument relationships"""

  @staticmethod
  def labels(observation, tag, target_label=None):
    predi_argu_nums = len(observation[8]) #All observation fields must be of same length
    srls = observation[8]
    wordpair_labels = torch.zeros(predi_argu_nums)
    for i in range(predi_argu_nums):
      if srls[i] not in tag.keys():
        import pdb
        pdb.set_trace()
      wordpair_labels[i] = tag[srls[i]]
    return wordpair_labels

class DEPTask:
  """word pairs that have head-dependency relationships"""

  @staticmethod
  def labels(observation, tag, target_label=None):
    head_dep_nums = len(observation[10]) #All observation fields must be of same length
    dep = observation[10]
    #import pdb
    #pdb.set_trace()
    wordpair_labels = torch.zeros(head_dep_nums)
    for i in range(head_dep_nums):
      if dep[i] not in tag.keys():
        import pdb
        pdb.set_trace()
      wordpair_labels[i] = tag[dep[i]]
    return wordpair_labels
