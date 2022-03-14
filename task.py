"""Contains classes describing linguistic tasks of interest on annotated data."""

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

class SyntaxTask(Task):
  """Maps observations to dependency parse distances between words."""
  #pos = ['PUNCT', 'CONJ', 'AUX', 'X', 'PROPN', 'DET', 'ADP', 'SCONJ', 'PART',
  #          'ADJ', 'NOUN', 'NUM', 'SYM', 'INTJ', 'PRON', 'ADV', 'VERB']
  @staticmethod
  def labels(observation):
    sentence_length = len(observation[0]) #All observation fields must be of same length
    word_pos = observation[3]
    #import pdb
    #pdb.set_trace()
    word_labels = torch.zeros(sentence_length)
    pos = []
    for i in range(sentence_length):
      if word_pos[i] in pos:
        word_labels[i] = pos.index(word_pos[i])
      else:
        pos.append(i)
        word_labels[i] = len(pos) - 1
    print("total labels: "+str(len(pos)))
    print(pos)
    return word_labels
  
  @staticmethod
  def depths(observation):
    sentence_length = len(observation[0]) #All observation fields must be of same length
    depths = torch.zeros(sentence_length)
    for i in range(sentence_length):
      depths[i] = SyntaxTask.get_ordering_index(observation, i)
    return depths

  @staticmethod
  def get_ordering_index(observation, i, head_indices=None):
    '''Computes tree depth for a single word in a sentence

    Args:
      observation: an Observation namedtuple, with a head_indices field.
          or None, if head_indies != None
      i: the word in the sentence to compute the depth of
      head_indices: the head indices (according to a dependency parse) of all
          words, or None, if observation != None.

    Returns:
      The integer depth in the tree of word i
    '''
    if observation:
      head_indices = []
      number_of_underscores = 0
      for elt in observation.head_indices:
        if elt == '_':
          head_indices.append(0)
          number_of_underscores += 1
        else:
          head_indices.append(int(elt) + number_of_underscores)
    length = 0
    i_head = i+1
    while True:
      i_head = head_indices[i_head - 1]
      if i_head != 0:
        length += 1
      else:
        return length

class POSTask(Task):
  """Maps observations to dependency parse distances between words."""
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

class binaryPOSTask(Task):
  """Maps observations to dependency parse distances between words."""
  #pos = ['PUNCT', 'CONJ', 'AUX', 'X', 'PROPN', 'DET', 'ADP', 'SCONJ', 'PART',
  #          'ADJ', 'NOUN', 'NUM', 'SYM', 'INTJ', 'PRON', 'ADV', 'VERB']
  @staticmethod
  def labels(observation, pos, aim_label):
    # pos is a dict
    sentence_length = len(observation[0])
    word_pos = observation[2]
    word_labels = torch.zeros(sentence_length)
    for i in range(sentence_length):
      p = word_pos[i]
      if p == aim_label:
        word_labels[i] = 1
      else:
        word_labels[i] = 0
      '''if p not in pos.keys():
        word_labels[i] = 21
        out += 1
      else:
        word_labels[i] = pos[p]'''
    return word_labels

class FREQTask:
  """Maps observations to a depth in the parse tree for each word"""

  @staticmethod
  def labels(observation, tag):
    # pos is a list
    sentence_length = len(observation[0])
    word_freq = observation[7]
    #import pdb
    #pdb.set_trace()
    word_labels = torch.zeros(sentence_length)
    for i in range(sentence_length):
      #import pdb
      #pdb.set_trace()
      word_labels[i] = int(word_freq[i])
    #print("total labels: "+str(len(pos)))
    #print(pos)
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

  @staticmethod
  def get_ordering_index(observation, i, head_indices=None):
    '''Computes tree depth for a single word in a sentence

    Args:
      observation: an Observation namedtuple, with a head_indices field.
          or None, if head_indies != None
      i: the word in the sentence to compute the depth of
      head_indices: the head indices (according to a dependency parse) of all
          words, or None, if observation != None.

    Returns:
      The integer depth in the tree of word i
    '''
    if observation:
      head_indices = []
      number_of_underscores = 0
      for elt in observation.head_indices:
        if elt == '_':
          head_indices.append(0)
          number_of_underscores += 1
        else:
          head_indices.append(int(elt) + number_of_underscores)
    length = 0
    i_head = i+1
    while True:
      i_head = head_indices[i_head - 1]
      if i_head != 0:
        length += 1
      else:
        return length

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

  @staticmethod
  def get_ordering_index(observation, i, head_indices=None):
    '''Computes tree depth for a single word in a sentence

    Args:
      observation: an Observation namedtuple, with a head_indices field.
          or None, if head_indies != None
      i: the word in the sentence to compute the depth of
      head_indices: the head indices (according to a dependency parse) of all
          words, or None, if observation != None.

    Returns:
      The integer depth in the tree of word i
    '''
    if observation:
      head_indices = []
      number_of_underscores = 0
      for elt in observation.head_indices:
        if elt == '_':
          head_indices.append(0)
          number_of_underscores += 1
        else:
          head_indices.append(int(elt) + number_of_underscores)
    length = 0
    i_head = i+1
    while True:
      i_head = head_indices[i_head - 1]
      if i_head != 0:
        length += 1
      else:
        return length

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
