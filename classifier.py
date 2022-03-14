import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from torch import optim
import os
import pickle

class classifier(nn.Module):
  """ Computes squared L2 norm of words after projection by a matrix."""

  def __init__(self, args, task_name):
    #print('Constructing OneWordPSDProbe')
    super(classifier, self).__init__()
    self.args = args
    self.class_num = args['model'][task_name]['class_number']
    self.model_dim = args['model'][task_name]['hidden_dim']
    self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.class_num))
    self.loss_fn = nn.CrossEntropyLoss()
    nn.init.uniform_(self.proj, -0.05, 0.05)
    self.params_path = os.path.join(args['reporting']['root'], args['reporting']['params_path'])
    self.to(args['device'])

  def forward(self, batch, tags):
    transformed = torch.matmul(batch, self.proj)
    # batchlen, seqlen, rank = transformed.size()
    return transformed, self.proj
  
  def set_optimizer(self, probe):
    """Sets the optimizer and scheduler for the training regimen.
  
    Args:
      probe: the probe PyTorch model the optimizer should act on.
    """

    self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,patience=5)
  
  def loss(self, predictions, label_batch, length_batch):
    """ Computes crossentropy loss on sequences.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted labels
      label_batch: A pytorch batch of true labels
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    # import pdb
    # pdb.set_trace()
    total_sents = torch.sum(length_batch != 0).float()
    labels_1s = (label_batch != -1).float()
    N, M, C = predictions.shape
    predictions_masked = predictions * labels_1s.view(N, M, 1).expand(N, M, C)
    # transpose = torch.zeros()
    # predictions_transposed = torch.matmul(predictions, transpose)
    labels_masked = label_batch * labels_1s
    loss_per_sent = torch.zeros(N).cuda()
    if total_sents > 0:
      for i in range(int(total_sents)):
        # loss_per_sent[i] = self.loss_fn(predictions_masked[i], labels_masked[i].long()) * M
        loss_per_sent[i] = self.loss_fn(predictions_masked[i], labels_masked[i].long())
      # loss_per_sent is a vector
      # normalized_loss_per_sent = loss_per_sent / length_batch.float()
      # batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
      batch_loss = torch.sum(loss_per_sent) / total_sents
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents

  def train_until_convergence(self, train_dataset, dev_dataset, epochs, P):
    """ Trains a probe until a convergence criterion is met.

    Trains until loss on the development set does not improve by more than epsilon
    for 5 straight epochs.

    Writes parameters of the probe to disk, at the location specified by config.

    Args:
      loss: An instance of loss.Loss, computing loss between predictions and labels
      train_dataset: a torch.DataLoader object for iterating through training data
      dev_dataset: a torch.DataLoader object for iterating through dev data
    """
    self.set_optimizer(self.proj)
    min_dev_loss = sys.maxsize
    min_dev_loss_epoch = -1
    # import pdb
    # pdb.set_trace()
    P = torch.FloatTensor(P).cuda()
    for epoch_index in range(epochs):
      epoch_train_loss = 0
      epoch_dev_loss = 0
      epoch_train_loss_count = 0
      epoch_dev_loss_count = 0
      train_class_words = torch.zeros(self.class_num)
      train_class_correct = torch.zeros(self.class_num)
      dev_class_words = torch.zeros(self.class_num)
      dev_class_correct = torch.zeros(self.class_num)
      correct = 0
      total_words = 0
      pred_other = 0
      # import pdb
      # pdb.set_trace()
      for batch in train_dataset:
        self.train()
        self.optimizer.zero_grad()
        word_representations, label_batch, length_batch, _ = batch
        word_representations = torch.matmul(word_representations, P)
        predictions, weights = self.forward(word_representations, label_batch)
        batch_loss, count = self.loss(predictions, label_batch, length_batch)
        batch_loss.backward()
        epoch_train_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
        epoch_train_loss_count += count.detach().cpu().numpy()
        pred = predictions.data.max(2)[1]
        pred_other += ((pred.eq(0)).sum().item()-(label_batch.eq(-1)).sum().item())
        # import pdb
        # pdb.set_trace()
        corrects = pred.long().eq(label_batch.long())
        correct += corrects.sum().item()
        for i in range(self.class_num):
          train_class_words[i] += label_batch.eq(i).sum().item()
          train_class_correct[i] += (label_batch.eq(i).int()+corrects.int()).eq(2).sum().item()
        # +2 means this predicted label equals to the true label and the max label.
        '''mid_tensor1 = torch.zeros(label_batch.shape).long().cuda()+2
          mid_tensor2 = pred.long().eq(label_batch.long()).long() + pred.long().eq(max_labels.long()).long()
          max_correct += mid_tensor1.eq(mid_tensor2).sum().item()
          max_words += label_batch.long().eq(max_labels).sum().item()'''
        
        #max_correct += 
        total_words += length_batch.sum().item()
        self.optimizer.step()
      train_acc = correct / float(total_words)
      recall = (correct-train_class_correct[0].item())/float(sum(train_class_words[1:]))
      ne_acc = (correct-train_class_correct[0].item())/float(total_words-pred_other)
      train_acc_all = train_class_correct/train_class_words.float()
      
      correct = 0
      total_words = 0
      pred_other = 0
      for batch in dev_dataset:
        self.optimizer.zero_grad()
        self.eval()
        word_representations, label_batch, length_batch, _ = batch
        word_representations = torch.matmul(word_representations, P)
        predictions, weights = self.forward(word_representations, label_batch)
        batch_loss, count = self.loss(predictions, label_batch, length_batch)
        epoch_dev_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
        epoch_dev_loss_count += count.detach().cpu().numpy()
        pred = predictions.data.max(2)[1]
        pred_other += ((pred.eq(0)).sum().item()-(label_batch.eq(-1)).sum().item())
        corrects = pred.long().eq(label_batch.long())
        correct += corrects.sum().item()
        for i in range(self.class_num):
          dev_class_words[i] += label_batch.eq(i).sum().item()
          dev_class_correct[i] += (label_batch.eq(i).int()+corrects.int()).eq(2).sum().item()
        total_words += length_batch.sum()
      
      dev_acc = correct / float(total_words)
      dev_acc_all = dev_class_correct/dev_class_words.float()
      dev_recall = (correct-dev_class_correct[0].item())/float(sum(dev_class_words[1:]))
      dev_ne_acc = (correct-dev_class_correct[0].item())/float(total_words-pred_other)
      self.scheduler.step(epoch_dev_loss)

      if epoch_dev_loss / epoch_dev_loss_count < min_dev_loss - 0.0001:
        torch.save(self.state_dict(), self.params_path)
        min_dev_loss = epoch_dev_loss / epoch_dev_loss_count
        min_dev_loss_epoch = epoch_index
      elif min_dev_loss_epoch < epoch_index - 4:
        break
    # return weights, epoch_train_loss/epoch_train_loss_count, epoch_dev_loss/epoch_dev_loss_count, \
    #   train_acc, dev_acc, train_acc_all, dev_acc_all, train_class_words, dev_class_words
    return weights, epoch_train_loss/epoch_train_loss_count, epoch_dev_loss/epoch_dev_loss_count, \
      train_acc, dev_acc, recall, dev_recall, train_acc_all, dev_acc_all


  def evaluation(self, test_dataset, P):
    P = torch.FloatTensor(P).cuda()
    epoch_test_loss = 0
    correct = total_words = 0
    pred_other = 0
    test_class_words = torch.zeros(self.class_num)
    test_class_correct = torch.zeros(self.class_num)
    for batch in test_dataset:
      #self.optimizer.zero_grad()
      self.eval()
      word_representations, label_batch, length_batch, _ = batch
      word_representations = torch.matmul(word_representations, P)
      predictions, weights = self.forward(word_representations, label_batch)
      batch_loss, count = self.loss(predictions, label_batch, length_batch)
      epoch_test_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
      #epoch_test_epoch_count += 1
      pred = predictions.data.max(2)[1]
      pred_other += ((pred.eq(0)).sum().item()-(label_batch.eq(-1)).sum().item())
      corrects = pred.long().eq(label_batch.long())
      correct += corrects.sum().item()
      total_words += length_batch.sum()
      for i in range(self.class_num):
        test_class_words[i] += label_batch.eq(i).sum().item()
        test_class_correct[i] += (label_batch.eq(i).int()+corrects.int()).eq(2).sum().item()
    accuracy = correct / float(total_words)
    test_acc_all = test_class_correct/test_class_words.float()
    test_recall = (correct-test_class_correct[0].item())/float(sum(test_class_words[1:]))
    test_ne_acc = (correct-test_class_correct[0].item())/float(total_words-pred_other)
    # return accuracy, test_acc_all, test_class_words
    return accuracy, test_recall, test_acc_all

  def load_model(self, param_path):
    p_matrix = torch.tensor(pickle.load(open(param_path, 'rb'))).cuda().transpose(0,1)
    for i in range(p_matrix.shape[1]):
      p_matrix[:,i] /= torch.norm(p_matrix[:,i])
    self.proj = torch.nn.Parameter(p_matrix)
