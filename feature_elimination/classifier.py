"""
the code is adapted from https://github.com/john-hewitt/structural-probes
"""
import torch
import torch.nn as nn
import sys
from torch import optim
import os
import pickle

class classifier(nn.Module):
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

  def forward(self, batch):
    transformed = torch.matmul(batch, self.proj)
    return transformed, self.proj
  
  def set_optimizer(self):
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
    total_sents = torch.sum(length_batch != 0).float()
    labels_1s = (label_batch != -1).float()
    N, M, C = predictions.shape
    predictions_masked = predictions * labels_1s.view(N, M, 1).expand(N, M, C)
    
    labels_masked = label_batch * labels_1s
    loss_per_sent = torch.zeros(N).cuda()
    if total_sents > 0:
      for i in range(int(total_sents)):
        loss_per_sent[i] = self.loss_fn(predictions_masked[i], labels_masked[i].long())
      batch_loss = torch.sum(loss_per_sent) / total_sents
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents

  def train(self, train_dataset, dev_dataset, epochs, P):
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

        corrects = pred.long().eq(label_batch.long())
        correct += corrects.sum().item()
        total_words += length_batch.sum().item()
        self.optimizer.step()
      train_acc = correct / float(total_words)
      
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
        total_words += length_batch.sum()
      
      dev_acc = correct / float(total_words)
      self.scheduler.step(epoch_dev_loss)

      if epoch_dev_loss / epoch_dev_loss_count < min_dev_loss - 0.0001:
        torch.save(self.state_dict(), self.params_path)
        min_dev_loss = epoch_dev_loss / epoch_dev_loss_count
        min_dev_loss_epoch = epoch_index
      elif min_dev_loss_epoch < epoch_index - 4:
        break
    return weights, epoch_train_loss/epoch_train_loss_count, epoch_dev_loss/epoch_dev_loss_count, \
      train_acc, dev_acc

  def evaluation(self, test_dataset, P):
    P = torch.FloatTensor(P).cuda()
    epoch_test_loss = 0
    correct = total_words = 0
    pred_other = 0
    for batch in test_dataset:
      #self.optimizer.zero_grad()
      self.eval()
      word_representations, label_batch, length_batch, _ = batch
      word_representations = torch.matmul(word_representations, P)
      predictions, _ = self.forward(word_representations, label_batch)
      batch_loss, count = self.loss(predictions, label_batch, length_batch)
      epoch_test_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
      #epoch_test_epoch_count += 1
      pred = predictions.data.max(2)[1]
      pred_other += ((pred.eq(0)).sum().item()-(label_batch.eq(-1)).sum().item())
      corrects = pred.long().eq(label_batch.long())
      correct += corrects.sum().item()
      total_words += length_batch.sum()
    accuracy = correct / float(total_words)
    return accuracy

  def load_model(self, param_path):
    p_matrix = torch.tensor(pickle.load(open(param_path, 'rb'))).cuda().transpose(0,1)
    for i in range(p_matrix.shape[1]):
      p_matrix[:,i] /= torch.norm(p_matrix[:,i])
    self.proj = torch.nn.Parameter(p_matrix)
