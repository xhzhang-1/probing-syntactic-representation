from classifier import classifier
import numpy as np
import torch
from tqdm import tqdm
import pickle
from inlp_projection import rowspace_projection, intersection_projection

def NP_loop(yaml_args, cli_args, model, dataset, epochs, rounds, base):
    # first, compute weight W and loss/score
    # second, compute the nullspace P of weight W
    # third, project word embeddings to nullspace P, that is, P*E
    # return to step 1, repeat until the loss/score doesn't change anymore.
    P = np.eye(yaml_args['model'][cli_args.target_task]['hidden_dim'])

    weight_list = []
    min_acc = 100
    train_dataset, dev_dataset, test_dataset = dataset
    count = 0
    for round_index in tqdm(range(rounds), desc='[null_projection rounds]'):
        weight, train_loss, dev_loss, train_acc, dev_accuracy \
                 = model.train(train_dataset, dev_dataset, epochs, P)
        test_acc = model.evaluation(test_dataset, P)
        weight = (weight.detach().cpu().numpy()).T
        weight = rowspace_projection(weight)
        weight_list.append(weight)
        
        tqdm.write('[round {}] \nTrain loss: {}, Dev loss: {}, train accuracy: {}, Dev accuracy: {}'.format(
            round_index, train_loss, dev_loss, train_acc, dev_accuracy))
        tqdm.write('Test accuracy: {}'.format(test_acc))
        if test_acc < base:
            print('early stopping')
            break
        if test_acc > min_acc:
            count += 1
        else:
            min_acc = test_acc
            count = 0
        if count > 5:
            print('cannot drop to random level, early stopping')
            break
        P = intersection_projection(weight_list, weight.shape[1])
        
    _, train_loss, dev_loss, train_acc, dev_accuracy \
            = model.train(train_dataset, dev_dataset, epochs, P)
    test_acc = model.evaluation(test_dataset, P)
    tqdm.write('[round {}] Train loss: {}, Dev loss: {}, train accuracy: {}, Dev accuracy: {}'.format(
        round_index, train_loss, dev_loss, train_acc, dev_accuracy))
    tqdm.write('Test accuracy: {}'.format(test_acc))

    return P, weight_list, min_acc

def NP_loop_pair(yaml_args, cli_args, model, dataset, epochs, rounds, base):
    task_dim = yaml_args['model'][cli_args.target_task]['hidden_dim']
    P = np.eye(task_dim)
    half_dim = int(task_dim/2)
    weight_list = []
    min_acc = 100
    count = 0
    train_dataset, dev_dataset, test_dataset = dataset
    for round_index in tqdm(range(rounds), desc='[null_projection rounds]'):
        weights, train_loss, dev_loss, train_acc, dev_accuracy \
                = model.train(train_dataset, dev_dataset, epochs, P)
        test_acc = model.evaluation(test_dataset, P)
        weights = (weights.detach().cpu().numpy()).T
        weights1 = np.vstack([weights[:, 0:half_dim], weights[:, half_dim:]])
        weights1 = rowspace_projection(weights1)
        weight_list.append(weights1)
        tqdm.write('[round {}] \nTrain loss: {}, Dev loss: {}, train accuracy: {}, Dev accuracy: {}'.format(
            round_index, train_loss, dev_loss, train_acc, dev_accuracy))
        tqdm.write('Test accuracy: {}'.format(test_acc))
        if test_acc < base:
            print('early stopping')
            break
        if test_acc > min_acc:
            count += 1
        else:
            min_acc = test_acc
            count = 0
        if count > 5:
            print('cannot drop to random level, early stopping')
            break
        P1 = intersection_projection(weight_list, weights1.shape[1])
        P = np.eye(task_dim)
        P[0:half_dim, 0:half_dim] = P1
        P[half_dim:, half_dim:] = P1
    _, train_loss, dev_loss, train_acc, dev_accuracy \
            = model.train(train_dataset, dev_dataset, epochs, P)
    test_acc = model.evaluation(test_dataset, P)
    tqdm.write('[round {}] Train loss: {}, Dev loss: {}, train accuracy: {}, Dev accuracy: {}'.format(
        round_index, train_loss, dev_loss, train_acc, dev_accuracy))
    tqdm.write('Test accuracy: {}'.format(test_acc))
    return P, weight_list, min_acc

def inlp(args, yaml_args, dataset):
    target_model = classifier(yaml_args, args.target_task)
    if args.target_task == 'dep' or args.target_task == 'srl':
        P, weight_list = NP_loop_pair(yaml_args, args, target_model, dataset, args.inlp_epoch, args.inlp_iter, args.base)
    else:
        P, weight_list = NP_loop(yaml_args, args, target_model, dataset, args.inlp_epoch, args.inlp_iter, args.base)
    pickle.dump(P, open(yaml_args['reporting']['root']+"/projection_matrix.pkl", "wb"))
    pickle.dump(weight_list, open(yaml_args['reporting']['root']+"/weight_list.pkl", "wb"))

def mvnp(args, yaml_args, dataset):
    """
    code for MVNP method
    """
    res = []
    embeds = {}
    labels = []
    task = args.target_task
    task_dim = yaml_args['model'][task]['hidden_dim']
    label_num = yaml_args['model'][task]['class_number']
    for batch in dataset[0]:
        word_vecs, label_batch, length_batch, _ = batch
        for i in range(length_batch.shape[0]):
            for j in range(length_batch[i]):
                temp = label_batch[i][j].item()
                if temp in embeds.keys():
                    embeds[temp].append(word_vecs[i][j].cpu())
                else:
                    embeds[temp] = []
                    embeds[temp].append(word_vecs[i][j].cpu())
    for i in range(label_num):
        embeds[i] = torch.stack(embeds[i])
        embeds[i] = embeds[i].float()
        res.append(embeds[i].mean(0))
        labels.append(i)
    res = torch.stack(res)
    res = np.array(res)
    pickle.dump(res, open(yaml_args['reporting']['root']+task+'_means.pkl', 'wb'))
    print('labels of task '+task)
    print(labels)
    if task == 'dep' or task == 'srl':
        p1 = rowspace_projection(res[:,0:task_dim])
        p2 = rowspace_projection(res[:,task_dim:])
        P = intersection_projection([p1, p2], task_dim)
    else:
        P = intersection_projection([res], task_dim)
    pickle.dump(P, open(yaml_args['reporting']['root']+task+'_mean_P.pkl', 'wb'))
    return P