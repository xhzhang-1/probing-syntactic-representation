from classifier import classifier
import numpy as np
from scipy.linalg import orth
from tqdm import tqdm
import data_ontonotes as data
from argparse import ArgumentParser
import task
import yaml
import pickle

def NIO_loop(yaml_args, cli_args, model, dataset, epochs, rounds, ith, base, method='bnlp'):
    # first, compute weight W and loss/score
    # second, compute the nullspace P of weight W
    # third, project word embeddings to nullspace P, that is, P*E
    # return to step 1, repeat until the loss/score doesn't change anymore.
    train_f = yaml_args['reporting']['root']+'/reporting_'+str(ith)+'.txt'
    trf = open(train_f, 'w')
    P = np.eye(yaml_args['model'][cli_args.target_task]['hidden_dim'])

    weight_list = []
    min_acc = 100
    train_dataset, dev_dataset, test_dataset = dataset
    count = 0
    for round_index in tqdm(range(rounds), desc='[null_projection rounds]'):
        # model = classifier(yaml_args, cli_args.target_task)
        weights, train_loss, dev_loss, train_acc, dev_accuracy, train_acc_all, \
                dev_acc_all, train_all_words, dev_all_words = model.train_until_convergence(train_dataset, dev_dataset, epochs, P)
        test_acc, test_acc_all, test_all_words = model.evaluation(test_dataset, P)
        weights = weights.detach().cpu().numpy()
        weights = weights.T
        weights = debias_weight(weights, method)
        weight_list.append(weights)
        
        tqdm.write('[round {}] \nTrain loss: {}, Dev loss: {}, train accuracy: {}, Dev accuracy: {}, \ntrain_acc_no_max: {}, dev_acc_no_max: {}'.format(
            round_index, train_loss, dev_loss, train_acc, dev_accuracy, train_acc_all, dev_acc_all))
        trf.write('[round {}] \nTrain loss: {}, Dev loss: {}, train accuracy: {}, Dev accuracy: {}, \ntrain_acc_no_max: {}, dev_acc_no_max: {}'.format(
            round_index, train_loss, dev_loss, train_acc, dev_accuracy, train_acc_all, dev_acc_all))
        trf.write("\n")
        tqdm.write('Test accuracy: {}, test_acc_all: {}'.format(test_acc, test_acc_all))
        trf.write('Test accuracy: {}, test_acc_all: {}'.format(test_acc, test_acc_all))
        trf.write("\n")
        if test_acc < base or test_acc_all < base:
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
        P = get_projection_to_intersection_of_nullspaces(weight_list, weights.shape[1])
        
    _, train_loss, dev_loss, train_acc, dev_accuracy, train_acc_all, \
            dev_acc_all, train_all_words, dev_all_words = model.train_until_convergence(train_dataset, dev_dataset, epochs, P)
    test_acc, test_acc_all, test_all_words = model.evaluation(test_dataset, P)
    tqdm.write('[round {}] Train loss: {}, Dev loss: {}, train accuracy: {}, Dev accuracy: {}, \ntrain_acc_no_max: {}, dev_acc_no_max: {}'.format(
        round_index, train_loss, dev_loss, train_acc, dev_accuracy, train_acc_all, dev_acc_all))
    trf.write('[round {}] Train loss: {}, Dev loss: {}, train accuracy: {}, Dev accuracy: {}, \ntrain_acc_no_max: {}, dev_acc_no_max: {}'.format(
        round_index, train_loss, dev_loss, train_acc, dev_accuracy, train_acc_all, dev_acc_all))
    trf.write("\n")
    tqdm.write('Test accuracy: {}, test_acc_all: {}'.format(test_acc, test_acc_all))
    trf.write('Test accuracy: {}, test_acc_all: {}'.format(test_acc, test_acc_all))
    trf.write("\n")
    tqdm.write('total labels for train: {}, total labels for dev: {}, total labels for test: {}'.format(train_all_words, dev_all_words, test_all_words))

    return P, weight_list, min_acc

def NIO_loop_pair(yaml_args, cli_args, model, t_dataset, epochs, rounds, ith, base, method='bnlp'):
    train_f = yaml_args['reporting']['root']+'/reporting_'+str(ith)+'.txt'
    trf = open(train_f, 'w')
    task_dim = yaml_args['model'][cli_args.target_task]['hidden_dim']
    P = np.eye(task_dim)
    half_dim = int(task_dim/2)
    weight_list = []
    min_acc = 100
    count = 0
    train_dataset, dev_dataset, test_dataset = t_dataset
    for round_index in tqdm(range(rounds), desc='[null_projection rounds]'):
        weights, train_loss, dev_loss, train_acc, dev_accuracy, train_acc_all, \
                dev_acc_all, train_all_words, dev_all_words = model.train_until_convergence(train_dataset, dev_dataset, epochs, P)
        test_acc, test_acc_all, test_all_words = model.evaluation(test_dataset, P)
        weights = weights.detach().cpu().numpy()
        weights = weights.T
        weights1 = np.vstack([weights[:, 0:half_dim], weights[:, half_dim:]])
        weights1 = debias_weight(weights1, method)
        weight_list.append(weights1)
        tqdm.write('[round {}] \nTrain loss: {}, Dev loss: {}, train accuracy: {}, Dev accuracy: {}, \ntrain_acc_no_max: {}, dev_acc_no_max: {}'.format(
            round_index, train_loss, dev_loss, train_acc, dev_accuracy, train_acc_all, dev_acc_all))
        trf.write('[round {}] \nTrain loss: {}, Dev loss: {}, train accuracy: {}, Dev accuracy: {}, \ntrain_acc_no_max: {}, dev_acc_no_max: {}'.format(
            round_index, train_loss, dev_loss, train_acc, dev_accuracy, train_acc_all, dev_acc_all))
        trf.write("\n")
        tqdm.write('Test accuracy: {}, test_acc_all: {}'.format(test_acc, test_acc_all))
        trf.write('Test accuracy: {}, test_acc_all: {}'.format(test_acc, test_acc_all))
        trf.write("\n")
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
        P1 = get_projection_to_intersection_of_nullspaces(weight_list, weights1.shape[1])
        P = np.eye(task_dim)
        P[0:half_dim, 0:half_dim] = P1
        P[half_dim:, half_dim:] = P1
    tqdm.write('total labels for train: {}, total labels for dev: {}, total labels for test: {}'.format(train_all_words, dev_all_words, test_all_words))
    return P, weight_list, min_acc

def debias_weight(weight, method):
    if method == 'bnlp':
        return weight
    elif method == 'inlp':
        return get_rowspace_projection(weight)
    else:
        print('invalid method! ')
        return 0

def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = orth(W.T) # orthogonal basis

    w_basis = w_basis * np.sign(w_basis[0][0]) # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace

    return P_W

def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices, input_dim: int):
    """ 
    the code of this function comes from
    """

    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis = 0)
    P = I - get_rowspace_projection(Q)
    print(np.linalg.matrix_rank(P))

    return P

def set_models(yaml_args, task_type, data_type, tag_dict):
    if task_type == 'ner':
        atask = task.NERTask
    elif task_type == 'pos':
        atask = task.POSTask
    elif task_type == 'srl':
        atask = task.SRLTask
    elif task_type == 'dep':
        atask = task.DEPTask
    elif task_type == 'freq':
        atask = task.FREQTask
    else:
        raise Exception('unknown task type!')
    if data_type == 'elmo':
        dataset = data.ELMoDataset
    elif data_type == 'bert':
        dataset = data.BERTDataset
    else:
        raise Exception('unknown data type!')
    
    expt_dataset = dataset(yaml_args, atask, task_type, tag_dict[task_type])
    clf = classifier(yaml_args, task_type)
    train_dataset = expt_dataset.get_train_dataloader()
    dev_dataset = expt_dataset.get_dev_dataloader()
    test_dataset = expt_dataset.get_test_dataloader()
    return clf, [train_dataset, dev_dataset, test_dataset]

if __name__ == "__main__":
    argp = ArgumentParser()
    argp.add_argument('experiment_config')
    argp.add_argument('--target_task', default='', help='task which related information would be nulled out')
    argp.add_argument('--data_type', default='', help='elmo or bert')
    argp.add_argument('--test_time', default=3, help='elmo or bert')
    argp.add_argument('--results-dir', default='',
      help='Set to reuse an old results dir; '
      'if left empty, new directory is created')
    
    cli_args = argp.parse_args()
    yaml_args= yaml.load(open(cli_args.experiment_config))

    srl = pickle.load(open('data/ontonotes/new_srl_labels.pkl', 'rb'))
    dep = pickle.load(open('data/ontonotes/new_dep_labels.pkl', 'rb'))
    pos = pickle.load(open('data/ontonotes/new_pos_labels.pkl', 'rb'))
    ner = pickle.load(open('data/ontonotes/new_ner_labels.pkl', 'rb'))
    tag_dict = {}
    tag_dict['pos'] = pos
    tag_dict['ner'] = ner
    tag_dict['srl'] = srl
    tag_dict['dep'] = dep

    target_model, t_dataset = set_models(yaml_args, cli_args.target_task, cli_args.data_type, tag_dict)

    acc_sum = 0
    base = 0.15
    method = 'inlp'
    for i in range(5, cli_args.test_time+5):
        target_model = classifier(yaml_args, cli_args.target_task)
        if cli_args.target_task == 'dep' or cli_args.target_task == 'srl':
            P, weight_list, min_acc = NIO_loop_pair(yaml_args, cli_args, target_model, t_dataset, 30, 50, i, base, method)
        else:
            P, weight_list, min_acc = NIO_loop(yaml_args, cli_args, target_model, t_dataset, 30, 50, i, base, method)
        pickle.dump(P, open(yaml_args['reporting']['root']+"/projection_matrix"+str(i)+".pkl", "wb"))
        pickle.dump(weight_list, open(yaml_args['reporting']['root']+"/weight_list"+str(i)+".pkl", "wb"))
        acc_sum += min_acc

    print("average test accuracy: "+str(acc_sum/cli_args.test_time))