from classifier import classifier
import numpy as np
from scipy.linalg import orth
from scipy import stats
import torch
from tqdm import tqdm
import data_ontonotes as data
from argparse import ArgumentParser
import task
import yaml
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

# note: the significant test used here is t-test for the sample size is small (10)

def eval_on_tasks(model, epochs, P, task_type, task_dataset, task_dim):
    #losses = []
    # task_type is the task that projection matrix comes from
    P1 = np.eye(task_dim)
    train_dataset = task_dataset[0]
    dev_dataset = task_dataset[1]
    test_dataset = task_dataset[2]
    if task_dim == P.shape[0]*2:
        P1[0:P.shape[0], 0:P.shape[0]] = P
        P1[P.shape[0]:, P.shape[0]:] = P
        P = P1
    elif task_dim*2 == P.shape[0]:
        P1 = P[0:task_dim, 0:task_dim]
        P2 = P[task_dim:, task_dim:]
        # P = np.matmul(P1, P2)
        P = P2
    '''if task_type == 'dep' or task_type == 'srl':
        # P[0:1024, 0:1024] = P[1024:, 1024:]
        P[1024:, 1024:] = P[0:1024, 0:1024]'''
    _, _, _, train_acc, dev_accuracy, train_f1, dev_f1, _, _ = \
        model.train_until_convergence(train_dataset, dev_dataset, epochs, P)
    test_acc, test_f1, _ = model.evaluation(test_dataset, P)
    # print(task_type+'----train_acc: '+str(train_acc)+'\t----dev_acc: '+str(dev_accuracy)+"\ttest_acc: "+str(test_acc))
    # print('train_acc_no_max: {}, dev_acc_no_max: {}, test_acc_no_max: {}'.format(train_acc_no_max, dev_acc_no_max, test_acc_no_max))
    '''f.write(task_type+'----train_acc: '+str(train_acc)+'\t----dev_acc: '+str(dev_accuracy)+"\ttest_acc: "+str(test_acc))
    f.write('train_acc_no_max: {}, dev_acc_no_max: {}, test_acc_no_max: {}'.format(train_acc_no_max, dev_acc_no_max, test_acc_no_max))
    f.write('\n')'''
    return train_acc, dev_accuracy, test_acc, train_f1, dev_f1, test_f1

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
    elif data_type == 'glove':
        dataset = data.GloveDataset      
    else:
        raise Exception('unknown data type!')
    ### to be continued
    expt_dataset = dataset(yaml_args, atask, task_type, tag_dict[task_type])
    train_dataset = expt_dataset.get_train_dataloader()
    dev_dataset = expt_dataset.get_dev_dataloader()
    test_dataset = expt_dataset.get_test_dataloader()
    return [train_dataset, dev_dataset, test_dataset]

def run_eval_multi_times(test_time, args_dict, dataset, null_task, all_task, \
    projection_matrix, result_dict, class_dict, epochs):
    # yaml_args is a dict
    task_dim = args_dict[null_task]['model'][null_task]['hidden_dim']
    P0 = np.eye(task_dim)
    for i in tqdm(range(test_time), desc='[test on '+null_task+']'):
        eval_model = classifier(args_dict[null_task], null_task)
        '''res = eval_on_tasks(eval_model, epochs, P0, null_task, dataset, task_dim)
        result_dict['original'].append(res[0:3])
        class_dict['original'].append(res[3:])'''
        
        for j in all_task[:1]:
            eval_model = classifier(args_dict[null_task], null_task)
            res = eval_on_tasks(eval_model, epochs, projection_matrix[j], j, dataset, task_dim)
            result_dict[j].append(res[0:3])
            class_dict[j].append(res[3:])

def run_eval_with_mean(test_time, args_dict, dataset, null_task, \
    proj_path, result_dict, class_dict):
    # yaml_args is a dict
    task_dim = args_dict[null_task]['model'][null_task]['hidden_dim']
    P0 = np.eye(task_dim)
    for i in tqdm(range(test_time), desc='[test on '+null_task+']'):
        eval_model = classifier(args_dict[null_task], null_task)
        eval_model.load_model(proj_path)
        res = eval_model.evaluation(dataset[2], P0)
        # res = eval_on_tasks(eval_model, epochs, P0, null_task, dataset, task_dim)
        result_dict['original'].append(res[0:3])
        class_dict['original'].append(res[3:])
    return res

def single_ttest(original, changed):
    if stats.levene(original, changed).pvalue > 0.01:
        p_val = stats.ttest_ind(original, changed)
    else:
        p_val = stats.ttest_ind(original, changed, equal_var=False)
    return p_val

def t_interval(data):
    #import pdb
    #pdb.set_trace()
    data = np.array(data)
    mean=data.mean()
    std=data.std()
    interval=stats.t.interval(0.95,data.shape[0]-1,mean,std)
    return interval

def run_ttest(result_dict, tasks, position):
    original = []
    p_vals = []
    intervals = []
    changed = {}
    for i in result_dict['original']:
        original.append(i[position])
    intervals.append(t_interval(original))
    for i in tasks:
        changed[i] = []
        for j in result_dict[i]:
            changed[i].append(j[position])
    for i in tasks:
        intervals.append(t_interval(changed[i]))
        p_vals.append(single_ttest(original, changed[i]))
    return p_vals, intervals


if __name__ == "__main__":
    argp = ArgumentParser()
    argp.add_argument('experiment_config1')
    argp.add_argument('experiment_config2')
    argp.add_argument('--data_type', default='', help='elmo or bert')
    argp.add_argument('--Projection_filepath', default='', help='Projection matrix that need to be loaded')
    argp.add_argument('--results_dir', default='',
      help='Set to reuse an old results dir; '
      'if left empty, new directory is created')
    
    cli_args = argp.parse_args()
    # cli_args.Projection_filepath = 'results_all/null_Ps/'
    yaml_args = yaml.load(open(cli_args.experiment_config1))
    dep_yaml_args = yaml.load(open(cli_args.experiment_config2))

    tasks = ['pos', 'ner', 'srl', 'dep']

    # computing precisions
    tag_dict = {}
    projection_matrix = {}
    args_dict = {}
    for i in tasks:
        # tag_dict[i] = pickle.load(open('data/ontonotes/new_'+i+'_labels.pkl', 'rb'))
        tag_dict[i] = pickle.load(open('data_zh/annotation_corpus/'+i+'_labels.pkl', 'rb'))
        projection_matrix[i] = pickle.load(open(cli_args.Projection_filepath+'null_'+i+'.pkl', 'rb'))
        # projection_matrix[i] = pickle.load(open(cli_args.Projection_filepath+'null_'+i+'.pkl', 'rb'))
        # proj = pickle.load(open(cli_args.Projection_filepath+i+'_mean_P.pkl', 'rb'))
        # projection_matrix[i] = np.eye(proj.shape[0]) - proj
        # projection_matrix[i] = proj
        args_dict[i] = yaml_args
    # projection_matrix['pos'] = pickle.load(open('results/results_bert/null_pos/null_pos_7_i/projection_matrix5.pkl', 'rb'))
    args_dict['dep'] = dep_yaml_args
    # print(projection_matrix)

    # projection_matrix['freq'] = pickle.load(open(cli_args.Projection_filepath+'null_freq.pkl', 'rb'))
    '''for single_task in tasks:
        proj_path = cli_args.Projection_filepath+single_task+'_means.pkl'
        dataset = set_models(args_dict[single_task], single_task, cli_args.data_type, tag_dict)
        result_dict = {}
        class_dict = {}
        result_dict['original'] = []
        class_dict['original'] = []
        run_eval_with_mean(1, args_dict, dataset, single_task, proj_path, result_dict, class_dict)
        print(result_dict)
        print(class_dict)'''

    # all_Ps = pickle.load(open('results/results_bert/null_srl/null_srl_7_i/projection_matrix5.pkl', 'rb'))
    epochs = 30
    for single_task in tasks:
        result_dict = {}
        class_dict = {}
        result_dict['original'] = []
        class_dict['original'] = []
        for i in tasks:
            result_dict[i] = []
            class_dict[i] = []
        dataset = set_models(args_dict[single_task], single_task, cli_args.data_type, tag_dict)

        # run_eval_multi_times(10, args_dict, dataset, single_task, tasks, projection_matrix, result_dict, class_dict, epochs)
        run_eval_multi_times(1, args_dict, dataset, single_task, tasks, projection_matrix, result_dict, class_dict, epochs)
        # pickle.dump(result_dict, open('results_ontonotes/null_tests_final/'+single_task+'_test.pkl', 'wb'))
        pickle.dump(result_dict, open(cli_args.results_dir+single_task+'_test.pkl', 'wb'))
        pickle.dump(class_dict, open(cli_args.results_dir+single_task+'_test_classes.pkl', 'wb'))
        # p_vals = run_ttest(result_dict, tasks)
        # print(p_vals)
        if single_task == 'ner':
            print(class_dict)
        else:
            print(result_dict)
        # print(class_dict)
        # pickle.dump(p_vals, open('results_ontonotes/null_tests_final/'+single_task+'_pvals.pkl', 'wb'))
        # pickle.dump(p_vals, open('results_all/null_tests/'+single_task+'_pvals.pkl', 'wb'))
    '''inter = {}
    position = 2
    for i in tasks:
        result_dict = pickle.load(open(cli_args.results_dir+i+'_test.pkl', 'rb'))
        if i == 'ner':
            result_dict = pickle.load(open(cli_args.results_dir+i+'_test_classes.pkl', 'rb'))
        p_vals, intervals = run_ttest(result_dict, tasks, position)
        inter[i] = intervals
    pickle.dump(inter, open(cli_args.results_dir+'t_intervals.pkl', 'wb'))
    for i in tasks:
        print(i)
        for j in inter[i]:
            print(str((j[1]+j[0])/2)+', '+str((j[1]-j[0])/2))'''
    
