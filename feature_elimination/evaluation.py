from classifier import classifier
import numpy as np
from scipy import stats
from tqdm import tqdm
from argparse import ArgumentParser
import yaml
import pickle
from utils import set_models, load_labels

# note: the significant test used here is t-test for the sample size is small (10)

def eval_on_tasks(model, epochs, P, task_dataset, task_dim):
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
        P = P2
    _, _, _, train_acc, dev_accuracy = \
        model.train(train_dataset, dev_dataset, epochs, P)
    test_acc = model.evaluation(test_dataset, P)
    return train_acc, dev_accuracy, test_acc

def run_eval_multi_times(test_time, args_dict, dataset, null_task, all_task, \
    projection_matrix, result_dict, epochs):
    task_dim = args_dict[null_task]['model'][null_task]['hidden_dim']
    for i in tqdm(range(test_time), desc='[test on '+null_task+']'):
        eval_model = classifier(args_dict[null_task], null_task)
        for j in all_task:
            eval_model = classifier(args_dict[null_task], null_task)
            res = eval_on_tasks(eval_model, epochs, projection_matrix[j], j, dataset, task_dim)
            result_dict[j].append(res)

def single_ttest(original, changed):
    if stats.levene(original, changed).pvalue > 0.01:
        p_val = stats.ttest_ind(original, changed)
    else:
        p_val = stats.ttest_ind(original, changed, equal_var=False)
    return p_val

def t_interval(data):
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
    argp.add_argument('config')
    argp.add_argument('config_dep')
    argp.add_argument('--data_type', default='bert', help='elmo or bert')
    argp.add_argument('--label_path', default='data/ontonotes/', help='label path')
    argp.add_argument('--epoch', default=30, help='number of epochs to train classifiers')
    argp.add_argument('--eval_times', default=10, help='number of times to run eval to compute confidence interval')
    argp.add_argument('--Projection_filepath', default='', help='Projection matrix path')
    argp.add_argument('--results_dir', default='', help='result dir')
    
    args = argp.parse_args()
    yaml_args = yaml.load(open(args.config))
    dep_yaml_args = yaml.load(open(args.config_dep))

    tasks = ['pos', 'ner', 'srl', 'dep']

    # computing precisions
    tag_dict = load_labels(args.label_path)
    projection_matrix = {}
    args_dict = {}
    for i in tasks:
        projection_matrix[i] = pickle.load(open(args.Projection_filepath+'null_'+i+'.pkl', 'rb'))
        args_dict[i] = yaml_args
    args_dict['dep'] = dep_yaml_args

    for single_task in tasks:
        result_dict = {}
        class_dict = {}
        result_dict['original'] = []
        class_dict['original'] = []
        for i in tasks:
            result_dict[i] = []
            class_dict[i] = []
        dataset = set_models(args_dict[single_task], single_task, args.data_type, tag_dict)
        run_eval_multi_times(1, args_dict, dataset, single_task, tasks, projection_matrix, result_dict, class_dict, epochs)
        pickle.dump(result_dict, open(args.results_dir+single_task+'_eval.pkl', 'wb'))
        
        print(result_dict)