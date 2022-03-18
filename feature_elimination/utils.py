import data_ontonotes as data
import task
import pickle

def set_models(yaml_args, task_type, data_type, tag_dict):
    if task_type == 'ner':
        atask = task.NERTask
    elif task_type == 'pos':
        atask = task.POSTask
    elif task_type == 'srl':
        atask = task.SRLTask
    elif task_type == 'dep':
        atask = task.DEPTask
    else:
        raise Exception('unknown task type!')
    if data_type == 'elmo':
        dataset = data.ELMoDataset
    elif data_type == 'bert':
        dataset = data.BERTDataset
    else:
        raise Exception('unknown data type!')
    expt_dataset = dataset(yaml_args, atask, task_type, tag_dict[task_type])
    train_dataset = expt_dataset.get_train_dataloader()
    dev_dataset = expt_dataset.get_dev_dataloader()
    test_dataset = expt_dataset.get_test_dataloader()
    return [train_dataset, dev_dataset, test_dataset]

def load_labels(label_root):
    srl = pickle.load(open(label_root+'srl_labels.pkl', 'rb'))
    dep = pickle.load(open(label_root+'dep_labels.pkl', 'rb'))
    pos = pickle.load(open(label_root+'pos_labels.pkl', 'rb'))
    ner = pickle.load(open(label_root+'ner_labels.pkl', 'rb'))
    tag_dict = {}
    tag_dict['pos'] = pos
    tag_dict['ner'] = ner
    tag_dict['srl'] = srl
    tag_dict['dep'] = dep
    return tag_dict

