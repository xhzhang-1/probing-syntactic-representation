from argparse import ArgumentParser
import yaml
from methods import inlp, mvnp
from utils import set_models, load_labels

if __name__ == "__main__":
    argp = ArgumentParser()
    argp.add_argument('experiment_config')
    argp.add_argument('--target_task', default='pos', help='task which related information would be nulled out')
    argp.add_argument('--method', default='mvnp', help='feature elimination method, mvnp or inlp')
    argp.add_argument('--label_path', default='data/ontonotes/', help='label path')
    argp.add_argument('--data_type', default='bert', help='elmo or bert')
    argp.add_argument('--inlp_epoch', default=30, help='training epochs when training classifiers in inlp method')
    argp.add_argument('--inlp_iter', default=50, help='iteration times in inlp method')
    
    args = argp.parse_args()
    yaml_args= yaml.load(open(args.experiment_config))

    tag_dict = load_labels(args.label_path)
    dataset = set_models(yaml_args, args.target_task, args.data_type, tag_dict)

    if args.method == 'inlp':
        inlp(args, yaml_args, dataset)
    elif args.method == 'mvnp':
        mvnp(args, yaml_args, dataset)