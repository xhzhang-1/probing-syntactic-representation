import pickle

def process_dep(in_file, dict_dep):
    f = open(in_file, 'r', encoding='utf-8')
    head_rel = [[]]
    for line in f:
        if not line.split():
            head_rel.append([])
            continue
        line = line.strip().split('\t')
        if len(line) < 5:
            continue
        head_rel[-1].append([int(line[6])-1, line[7]])
        if line[7] in dict_dep.keys():
            dict_dep[line[7]] += 1
        else:
            dict_dep[line[7]] = 1
    
    return head_rel[0:-1], dict_dep

files = ['train', 'dev', 'test']
root = '../structural-probes/UD_English/'
dict_dep = {}
for f in files:
    in_file = root+'en-ud-'+f+'.conllu'
    head_rel, dict_dep = process_dep(in_file, dict_dep)
    out_file = open('ud-english/'+f+'-dep.pkl', 'wb')
    pickle.dump(head_rel, out_file)
    #import pdb
    #pdb.set_trace()
print(dict_dep)
print(len(dict_dep.keys()))
pickle.dump(dict_dep, open('dep_labels.pkl', 'wb'))