import os
import h5py
import pickle
import scipy.io as scio
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder

def generate_corpus(file_folder, save_path, ners, srls):
    files = os.listdir(file_folder)
    sf = open(save_path+'_labels.txt', 'a', encoding='utf-8')
    textf = open(save_path+'_text.txt', 'a', encoding='utf-8')
    srlf = open(save_path+'_srls.txt', 'a', encoding='utf-8')
    for file in files:
        root2 = file_folder+'/'+file
        suffexs = os.listdir(root2)
        for suffex in suffexs:
            root = root2+'/'+suffex
            if not os.path.exists(root):
                continue
            gfs = os.listdir(root)
            
            for gf in gfs:
                if gf[-5:] == 'conll':
                    f = open(root+'/'+gf, 'r', encoding='utf-8')
                    sents = ""
                    sout = out = True
                    for line in f:
                        line = line.strip().split()
                        if len(line) < 6:
                            if len(line) == 0:
                                if len(sents) > 2 and sents[-3] == '/':
                                    sents = sents[0:-3]+sents[-2:]
                                textf.write(sents[0:-1]+'\n')
                                sents = ""
                                srlf.write('\n')
                                sf.write('\n')
                            continue
                        sents += line[3]
                        sents += " "
                        if out:
                            if line[10] == '*':
                                ner = '*'
                                ners['*'] += 1
                            elif line[10][0] == '(':
                                if line[10][-1] != ')':
                                    out = False
                                ner = line[10][1:-1]
                                if ner in ners.keys():
                                    ners[ner] += 1
                                else:
                                    ners[ner] = 1
                        else:
                            if line[10][-1] == ')':
                                out = True
                        sf.write(line[2]+'\t'+line[3]+'\t'+line[4]+'\t'+line[6]+'\t'+line[7]+'\t'+line[8]+'\t'+ner+'\n')
                        for i in line[11:-1]:
                            if sout:
                                if i == '*':
                                    srl = '*'
                                    srls['*'] += 1
                                elif i[0] == '(':
                                    end = -2
                                    if i[-1] != ')':
                                        end = -1
                                        sout = False
                                    srl = i[1:end]
                                    if srl in srls.keys():
                                        srls[srl] += 1
                                    else:
                                        srls[srl] = 1
                            else:
                                if i[-1] == ')':
                                    sout = True
                            srlf.write(srl+'\t')
                        srlf.write('\n')

def delete_slash(old_file, new_file):
    with open(old_file, 'r') as rf:
        with open(new_file, 'w') as wf:
            for line in rf:
                new_line = line
                if len(line) > 2:
                    if line[-3] == '/':
                        new_line = line[0:-3]+line[-2:]
                wf.write(new_line)
    wf.close()
    rf.close()

def convert_text_to_elmo(text_path):
    options_file = "../elmo_files/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "../elmo_files/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    files = ['development','test','train']
    elmo = ElmoEmbedder(options_file, weight_file)
    for i in files:
        print("computing "+i+" embeddings")
        hf = h5py.File(text_path+i+".elmo-layers.hdf5", "w")
        with open(text_path+i+"_text.txt", 'r') as tf:
            i = 0
            for line in tf:
                line = line.strip('\n').split(' ')
                elmo_embed = elmo.embed_sentence(line)
                dset = hf.create_dataset(str(i), elmo_embed.shape, dtype='f4')
                dset[...] = elmo_embed
                i += 1
        tf.close()
        hf.close()

def convert_h5py_to_mat(filepath, layer_index):
    hf = h5py.File(filepath, 'r') 
    indices = filter(lambda x: x != 'sentence_to_index', list(hf.keys()))
    single_layer_features_list = []
    valid = []
    for index in sorted([int(x) for x in indices]):
        feature_stack = hf[str(index)]
        single_layer_features = feature_stack[layer_index]
        for i in single_layer_features:
            single_layer_features_list.append(i)
            valid.append(1)
        single_layer_features_list.append(np.zeros(1024))
        valid.append(0)
    
    return np.array(single_layer_features_list[0:-1]), np.array(valid[0:-1])

def convert_story_to_elmo(file_folder):
    # discard the '.' in the end of the sentence, and add an all-zero vector in 
    # convert_h5py_to_mat()
    options_file = "../elmo_files/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "../elmo_files/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    # file_folder = 'story_script'
    files = os.listdir(file_folder)
    exits = os.listdir('story_elmos')
    elmo = ElmoEmbedder(options_file, weight_file)
    for i in files:
        print("computing "+i+" embeddings")
        if "story_elmos/"+i[0:-4]+".elmo-layers.hdf5" in exits:
            continue
        hf = h5py.File("story_elmos/"+i[0:-4]+".elmo-layers.hdf5", "w")
        with open("story_script/"+i, 'r') as tf:
            temp = 0
            for line in tf:
                if line[-2] == ' ':
                    line = line[0:-2]
                line = line.strip('\n').split()
                elmo_embed = elmo.embed_sentence(line)
                dset = hf.create_dataset(str(temp), elmo_embed.shape, dtype='f4')
                dset[...] = elmo_embed
                temp += 1
        tf.close()
        hf.close()

def convert_ontonotes_to_elmo(root):
    from allennlp.commands.elmo import ElmoEmbedder
    options_file = "../elmo_files/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "../elmo_files/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    files = ['development','test','train']
    elmo = ElmoEmbedder(options_file, weight_file)
    for i in files:
        print("computing "+i+" embeddings")
        hf = h5py.File("ontonotes_new/"+root+"/"+i+".elmo-layers.hdf5", "w")
        with open("ontonotes_new/"+root+"/"+i+"_text.txt", 'r') as tf:
            i = 0
            for line in tf:
                line = line.strip('\n').split(' ')
                elmo_embed = elmo.embed_sentence(line)
                dset = hf.create_dataset(str(i), elmo_embed.shape, dtype='f4')
                dset[...] = elmo_embed
                i += 1
        tf.close()
        hf.close()

def convert_ontonotes_to_glove(root, embed_file, out):
    all_embeds = open(embed_file, 'r', encoding='utf-8')
    word_dict = {}
    embeds = []
    count = 0
    for line in all_embeds:
        line = line.strip().split()
        word_dict[line[0]] = count
        embeds.append([float(line[i]) for i in range(1, 301)])
        count += 1
    files = ['development','test','train']
    for f in files:
        print("computing "+f+" embeddings")
        hf = h5py.File(root+f+".glove.hdf5", "w")
        with open(root+f+"_text.txt", 'r') as tf:
            i = 0
            for line in tf:
                line = line.strip('\n').split(' ')
                sent_embed = []
                for word in line:
                    word = word.lower()
                    if word in word_dict.keys():
                        sent_embed.append(embeds[word_dict[word]])
                    else:
                        if word in out.keys():
                            out[word] += 1
                        else:
                            out[word] = 1
                        temp = [0 for i in range(300)]
                        for char in word:
                            temp += embeds[word_dict[char]]
                        
                        temp = [temp[k]/len(word) for k in range(300)]
                        sent_embed.append(temp)
                assert(len(sent_embed) == len(line))
                sent_embed = np.array(sent_embed)
                dset = hf.create_dataset(str(i), sent_embed.shape, dtype='f4')
                dset[...] = sent_embed
                i += 1
        tf.close()
        hf.close()
    return out

def convert_udenglish_to_glove(embed_file, out):
    root = '../structural-probes/UD_English/'
    all_embeds = open(embed_file, 'r', encoding='utf-8')
    word_dict = {}
    embeds = []
    count = 0
    for line in all_embeds:
        line = line.strip().split()
        word_dict[line[0]] = count
        embeds.append([float(line[i]) for i in range(1, 301)])
        count += 1
    files = ['en-ud-dev','en-ud-test','en-ud-train']
    for f in files:
        print("computing "+f+" embeddings")
        hf = h5py.File(root+f+".glove.hdf5", "w")
        with open(root+f+".conllu", 'r') as tf:
            i = 0
            for line in tf:
                if line[0] == '#':
                    sent_embed = []
                    continue
                if len(line) < 2:
                    sent_embed = np.array(sent_embed)
                    dset = hf.create_dataset(str(i), sent_embed.shape, dtype='f4')
                    dset[...] = sent_embed
                    i += 1
                    continue
                line = line.strip('\n').split('\t')
                word = line[1].lower()
                if word in word_dict.keys():
                    sent_embed.append(embeds[word_dict[word]])
                elif line[2].lower() in word_dict.keys():
                    sent_embed.append(embeds[word_dict[line[2].lower()]])
                else:
                    if word in out.keys():
                        out[word] += 1
                    else:
                        out[word] = 1
                    temp = [0 for i in range(300)]
                    for char in word:
                        temp += embeds[word_dict[char]]
                        
                    temp = [temp[k]/len(word) for k in range(300)]
                    sent_embed.append(temp)
                
        tf.close()
        hf.close()
    return out

def convert_story_to_glove(root, out_root, files, embed_file, out):
    all_embeds = open(embed_file, 'r', encoding='utf-8')
    word_dict = {}
    embeds = []
    count = 0
    out_number = 0
    for line in all_embeds:
        line = line.strip().split()
        word_dict[line[0]] = count
        embeds.append([float(line[i]) for i in range(1, 301)])
        count += 1
    import pdb
    pdb.set_trace()
    for in_f in files:
        story_embed = []
        print("computing "+in_f+" embeddings")
        words = (scio.loadmat(root+in_f))['word']
        for word in words:
            word = word.strip()
            word = word.lower()
            if word in word_dict.keys():
                story_embed.append(embeds[word_dict[word]])
            else:
                out_number += 1
                if word in out.keys():
                    out[word] += 1
                else:
                    out[word] = 1
                temp = oov_embedding(word, word_dict, embeds)
                story_embed.append(temp)
        story_embed = np.array(story_embed)
        print(story_embed.shape)
        print(out_number)
        scio.savemat(out_root+in_f[0:-4]+'_glove.mat', {'data':story_embed})

def oov_embedding(word, word_dict, embeds):
    # 1. if "'" in word, then check if n't in word;
    # 2. if n't in word, split word into xxx+n't, find xxx and n't in word_dict;
    # 3. if n't not in word, split word into xxx+'yy, find xxx and 'yy in word_dict;
    # 4. if ' not in word, add all character embeddings together.
    word = word.split("'")
    temp = [0 for i in range(300)]
    sub_num = 0
    for subword in word:
        if subword in word_dict.keys():
            temp += embeds[word_dict[subword]]
            sub_num += 1
        else:
            for char in subword:
                temp += embeds[word_dict[char]]
                sub_num += 1
    temp = [temp[k]/sub_num for k in range(300)]
    return temp

def count_samples(file_path, p):
    # count sample numbers of pos, ner, wf
    res = 0
    label_dict = {}
    pos_dict = {}
    with open(file_path, 'r') as rf:
        for line in rf:
            line = line.strip('\n').split('\t')
            if len(line) > 1:
                res += 1
                if line[p] in label_dict.keys():
                    label_dict[line[p]] += 1
                else:
                    label_dict[line[p]] = 1
                if line[p] != '*':
                    if line[2] in pos_dict.keys():
                        pos_dict[line[2]] += 1
                    else:
                        pos_dict[line[2]] = 1

                #if line[2] == 'JJ' and line[p] == 'LOC':
                #    print(line)
    return res, label_dict, pos_dict

def add_freq(file_path, label_dict):
    out = {}
    with open(file_path+'_old.txt', 'r') as rf:
        with open(file_path+'.txt', 'w') as wf:
            for line in rf:
                n_line = line.strip().split('\t')
                if len(n_line) > 1:
                    if n_line[1] in label_dict.keys():
                        freq = label_dict[n_line[1]]
                    elif n_line[1] in out.keys():
                        out[n_line[1]] += 1
                        freq = 4
                    else:
                        out[n_line[1]] = 1
                        freq = 4
                    if freq>10000:
                        freq = 0
                    elif freq>1000:
                        freq = 1
                    elif freq>100:
                        freq = 2
                    elif freq>10:
                        freq = 3
                    else:
                        freq = 4
                    wf.write(line[0:-1]+'\t'+str(freq)+'\n')
                else:
                    wf.write('\n')
    rf.close()
    wf.close()
    return out

def count_srls(file_path, sent_num):
    srls = open(file_path, 'r')
    labels = [[] for i in range(sent_num)]
    label_dict = {}
    label_dict['*'] = 0
    label_dict['_'] = 0
    p = 0
    res = 0
    discard = ['ARGM-REC', 'ARGM-GOL', 'ARGM-DSP', 'ARGM-PRR', 'ARGM-COM', 'ARGM-PRX', 'ARGA', 'ARGM-LVB']
    for line in srls:
        if line == '\n':
            p += 1
            '''if p == sent_num:
                break'''
            continue
        line = line.strip().split('\t')
      
        if 'V' in line:
            stars = []
            for i in range(len(line)):
                if line[i] != '*' and line[i] != 'V':
                    if line[i] not in discard:
                        res += 1
                        labels[p].append(line[i])
                        if line[i] in label_dict.keys():
                            label_dict[line[i]] += 1
                        else:
                            label_dict[line[i]] = 1
                elif line[i] == '*':
                    stars.append(i)
            if len(stars) == len(line)-1:
                for i in range(2):
                    if i > len(stars)-1:
                        break
                    labels[p].append('*')
                    label_dict['*'] += 1
                    res += 1

        else:
            for i in range(len(line)-1):
                labels[p].append('_')
                label_dict['_'] += 1
                res += 1
            labels[p].append('_')
            label_dict['_'] += 1
            res += 1

    return res, label_dict

def count_deps(file_path):
    deps = pickle.load(open(file_path, 'rb'))
    label_dict = {}
    res = 0
    label_dict['none'] = 0
    discard = ['iobj', 'det:predet', 'csubj', 'remnant', 'csubjpass', 'vocative', 'dislocated', 'cc:preconj', 'reparandum', 'foreign', 'dep', 'goeswith']
    for i in deps:
        for j in i:
            if j[1] in discard:
                continue
            if j[0] != -1:
                res += 1
                if j[1] in label_dict.keys():
                    label_dict[j[1]] += 1
                else:
                    label_dict[j[1]] = 1
                if j[1] == 'root':
                    import pdb
                    pdb.set_trace()
            else:
                label_dict['none'] += 1
                res += 1
        
    return res, label_dict

def combine_corpus(rfiles, wfile):
    with open(wfile, 'w') as wf:
        for f in rfiles:
            with open(f, 'r') as rf:
                for line in rf:
                    wf.write(line)
            rf.close()

def combine_labels(rfiles, wfile, p=None):
    dicts = [{}, {}, {}]
    # p = [2, 6, 7]
    with open(wfile, 'w') as wf:
        for f in rfiles:
            with open(f, 'r') as rf:
                for line in rf:
                    wf.write(line)
                    line = line.strip().split('\t')
                    for i in p:
                        if line[i] in dicts[i].keys():
                            dicts[i][line[i]] += 1
                        else:
                            dicts[i][line[i]] = 1
            rf.close()
    return dicts

def combine_srls(rfiles, wfile):
    with open(wfile, 'w') as wf:
        for f in rfiles:
            with open(f, 'r') as rf:
                for line in rf:
                    wf.write(line)
                    line = line.strip().split('\t')

            rf.close()

def delete_irrevelant_srl(r_file, w_file, pf_file, remains):
    # do not label unimportant words in argument phrases
    pf = open(pf_file, 'r')
    count = 0
    pos = [[]]
    for line in pf:
        if line == '\n':
            pos.append([])
        else:
            line = line.strip().split('\t')
            pos[-1].append(line[2])
    with open(w_file, 'w') as wf:
        with open(r_file, 'r') as rf:
            for line in rf:
                if line == '\n':
                    count += 1
                    wf.write('\n')
                else:
                    new_line = ''
                    line = line.strip().split('\t')

                    assert(len(line) == len(pos[count]))
                    for i in range(len(line)):
                        if pos[count][i] not in remains:
                            new_line += '*'
                            new_line += '\t'
                        else:
                            new_line += line[i]
                            new_line += '\t'
                    wf.write(new_line[0:-1]+'\n')


if __name__ == "__main__":

    # convert_story_to_elmo('story_script')
    '''root = 'E:/brain/fmri_related/task_stimuli/feature/'
    out_root = 'D:/null_space_backup/null_space/data_encoding/story_embeds/story_glove/'
    files = []
    for i in range(1, 52):
        files.append('story_%02d_time_features.mat'%i)
    embed_file = 'D:/datasets/glove_embedding/glove.6B.300d.txt'
    out = {}
    convert_story_to_glove(root, out_root, files, embed_file, out)'''
    embed_file = 'data/glove.6B.300d.txt'
    out = {}
    convert_udenglish_to_glove(embed_file, out)
    import pdb
    pdb.set_trace()

    '''file_type = ['train', 'test', 'development']
    remains = ['_', 'WP', 'WRB', 'PRP', 'PRP$', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBZ', 'VBP', 'VBG', 'VBN', 'VBD', 'JJ', 'JJR', 'JJS']
    for f in file_type:
        pf_file = 'ontonotes/bc/'+f+'_labels.txt'
        # r_file = 'ontonotes/bc/'+f+'_srls_processed.txt'
        # w_file = 'ontonotes/bc/'+f+'_srls_less.txt'
        # delete_irrevelant_srl(r_file, w_file, pf_file, remains)
        res, label_dict, pos_dict = count_samples(pf_file, 6)
        import pdb
        pdb.set_trace()'''
    '''freq_dict = pickle.load(open('ontonotes/word_fredict.pkl', 'rb'))
    file_type = ['train', 'test', 'development']
    for f in file_type:
        file_path = 'ontonotes_new/all/'+f+'_labels'
        out = add_freq(file_path, freq_dict)
        import pdb
        pdb.set_trace()'''
    '''file_type = ['train', 'test', 'development']
    folders = ['bc', 'bn', 'nw', 'tc']
    path = "conll-formatted-ontonotes-5.0/data/" # file folder name
    ners = {}
    ners['*'] = 0
    srls = {}
    srls['*'] = 0
    pos = {}
    for i in file_type:
        for j in folders:
            file_folder = path + i + '/data/english/annotations/' + j
            # output_file = 'ontonotes_new/' + j + '/' + i
            output_file = 'ontonotes_new/all/' + i
            generate_corpus(file_folder, output_file, ners, srls)'''
    # old_file_root = 'ontonotes/bc/'
    # new_file_root = 'ontonotes/bc_new/'
    '''for i in file_type:
        old_file = old_file_root + i + '_text.txt'
        new_file = new_file_root + i + '_text.txt'
        delete_slash(old_file, new_file)'''

    # convert_text_to_elmo(new_file_root)
    '''for f in file_type:
        file_path = 'ontonotes/bc/'+f+'_srls_processed.txt'
        res = count_srls(file_path, 20000)
        print(f+': '+str(res))'''
    '''file_type = ['train', 'test', 'development']
    for f in file_type:
        file_path = 'ontonotes_new/bc/'+f+'_labels.txt'
        res = count_samples(file_path, 6) # pos=2,ner=6, freq=7
        print(f+': '+str(res))'''
    '''for f in file_type:
        file_path = 'ud-english/' + f + '-dep.pkl'
        res = count_deps(file_path)
        print(f+': '+str(res))'''
    import pdb
    pdb.set_trace()
