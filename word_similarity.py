import csv
import scipy.io as scio
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_csv(file_path, pair_dict, word_dict, scale):
    with open(file_path, 'r') as rf:
        reader = csv.reader(rf)
        for line in reader:
            if len(line) != 4:
                continue
            if line[0] == '':
                continue
            pair = line[1]+' '+line[2]
            if pair in pair_dict.keys():
                # pair_dict[pair] = (pair_dict[pair]+float(line[3])*scale)/2
                continue
            else:
                pair_dict[pair] = float(line[3])*scale
                if float(line[3])*scale>10:
                    import pdb
                    pdb.set_trace()
            if line[1] not in word_dict.keys():
                word_dict[line[1]] = 0
            if line[2] not in word_dict.keys():
                word_dict[line[2]] = 0

def load_simverb(file_path, pair_dict, word_dict, scale):
    with open(file_path, 'r') as rf:
        reader = csv.reader(rf)
        count = 0
        for line in reader:
            count += 1
            if len(line) != 5:
                continue
            if line[0] == '':
                continue
            pair = line[2]+' '+line[3]
            if pair in pair_dict.keys():
                # pair_dict[pair] = (pair_dict[pair]+float(line[1])*scale)/2
                continue
            else:
                if float(line[1])*scale>10:
                    import pdb
                    pdb.set_trace()
                pair_dict[pair] = float(line[1])*scale
            if line[2] not in word_dict.keys():
                word_dict[line[2]] = 0
            if line[3] not in word_dict.keys():
                word_dict[line[3]] = 0
        print(count)

def load_men(file_path, pair_dict, word_dict, scale):
    with open(file_path, 'r') as rf:
        reader = csv.reader(rf)
        count = 0
        for line in reader:
            count += 1
            if len(line) != 4:
                continue
            if line[0] == '':
                continue
            pair = line[1][0:-2]+' '+line[2][0:-2]
            if pair in pair_dict.keys():
                # pair_dict[pair] = (pair_dict[pair]+float(line[3])*scale)/2
                continue
            else:
                pair_dict[pair] = float(line[3])*scale
                if float(line[3])*scale>10:
                    import pdb
                    pdb.set_trace()
            if line[1] not in word_dict.keys():
                word_dict[line[1][0:-2]] = 0
            if line[2] not in word_dict.keys():
                word_dict[line[2][-2]] = 0
        print(count)

def find_words_in_story(file_path, elmo_path, word_dict, vector_dict):
    # return a set of words and corresponding vectors
    text_elmo = scio.loadmat(elmo_path)
    text_elmo = text_elmo['data']
    with open(file_path, 'r') as rf:
        count = 0
        for line in rf:
            if line[-2] != ' ':
                line = line[0:-2]
            line = line.strip('\n').split()
            '''if len(line) != text_elmo[count].shape:
                import pdb
                pdb.set_trace()'''
            for word in line:
                if word in word_dict:
                    if word == 'salt':
                        import pdb
                        pdb.set_trace()
                    word_dict[word] += 1
                    vector_dict[word] += text_elmo[count]
                count += 1
            count += 1
        if count != text_elmo.shape[0]:
            print('error!'+file_path)

if __name__ == "__main__":
    csv_path = ["mturk-287.csv", "mturk-771.csv", "rg-65.csv", 'simlex999.csv', 'verb-143.csv', 'wordsim353-rel.csv', 'wordsim353-sim.csv', 'yp-130.csv']
    scales = [2, 2, 2.5, 1, 2.5, 1, 1, 2.5]
    story_path = ""
    elmo_path = ""

    pair_dict = {}
    word_dict = {}

    '''for i, j in zip(csv_path, scales):
        load_csv('word_similarity_en/'+i, pair_dict, word_dict, j)
        print(len(word_dict.keys()))
    load_men('word_similarity_en/men.csv', pair_dict, word_dict, 0.2)'''
    load_simverb('word_similarity_en/simverb-3500.csv', pair_dict, word_dict, 1)

    vector_dict = {}
    for i in word_dict.keys():
        vector_dict[i] = np.zeros([1, 1024])
    for i in range(1, 10):
        file_path = 'story_script/story_0'+str(i)+'.txt'
        elmo_path = 'story_elmos_mat/story_0'+str(i)+'.mat'
        find_words_in_story(file_path, elmo_path, word_dict, vector_dict)
    for i in tqdm(range(10, 52)):
        file_path = 'story_script/story_'+str(i)+'.txt'
        elmo_path = 'story_elmos_mat/story_'+str(i)+'.mat'
        find_words_in_story(file_path, elmo_path, word_dict, vector_dict)
    import pdb
    pdb.set_trace()
    for i in word_dict.keys():
        vector_dict[i] = np.sum(vector_dict[i], 0)/word_dict[i]
    words = list(word_dict.keys())
    pair_list = []
    vector_list = []
    similarity = {}
    similarity['ori'] = []
    corrs = {}
    rating = []
    import pdb
    pdb.set_trace()
    for i in tqdm(range(len(words))):
        if word_dict[words[i]] == 0:
            continue
        for j in range(i+1, len(words)):
            if word_dict[words[j]] == 0:
                continue
            t = words[i]+' '+words[j]
            if t in pair_dict.keys():
                pair_list.append(t)
                vector_list.append([vector_dict[words[i]], vector_dict[words[j]]])
                p = cosine_similarity(vector_dict[words[i]].reshape(1, 1024), vector_dict[words[j]].reshape(1, 1024))
                if p == 0:
                    import pdb
                    pdb.set_trace()
                similarity['ori'].append(p[0][0])
                rating.append(pair_dict[t])
    corrs['ori'] = np.corrcoef(rating, similarity['ori'])
    # projection_path = 'results_ontonotes/null_Ps/'
    projection_path = 'results_all/null_Ps/'
    tasks = ['pos', 'ner', 'srl', 'dep']
    for i in tasks:
        similarity[i] = []
        p_matrix = pickle.load(open(projection_path+'null_'+i+'.pkl', 'rb'))
        if p_matrix.shape[0] == 2048:
            p_matrix = p_matrix[0:1024, 0:1024]
        for j in vector_list:
            p = cosine_similarity(np.matmul(p_matrix, j[0]).reshape(1, 1024), np.matmul(p_matrix, j[1]).reshape(1, 1024))
            similarity[i].append(p[0][0])
    import pdb
    pdb.set_trace()
    for i in tasks:
        corrs[i] = np.corrcoef(rating, similarity[i])
    import pdb
    pdb.set_trace()
    print(corrs)

