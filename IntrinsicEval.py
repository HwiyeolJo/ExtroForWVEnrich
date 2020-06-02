import operator
import scipy.stats
import numpy as np

def SemEvalLoader():
    """ SemEval """
    fp_trial = open("../../Data/WordSimDataset/SemEval17-Task2/trial/subtask1-monolingual/data/en.trial.data.txt", 'r'); trial = fp_trial.read().split('\n')
    fp_test = open("../../Data/WordSimDataset/SemEval17-Task2/test/subtask1-monolingual/data/en.test.data.txt", 'r'); test = fp_test.read().split('\n')
    fp_trial_score = open("../../Data/WordSimDataset/SemEval17-Task2/trial/subtask1-monolingual/keys/en.trial.gold.txt", 'r'); trial_score = fp_trial_score.read().split('\n')
    fp_test_score = open("../../Data/WordSimDataset/SemEval17-Task2/test/subtask1-monolingual/keys/en.test.gold.txt", 'r'); test_score = fp_test_score.read().split('\n')
    data_sem, score_sem = [], []
    for row in trial: data_sem.append(row.split('\t'))
    for row in test: data_sem.append(row.split('\t'))
    for row in trial_score: score_sem.append(row)
    for row in test_score: score_sem.append(row)
    
    fp_trial.close(); fp_test.close(); fp_trial_score.close(); fp_test_score.close()
    data_sem.pop(); score_sem.pop()
    data_sem = np.array(data_sem); score_sem = np.array(score_sem)

    word_to_idx_sem = {}

    idx = 0
    for w in data_sem[:,0]:
        try: word_to_idx_sem[w]
        except KeyError:
            word_to_idx_sem[w] = idx
            idx = idx+1

    for w in data_sem[:,1]:
        try: word_to_idx_sem[w]
        except KeyError:
            word_to_idx_sem[w] = idx
            idx = idx+1

    word_to_idx_sem = sorted(word_to_idx_sem.items(), key=operator.itemgetter(1))
    return data_sem, dict(word_to_idx_sem), score_sem
    
""" MEN-3k """
def MENLoader():
    fp_men = open("../../Data/WordSimDataset/MEN/MEN_dataset_natural_form_full", 'r')
    fp_men_ = fp_men.read().split('\n')

    data_men = []
    for row in fp_men_: data_men.append(row.split(' '))
    data_men.pop()
    data_men = np.array(data_men)
    fp_men.close()

    word_to_idx_men = {}

    idx = 0
    for w in data_men[:,0]:
        try: word_to_idx_men[w]
        except KeyError:
            word_to_idx_men[w] = idx
            idx = idx+1

    for w in data_men[:,1]:
        try: word_to_idx_men[w]
        except KeyError:
            word_to_idx_men[w] = idx
            idx = idx+1

    # print(word_to_idx_men[max(word_to_idx_men, key=word_to_idx_men.get)])
    # print(len(data_men[:,0]) + len(data_men[:,1])) # diff 1 because of starting index 0
    word_to_idx_men = sorted(word_to_idx_men.items(), key=operator.itemgetter(1))
    # for i in range(5): print(data_men[i])
    return data_men, dict(word_to_idx_men)
    
def WSLoader():
    """ WordSim353 """
    fp_ws = open("../../Data/WordSimDataset/WordSim353/combined.csv", 'r', encoding='utf-8')
    fp_ws_ = fp_ws.read().split('\n')
    # print(fp_ws_)

    data_ws = []
    for row in fp_ws_: data_ws.append(row.split(','))
    data_ws.pop(0); data_ws.pop()
    data_ws = np.array(data_ws)
    fp_ws.close()

    word_to_idx_ws = {}

    idx = 0
    for w in data_ws[:,0]:
        try:
            word_to_idx_ws[w]
    #         print(w)
        except KeyError:
            word_to_idx_ws[w] = idx
            idx = idx+1

    for w in data_ws[:,1]:
        try:
            word_to_idx_ws[w]
        except KeyError:
            word_to_idx_ws[w] = idx
            idx = idx+1

    # print(word_to_idx_ws[max(word_to_idx_ws, key=word_to_idx_ws.get)])
    # print(len(data_ws[:,0]) + len(data_ws[:,1])) # diff 1 because of starting index 0
    word_to_idx_ws = sorted(word_to_idx_ws.items(), key=operator.itemgetter(1))
    # for i in range(5): print(data_ws[i])
    return data_ws, dict(word_to_idx_ws)

def SimLexLoader():
    """SimLex-999"""
    fp_sim = open("../../Data/WordSimDataset/SimLex-999/SimLex-999.txt", 'r')
    fp_sim.readline() # Removing its header
    fp_sim_ = fp_sim.read().split('\n')

    data_sim = []
    for row in fp_sim_: data_sim.append(row.split('\t'))
    data_sim.pop()
    data_sim = np.array(data_sim)
    fp_sim.close()

    word_to_idx_sim = {}

    idx = 0
    for w in data_sim[:,0]:
        try:
            word_to_idx_sim[w]
        except KeyError:
            word_to_idx_sim[w] = idx
            idx = idx+1

    for w in data_sim[:,1]:
        try:
            word_to_idx_sim[w]
        except KeyError:
            word_to_idx_sim[w] = idx
            idx = idx+1

    # print(word_to_idx_sim[max(word_to_idx_sim, key=word_to_idx_sim.get)])
    # print(len(data_sim[:,0]) + len(data_sim[:,1])) # diff 1 because of starting index 0
    word_to_idx_sim = sorted(word_to_idx_sim.items(), key=operator.itemgetter(1))
    # for i in range(5): print(data_sim[i])
    return data_sim, dict(word_to_idx_sim)

def RGLoader():
    """ RG-65 """
    fp_rg = open("../../Data/WordSimDataset/rg65.csv", 'r')
    fp_rg_ = fp_rg.read().split('\n')

    data_rg = []
    for row in fp_rg_:
        data_rg.append(row.split(';'))
    # data_rg.pop(0); data_rg.pop()
    data_rg = np.array(data_rg)
    # data_rg[:,-1] = float(data_rg)
    # print(data_rg)
    fp_rg.close()

    word_to_idx_rg = {}

    idx = 0
    for w in data_rg[:,0]:
        try:
            word_to_idx_rg[w]
    #         print(w)
        except KeyError:
            word_to_idx_rg[w] = idx
            idx = idx+1

    for w in data_rg[:,1]:
        try:
            word_to_idx_rg[w]
        except KeyError:
            word_to_idx_rg[w] = idx
            idx = idx+1

    # print(word_to_idx_rg[max(word_to_idx_rg, key=word_to_idx_rg.get)])
    # print(len(data_rg[:,0]) + len(data_rg[:,1])) # diff 1 because of starting index 0
    word_to_idx_rg = sorted(word_to_idx_rg.items(), key=operator.itemgetter(1))
    # for i in range(5): print(data_rg[i])
    return data_rg, dict(word_to_idx_rg)
    
def RareWordsLoader():
    """Rare Words"""
    fp_rw = open("../../Data/WordSimDataset/rw/rw.txt", 'r')
    # fp_sim.readline() # Removing its header
    fp_rw_ = fp_rw.read().split('\n')

    data_rw = []
    for row in fp_rw_: data_rw.append(row.split('\t')[:3])
    data_rw.pop()

    data_rw = np.array(data_rw)
    fp_rw.close()

    word_to_idx_rw = {}

    idx = 0
    for w in data_rw[:,0]:
        try:
            word_to_idx_rw[w]
        except KeyError:
            word_to_idx_rw[w] = idx
            idx = idx+1

    for w in data_rw[:,1]:
        try:
            word_to_idx_rw[w]
        except KeyError:
            word_to_idx_rw[w] = idx
            idx = idx+1

    # print(word_to_idx_rw[max(word_to_idx_rw, key=word_to_idx_rw.get)])
    # print(len(data_rw[:,0]) + len(data_rw[:,1])) # diff 1 because of starting index 0
    word_to_idx_rw = sorted(word_to_idx_rw.items(), key=operator.itemgetter(1))
    # for i in range(5): print(data_rw[i])
    return data_rw, dict(word_to_idx_rw)
    
def SimVerbLoader():
    """SimVerb"""
    fp_sv = open("../../Data/WordSimDataset/SimVerb-3500/SimVerb-3500.txt", 'r')
    fp_sv_ = fp_sv.read().split('\n')

    data_sv = []
    for row in fp_sv_: data_sv.append(row.split('\t')[0:2] + row.split('\t')[3:4])
    data_sv.pop()

    data_sv = np.array(data_sv)
    fp_sv.close()

    word_to_idx_sv = {}

    idx = 0
    for w in data_sv[:,0]:
        try:
            word_to_idx_sv[w]
        except KeyError:
            word_to_idx_sv[w] = idx
            idx = idx+1

    for w in data_sv[:,1]:
        try:
            word_to_idx_sv[w]
        except KeyError:
            word_to_idx_sv[w] = idx
            idx = idx+1

    # print(word_to_idx_sv[max(word_to_idx_sv, key=word_to_idx_sv.get)])
    # print(len(data_sv[:,0]) + len(data_sv[:,1])) # diff 1 because of starting index 0
    word_to_idx_sv = sorted(word_to_idx_sv.items(), key=operator.itemgetter(1))
    # for i in range(5): print(data_sv[i])
    return data_sv, dict(word_to_idx_sv)

""" Calculating Similarity """
def LoadInputVector(wordvec, data, lookup):
    input1, input2 = [], []
    unk_cnt = 0
    veclen = len(wordvec[list(wordvec.keys())[0]])
    for i in range(len(data[:,0])):
#         print(wordvec)
        try: input1.append(wordvec[data[i,0]])
        except KeyError: input1.append(np.random.normal(0., 1., veclen)); unk_cnt = unk_cnt+1
        try: input2.append(wordvec[data[i,1]])
        except KeyError: input2.append(np.random.normal(0., 1., veclen)); unk_cnt = unk_cnt+1
#     print("UNK_CNT :", unk_cnt, end='/')
    return np.array(input1), np.array(input2)

def Evaluating_Semeval(wordvec, data, lookup, score):
    input1, input2 = LoadInputVector(wordvec, data, lookup)
    output = []
    epsilon = 1e-5
    for i in range(len(input1)):
        output.append(np.dot(input1[i], input2[i])/(np.linalg.norm(input1[i])*np.linalg.norm(input2[i])))
    output = (np.array(output)).reshape(-1)
    return round(scipy.stats.spearmanr(output, np.array(score, dtype=float))[0], 4)

def Evaluating_MEN(wordvec, data, lookup):
    input1, input2 = LoadInputVector(wordvec, data, lookup)
    output = []
    epsilon = 1e-5
    for i in range(len(input1)):
        output.append(np.dot(input1[i], input2[i])/(np.linalg.norm(input1[i])*np.linalg.norm(input2[i])))
    output = (np.array(output)).reshape(-1)
    return round(scipy.stats.spearmanr(output, np.array(data[:,2], dtype=float))[0], 4)

def Evaluating_WS(wordvec, data, lookup):
    input1, input2 = LoadInputVector(wordvec, data, lookup)
    output = []
    epsilon = 1e-5
    for i in range(len(input1)):
        output.append(np.dot(input1[i], input2[i])/(np.linalg.norm(input1[i])*np.linalg.norm(input2[i])))
    output = (np.array(output)).reshape(-1)
    return round(scipy.stats.spearmanr(output, np.array(data[:,2], dtype=float))[0], 4)

def Evaluating_SIM(wordvec, data, lookup):
    input1, input2 = LoadInputVector(wordvec, data, lookup)
    output = []
    epsilon = 1e-5
    for i in range(len(input1)):
        output.append(np.dot(input1[i], input2[i])/(np.linalg.norm(input1[i])*np.linalg.norm(input2[i])))
    output = (np.array(output)).reshape(-1)
    return round(scipy.stats.spearmanr(output, np.array(data[:,3], dtype=float))[0], 4)

def Evaluating_RG(wordvec, data, lookup):
    input1, input2 = LoadInputVector(wordvec, data, lookup)
    output = []
    epsilon = 1e-5
    for i in range(len(input1)):
        output.append(np.dot(input1[i], input2[i])/(np.linalg.norm(input1[i])*np.linalg.norm(input2[i])))
    output = (np.array(output)).reshape(-1)
    return round(scipy.stats.spearmanr(output, np.array(data[:,2], dtype=float))[0], 4)

def Evaluating_RW(wordvec, data, lookup):
    input1, input2 = LoadInputVector(wordvec, data, lookup)
    output = []
    epsilon = 1e-5
    for i in range(len(input1)):
        output.append(np.dot(input1[i], input2[i])/(np.linalg.norm(input1[i])*np.linalg.norm(input2[i])))
    output = (np.array(output)).reshape(-1)
    return round(scipy.stats.spearmanr(output, np.array(data[:,2], dtype=float))[0], 4)

def Evaluating_SV(wordvec, data, lookup):
    input1, input2 = LoadInputVector(wordvec, data, lookup)
    output = []
    epsilon = 1e-5
    for i in range(len(input1)):
        output.append(np.dot(input1[i], input2[i])/(np.linalg.norm(input1[i])*np.linalg.norm(input2[i])))
    output = (np.array(output)).reshape(-1)
    return round(scipy.stats.spearmanr(output, np.array(data[:,2], dtype=float))[0], 4)

