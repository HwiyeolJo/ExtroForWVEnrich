from __future__ import print_function
import gzip
import math
import numpy as np
import re
from copy import deepcopy
import h5py
from tqdm.notebook import tqdm
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

### Hyperparameter
Step = 1
WordDim = 300
NormRead = True
nNorm = 2
#####
# WordVecKinds = "Extro" # "GloVe", "TestGloVe","Word2Vec", or "FastText" + "Extro"
#####

def read_word_vecs(filename):
    print("Vectors read from", filename)
    wordVectors = {}
    if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
    elif filename.endswith('.h5'):
        fileObject_ = h5py.File(filename, 'r')
        fileObject = fileObject_[list(fileObject_.keys())[0]]
        wordlist = np.array(fileObject[u'axis1'])
        pbar = tqdm(total = len(wordlist))
        unk_counter = 0
        for i, w in enumerate(wordlist):
            pbar.update(1)
            wordVectors[w[6:]] = fileObject[u'block0_values'][i]
        pbar.close()
    else:
        fileObject = open(filename, 'r', encoding='utf-8')
        fileObject.readline() # For handling First Line
        for line in fileObject:
            line = line.strip().lower()
            word = line.split()[0]
#             wordVectors[word] = np.zeros(len(line.split())-1, dtype=np.float64)
            vector = line.split()[1:]
            if len(vector) == WordDim:
#                 for index, vecVal in enumerate(vector):
                wordVectors[word] = list(map(float, vector))
#                     wordVectors[word][index] = float(vecVal)
                if NormRead:
                    wordVectors[word] = np.array(wordVectors[word]) / math.sqrt((np.array(wordVectors[word])**2).sum() + 1e-5)
            else:
                print(line)
                continue
    return wordVectors

def norm_word(word): # Could Add Preprocessing
    isNumber = re.compile(r'\d+.*')
    if isNumber.search(word.lower()):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word.lower()
    
def read_lexicon(filename, wordVecs):
    lexicon = {}
    for line in open(filename, 'r', encoding='utf-8'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon

def wordVecsLDA(wordVecs):
    newWordVecs = deepcopy(wordVecs)
    wordVec_np = []
    
#     pbar = tqdm(total = len(newWordVecs))
    for k, v in newWordVecs.items():
#         pbar.update(1)
        if len(v) != WordDim+2:
            print(k, len(v))
            return
            
        wordVec_np.append(v)
    wordVec_np = np.array(wordVec_np)
#     pbar.close()
    print("Run LDA ...")
    
    pbar = tqdm(total = 1)
    lda = LinearDiscriminantAnalysis(n_components=WordDim)
    print(wordVec_np.shape)
    print(wordVec_np[:,:-1].shape, wordVec_np[:,-1].shape)
    wordVec_np = lda.fit_transform(wordVec_np[:,:-1], wordVec_np[:,-1].astype('int'))
    print(wordVec_np.shape)
    pbar.update(1)
    pbar.close()
    
    print("LDA Done ...")
    pbar = tqdm(total = len(newWordVecs))
    for i, k in enumerate(newWordVecs):
        pbar.update(1)
        newWordVecs[k] = wordVec_np[i]
    pbar.close()
    return newWordVecs

def extrofit(wordVecs, lexicon, it):
    newWordVecs = deepcopy(wordVecs)
    wvVocab = set(newWordVecs.keys())
    loopVocab = wvVocab.intersection(set(lexicon.keys()))
    if it == 0:
        print(len(loopVocab), "words will be extrofitted")
        
    wordidx = 0
    
    for w in wvVocab:
        newWordVecs[w] = np.hstack((newWordVecs[w], np.mean(wordVecs[w])))
        newWordVecs[w] = np.hstack((newWordVecs[w], np.zeros(1)))
        
    for word in wvVocab:
        wordidx = wordidx+1
        try:
            wordNeighbours = set(lexicon[word]).intersection(wvVocab)
            numNeighbours = len(wordNeighbours)
        except KeyError:
#             print("KeyError")
            numNeighbours = 0

        if numNeighbours == 0:
            newWordVecs[word][-1] = wordidx
        else:
            newWordVecs[word][-2] += np.mean([np.mean(wordVecs[w]) for w in wordNeighbours])
            for w in wordNeighbours:
                newWordVecs[w][-1] = wordidx

    ### LDA for dimension reduction
    print("Dimension Reduction ... ")
    newWordVecs = wordVecsLDA(newWordVecs)

    return newWordVecs

def print_word_vecs(wordVectors, outFileName):
    print('Writing down the vectors in', outFileName)
    outFile = open(outFileName, 'w', encoding="utf-8")
    outFile.write(str(len(wordVectors)) + ' ' + str(WordDim) + '\n')
    pbar = tqdm(total = len(wordVectors), desc = 'Writing')
    for word, values in wordVectors.items():
        pbar.update(1)
        outFile.write(word+' ')
        for val in wordVectors[word]:
            outFile.write('%.5f' %(val)+' ')
        outFile.write('\n')
    outFile.close()
    pbar.close()
    
def retrofit(wordVecs, lexicon, numIters):
    newWordVecs = deepcopy(wordVecs)
    wvVocab = set(newWordVecs.keys())
    loopVocab = wvVocab.intersection(set(lexicon.keys()))
    print(len(loopVocab), "NE will be retrofitted")
    pbar = tqdm(total = numIters, desc = 'Epoch')
    cntNeighbours = 0
    for it in range(numIters):
        pbar.update(1)
        for word in loopVocab:
            wordNeighbours = set(lexicon[word]).intersection(wvVocab)
            numNeighbours = len(wordNeighbours)
#             print(*wordNeighbours, end=' ')
            cntNeighbours = cntNeighbours + numNeighbours
            if numNeighbours == 0:
                continue
            ### Retrofitting
            newVec = numNeighbours * np.array(wordVecs[word])
            for ppWord in wordNeighbours:
                newVec += newWordVecs[ppWord]
            newWordVecs[word] = newVec/(2*numNeighbours)
            #####
    pbar.close()
    print(cntNeighbours/numIters)
    return newWordVecs