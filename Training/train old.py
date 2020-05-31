# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:43:56 2020

@author: Donal
"""

import os
import json
import codecs
import pandas as pd
import numpy as np
import random

from itertools import chain
import nltk
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
import conversation.smalltalk

import pycrfsuite
import spacy
import en_core_web_sm


##prepare dataset
from os import getcwd, chdir


def intentdataset(jsonfilepath):
    
    def loadjson(jsonfilepath):
        return json.load(codecs.open(jsonfilepath, 'r', 'utf-8'))

    def getintentlist (rawdata):
        intentlist = list(rawdata["intents"].keys())
        with open("models\\intentlist.pkl",'wb') as fp:
            pickle.dump(intentlist,fp)
        return intentlist

    def getlabeleddataset(intentlist,rawdata):
        train_list = []
        for intentno in range(len(intentlist)):
            subtrainlist = []
            for utterance in rawdata["intents"][intentlist[intentno]]["utterances"]:
                line = ""
                for sequence in utterance['data']:
                    line += sequence['text']
                subtrainlist.append([line.lower(),intentno])                        
            train_list.append(subtrainlist)
        return train_list
    
    def findkbest (X_train,X_test) :
        bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
        train_bigram_vectors = bigram_vectorizer.fit_transform(X_train)
        test_bigram_vectors = bigram_vectorizer.transform(X_test)
        with open('models\\vectorizer.pk','wb') as fin:
            pickle.dump(bigram_vectorizer,fin)
        
        #print(train_bigram_vectors.shape)
        #print(test_bigram_vectors.shape)
        
        ch21 = SelectKBest(chi2, k='all')
        train_bigram_Kbest = ch21.fit_transform(train_bigram_vectors, y_train)
        test_bigram_Kbest = ch21.transform(test_bigram_vectors)
        with open('models\\kbest.pk','wb') as fin:
            pickle.dump(ch21,fin)
        
        return train_bigram_Kbest, test_bigram_Kbest

    def trylogisticregression(train_bigram_Kbest, test_bigram_Kbest, y_train,  y_test):
        print("Try Logistic Regression")
        print("=======================")
        clf_ME = LogisticRegression(random_state=0, solver='lbfgs').fit(train_bigram_Kbest, y_train)
        predME = clf_ME.predict(test_bigram_Kbest)
        pred = list(predME)
        print(metrics.confusion_matrix(y_test, pred))
        print(metrics.classification_report(y_test, pred))
        
    def trysvm(train_bigram_Kbest, test_bigram_Kbest, y_train,  y_test):
        print("Try SVM")
        print("========")
        model_svm = SVC(C=5000.0, gamma="auto", kernel='rbf', probability= True)
        clr_svm = model_svm.fit(train_bigram_Kbest, y_train)   
        predicted = clr_svm.predict(test_bigram_Kbest)
        print(metrics.confusion_matrix(y_test, predicted))
        print(np.mean(predicted == y_test) )
        print(metrics.classification_report(y_test, predicted))
        with open('models\\svm.pk','wb') as fin:
            pickle.dump(clr_svm,fin)
    print("Starting Intent Training")
    rawdata = loadjson(jsonfilepath)
    intentlist = getintentlist (rawdata)
    labeleddataset = getlabeleddataset(intentlist,rawdata)
    labeleddatasetflattenX, labeleddatasetflattenY = map(list, zip(*[j for i in labeleddataset for j in i]))
    X_train,X_test, y_train,  y_test = train_test_split(labeleddatasetflattenX,labeleddatasetflattenY, test_size=0.2, random_state=42)
    train_bigram_Kbest, test_bigram_Kbest = findkbest (X_train,X_test)
    #trylogisticregression(train_bigram_Kbest, test_bigram_Kbest, y_train,  y_test)
    trysvm(train_bigram_Kbest, test_bigram_Kbest, y_train,  y_test)
    print("Completed Intent Training")


def slotdetection(jsonfilepath):
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    def loadjson(jsonfilepath):
        return json.load(codecs.open(jsonfilepath, 'r', 'utf-8'))
    def getintentlist (rawdata):
        return list(rawdata["intents"].keys())
    def getlabeleddataset(intentlist,rawdata):
        train_list = []
        for intentno in range(len(intentlist)):
            subtrainlist = []
            for utterance in rawdata["intents"][intentlist[intentno]]["utterances"]:
                wordList=[]
                tagList=[]
                posList=[]
                sentlist=[]
                
                for sequence in utterance['data']:
                    text = sequence['text']
                    tokenList = text.split()
                    
                    if 'entity' not in sequence:
                        for tok in tokenList:
                            wordList.append(tok)
                            tagList.append('O')
                    else:
                        for idx,tok in enumerate(tokenList):
                            wordList.append(tok)
                            if idx:
                                tagList.append('I-'+sequence['entity'])
                            else:
                                tagList.append('B-'+sequence['entity'])
                                
                sent = ' '.join(wordList)
                sent_nlp = nlp(sent)
                
                for token in sent_nlp:
                    posList.append(token.tag_)
                    
                    
                for idx,word in enumerate(wordList):
                    sentlist.append((word,posList[idx],tagList[idx]))
            
                subtrainlist.append(sentlist)
            train_list.append(subtrainlist)
        return train_list
    
    
    def word2features(sent, i):
        word = sent[i][0]
        postag = sent[i][1]
        features = [  # for all words
            'bias',
            'word.lower=' + word.lower(),
            #'word[-3:]=' + word[-3:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            'postag=' + postag,
            'postag[:2]=' + postag[:2],
        ]
        if i > 0: # if not <S>
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isupper=%s' % word1.isupper(),
                '-1:word.isdigit=%s' % word1.isdigit(),
                '-1:postag=' + postag1,
                '-1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('BOS')  # beginning of statement
            
        if i < len(sent)-1:  # if not <\S>
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isupper=%s' % word1.isupper(),
                '+1:word.isdigit=%s' % word1.isdigit(),
                '+1:postag=' + postag1,
                '+1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('EOS')
                    
        return features
    
    
    def sent2features(sent):
        return [word2features(sent, i) for i in range(len(sent))]
    
    def sent2labels(sent):
        return [label for token, postag, label in sent]
    
    def sent2tokens(sent):
        return [token for token, postag, label in sent]
    
    
    print("Starting Slots Training")
    
    rawdata = loadjson(jsonfilepath)
    intentlist = getintentlist(rawdata)
    labelleddata = getlabeleddataset(intentlist,rawdata)
    flattendata = [j for i in labelleddata for j in i]
    X_dataset = [sent2features(s) for s in flattendata]
    y_dataset = [sent2labels(s) for s in flattendata]
    #X_train,X_test, y_train,  y_test = train_test_split(X_dataset,y_dataset, test_size=0.2, random_state=42)
    X_train,X_test, y_train,  y_test = X_dataset,X_dataset, y_dataset,y_dataset
    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
    
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    
    trainer.params()
    
    trainer.train('models\\CRFModel.crfsuite')
    tagger = pycrfsuite.Tagger()
    tagger.open('models\\CRFModel.crfsuite')
    
    example_sent = X_test[3]
    print(X_test[3])
    print("Predicted:[\'", '\', \''.join(tagger.tag(example_sent))+'\']')
    print("Correct:  ", y_test[3])
    
    def bio_classification_report(y_true, y_pred):
        """
        Classification report for a list of BIO-encoded sequences.
        It computes token-level metrics and discards "O" labels.
        
        Note that it requires scikit-learn 0.15+ (or a version from github master)
        to calculate averages properly!
        """
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
            
        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
        
        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels = [class_indices[cls] for cls in tagset],
            target_names = tagset,
        )
    
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    
    print(bio_classification_report(y_test, y_pred))
    
    ##############################################################
    ##Let's check what classifier learned
    
    from collections import Counter
    info = tagger.info()
    
    def print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
    
    print("Top likely transitions:")
    print_transitions(Counter(info.transitions).most_common(15))
    
    print("\nTop unlikely transitions:")
    print_transitions(Counter(info.transitions).most_common()[-15:])
    
    ##Check the state features:
    def print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-6s %s" % (weight, label, attr))    
    
    print("Top positive:")
    print_state_features(Counter(info.state_features).most_common(20))
    
    print("\nTop negative:")
    print_state_features(Counter(info.state_features).most_common()[-20:])
    
    print("Completed Slots Training")

    #####################################################################




def train():
    intentdataset("Dataset\\Converted\\smalltalk.json")
    slotdetection("Dataset\\Converted\\tp.json")




