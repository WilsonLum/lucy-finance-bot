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
from conversation import smalltalk, fallback, Finance


import pycrfsuite
import spacy
import en_core_web_sm


##prepare dataset
from os import getcwd, chdir


utterance = 'latest news'

def intentdetection(utterance):
    intent = ""
    utterance_pred = 'root'
    #nextclass = "Finance"
    while 1:
        try:
            vectorizer = pickle.load(open("models//vec_"+utterance_pred+".pk",'rb'))
            kbest = pickle.load(open("models//kbest_"+utterance_pred+".pk",'rb'))
            svm = pickle.load(open("models//svm_"+utterance_pred+".pk",'rb'))
            label = pickle.load(open("models//lbl_"+utterance_pred+".pk",'rb'))
            listlevelnames = pickle.load(open("models//listlevelnames.pkl",'rb'))
            utterance_vect = vectorizer.transform([utterance])
            utterance_kbest = kbest.transform(utterance_vect)
            utterance_pred, pred_score = list(svm.predict(utterance_kbest))[0], np.amax(svm.predict_proba(utterance_kbest)[0])
            #nextclass = label[utterance_pred]
            intent += utterance_pred + "_"
        except:
            intent = intent[:-1]
            intent = ''.join([i if i != "-" else "_" for i in intent ])
            break
        
    return intent, pred_score