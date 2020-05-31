# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:05:34 2020

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
from conversation import smalltalk, fallback, Finance
from utils.intent import intentdetection
from utils.attentionIntent import attentionIntentDetection
import settings.store

import pycrfsuite
import spacy
import en_core_web_sm


##prepare dataset
from os import getcwd, chdir



#def intentdetection (utterance):
#    intentlist = pickle.load(open("models//intentlist.pkl",'rb'))
#    vectorizer = pickle.load(open("models//vectorizer.pk",'rb'))
#    kbest = pickle.load(open("models//kbest.pk",'rb'))
#    svm = pickle.load(open("models//svm.pk",'rb'))
#    
#    utterance_vect = vectorizer.transform([utterance])
#    utterance_kbest = kbest.transform(utterance_vect)
#    utterance_pred, pred_score = int(svm.predict(utterance_kbest)), np.amax(svm.predict_proba(utterance_kbest)[0])
#    print(utterance_pred)
#    utterance_intent = intentlist[utterance_pred]
#    return utterance_intent , pred_score

def reformatting (Reply, Intent, Confidence):
    result = {"Reply" : Reply,
            "Intent": Intent,
            "Confidence": Confidence}
    return result



#def getreply(update, context):
#    utterance = update.message.text
#    intent, pred_score = intentdetection(utterance)
#    if pred_score <= 0.2 :
#        method_to_call = fallback()
#        result = reformatting("".join(method_to_call), intent, pred_score)
#    else:
#        if intent[:intent.index('_')] == 'smalltalk':
#            method_to_call = getattr(smalltalk, intent)()
#            result = reformatting("".join(method_to_call), intent, pred_score)

#        elif intent[:intent.index('_')] == 'Finance':
#            method_to_call = getattr(Finance, intent)(utterance)
#            result = reformatting("".join(method_to_call), intent, pred_score)
        
#    return result
    


def getreply(update, context):
    use_attention = False
    if use_attention: 
        intent_detection = attentionIntentDetection
    else:
        intent_detection = intentdetection
    
    settings.store = context
    utterance = update.message.text
    user_data = context.user_data
    try:
        context.user_data["PI"]
    except:
        context.user_data["PI"] = ""

    if context.user_data["PI"] == "date":
        context.user_data["DATE"] = utterance
        context.user_data["PI"] = ""

        result = reformatting("".join("date changed to "+context.user_data["DATE"]), 0, 0)
    else:
        intent, pred_score = intent_detection(utterance)
        print(f'Intent :  {intent}  Pred_score : {pred_score} ')
        # confidence threshold
        if pred_score <= 0.2 :

            method_to_call = getattr(fallback, "fallback")()
            result = reformatting("".join(method_to_call), intent, pred_score)

        else:

            if intent[:intent.index('_')] == 'smalltalk':
                method_to_call = getattr(smalltalk, intent)()
                result = reformatting("".join(method_to_call), intent, pred_score)

            elif intent[:intent.index('_')] == 'Finance':
                    if intent[intent.index('_')+1:intent.index('_')+1+intent[intent.index('_')+1:].index('_')] == 'slots':
                        #handles no Past Intent with past slots
                        if user_data["PI"] != "":
                            print("Past Context route to " + user_data["PI"])
                            method_to_call = getattr(Finance, user_data["PI"])(utterance, context)
                            result = reformatting("".join(method_to_call), intent, pred_score)
                        else:
                            #handles Past Intent with past slots
                            reply = "What do you want to know about " + utterance + "?"
                            user_data["ticker"] = utterance
                            print("slots: " + user_data["ticker"])
                            result = reformatting("".join(reply), 0, 0)
                    else:
                        try:
                            user_data["PI"] = intent
                            method_to_call = getattr(Finance, intent)(utterance, context)
                            print("new intent " + intent)
                            result = reformatting("".join(method_to_call), intent, pred_score)
                        except:
                            method_to_call = getattr(fallback, "fallback")()
                            result = reformatting("".join(method_to_call), "fallback", 0)
    return result
    

