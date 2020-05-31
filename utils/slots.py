# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:48:13 2020

@author: Donal
"""
import en_core_web_sm
import pycrfsuite
from utils.NametoSymb import convert
nlp = en_core_web_sm.load()

def getslots (utterance):
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
    

    sentlist = []
    posList = []
    utterance_tok = nlp(utterance)
    for token in utterance_tok:
        posList.append(token.tag_)
    for idx,word in enumerate(utterance.split()):
        sentlist.append((word,posList[idx]))
    utterance_tokens = sent2features(sentlist)
    tagger = pycrfsuite.Tagger()
    tagger.open('models\\CRFModel.crfsuite')

    results = tagger.tag(utterance_tokens)
    zipped = [i if i[1][1] != '-'  else (i[0],i[1][2:]) for i in zip(utterance.split(), results) if i[1] != 'O' ]
    
    slots = {i[1]:i[0] for i in zipped}

    try:
        slots['stockname'] = convert(slots['stockname'])
    except:
        slots['stockname'] = ""
 
    return slots
  

