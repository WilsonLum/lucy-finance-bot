# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:23:40 2020

@author: Donal
"""

from __future__ import unicode_literals, print_function, division
import pickle
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os

import Training
from Training.attentiontraining import Utterances, Labels

def attentionIntentDetection(user_utterance):
    device = torch.device("cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LENGTH = 20
    SOS_token = 0
    EOS_token = 1
    
    utterance = Utterances("utterance")
    label = Labels("label")
    
    with open('models\\utterance.pk', 'rb') as handle:
        utterance = pickle.load(handle)  
    with open('models\\Labels.pk', 'rb') as handle:
        label = pickle.load(handle)
        
    intent = open('models\\intentlist.txt', encoding='utf-8').read().strip().split('\n')
    
    
    class EncoderRNN(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(EncoderRNN, self).__init__()
            self.hidden_size = hidden_size
    
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size)
    
        def forward(self, input, hidden):
            embedded = self.embedding(input).view(1, 1, -1)
            output = embedded
            output, hidden = self.gru(output, hidden)
            return output, hidden
    
        def initHidden(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)
    
    class DecoderRNN(nn.Module):
        def __init__(self, hidden_size, output_size):
            super(DecoderRNN, self).__init__()
            self.hidden_size = hidden_size
    
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, output_size)
            self.softmax = nn.LogSoftmax(dim=1)
    
        def forward(self, input, hidden):
            output = self.embedding(input).view(1, 1, -1)
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
            output = self.softmax(self.out(output[0]))
            return output, hidden
    
        def initHidden(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)
    
    
    class AttnDecoderRNN(nn.Module):
        def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
            super(AttnDecoderRNN, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.dropout_p = dropout_p
            self.max_length = max_length
    
            self.embedding = nn.Embedding(self.output_size, self.hidden_size)
            self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
            self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.dropout = nn.Dropout(self.dropout_p)
            self.gru = nn.GRU(self.hidden_size, self.hidden_size)
            self.out = nn.Linear(self.hidden_size, self.output_size)
    
        def forward(self, input, hidden, encoder_outputs):
            embedded = self.embedding(input).view(1, 1, -1)
            embedded = self.dropout(embedded)
    
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                     encoder_outputs.unsqueeze(0))
    
            output = torch.cat((embedded[0], attn_applied[0]), 1)
            output = self.attn_combine(output).unsqueeze(0)
    
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
    
            output = F.softmax(self.out(output[0]), dim=1)
            return output, hidden, attn_weights
    
        def initHidden(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)
    checkpoint = torch.load("models\\model.hd5")
    hidden_size = 256
    encoder2 = EncoderRNN(len(checkpoint['encoder']['embedding.weight']), hidden_size).to(device)
    attn_decoder2= AttnDecoderRNN(hidden_size, len(checkpoint['decoder']['out.weight']), dropout_p=0.1).to(device)
    
    encoder2.load_state_dict(checkpoint['encoder'])
    attn_decoder2.load_state_dict(checkpoint['decoder'])
    
    
    
    
    def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
        with torch.no_grad():
            input_tensor = tensorFromSentence(sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()
    
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]
    
            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
    
            decoder_hidden = encoder_hidden
    
            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)
    

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[0] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            #if topi.item() == EOS_token:
            #    decoded_words.append('<EOS>')
            #    break
            #else:
            decoded_words.append(label.index2label[topi.item()])

            decoder_input = topi.squeeze().detach()
    
            return decoded_words, decoder_attentions[:0 + 1], decoder_output.data[0][decoder_output.data.argmax()]
    
    def evaluateAndShowAttention(input_sentence):
        output_words, attentions, score = evaluate(
            encoder2, attn_decoder2, input_sentence)
        #print('input =', input_sentence)
        #print('output =', ' '.join(output_words))
        return output_words, score
    
    
    def indexesFromSentence(sentence):
        indexing = []
        for word in sentence.split(' '):
            try:
                G = utterance.word2index[word]
            except:
                G = utterance.word2index["UNK"]
            indexing.append(G)
        return indexing
        #return [utterance.word2index[word] for word in sentence.split(' ')]
    
    def indexesFromLabels(labelnum):
        return label.label2index[labelnum]
    
    def tensorFromSentence(sentence):
        indexes = indexesFromSentence(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
    
    def tensorFromLabel(label):
        indexes = (indexesFromLabels(label))
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
    
    
    def tensorsFromPair(pair):
        input_tensor = tensorFromSentence(pair[0])
        target_tensor = tensorFromLabel(pair[1])
        return (input_tensor, target_tensor)
    
    user_utterance = re.sub(r'[^\w]', ' ', user_utterance)
    output = [i for i in evaluateAndShowAttention(user_utterance)]
    prediction, score = intent[int(output[0][0])], float(output[1])
    prediction = prediction.replace("-","_")
    #prediction, score = intent[int(evaluateAndShowAttention(user_utterance)[0][0])]
    return prediction, score

