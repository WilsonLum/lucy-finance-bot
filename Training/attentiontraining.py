from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Utterances:
    def __init__(self, name):
        self.name = name
        self.word2index = {"UNK": 2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class Labels:
    def __init__(self, name):
        self.name = name
        self.label2index = {}
        self.index2label = {}
        self.n_labels = 0
        
    def addLabels(self,label):
        if label not in self.label2index:
            self.label2index[label] = self.n_labels
            self.index2label[self.n_labels] = label
            self.n_labels += 1
    

def train():
    print("Starting Utterence Training using Attention RNN")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    SOS_token = 0
    EOS_token = 1
    
    import pickle

    
    

    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    
    # Lowercase, trim, and remove non-letter characters
    
    
    
    def readLangs(reverse=False):
        print("Reading lines...")
    
        # Read the file and split into lines
        lines = open('models/train.txt', encoding='utf-8').\
            read().strip().split('\n')
    
        # Split every line into pairs and normalize
        pairs = [[s for s in l.split('\t')] for l in lines]
    
        utterance = Utterances("data")
        label = Labels("label")
    
        return utterance, label, pairs
    
    
    MAX_LENGTH = 20
    
    
    
    def prepareData(reverse=False):
        utterance, label, pairs = readLangs(reverse)
        print("Read %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            utterance.addSentence(pair[0])
            label.addLabels(pair[1])
        print("Counted words:")
        print(utterance.name, utterance.n_words)
        return utterance, label, pairs
    
    utterance, label, pairs = prepareData(False)
    print(random.choice(pairs))
    
    
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
    
            output = F.log_softmax(self.out(output[0]), dim=1)
            return output, hidden, attn_weights
    
        def initHidden(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)
    
    
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
    
    
    def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
        encoder_hidden = encoder.initHidden()
    
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
        loss = 0
    
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
    
        decoder_input = torch.tensor([[SOS_token]], device=device)
    
        decoder_hidden = encoder_hidden
    
    
    
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        
    
    
        loss += criterion(decoder_output, target_tensor[0])
    
        loss.backward()
    
        encoder_optimizer.step()
        decoder_optimizer.step()
    
        return loss.item() / target_length
    
    
    import time
    import math
    
    
    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
    
    def timeSince(since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    
    
    def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
    
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        training_pairs = [tensorsFromPair(random.choice(pairs))
                          for i in range(n_iters)]
        criterion = nn.NLLLoss()
    
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            #print(target_tensor)
    
            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss
    
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))
    
            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
    
        showPlot(plot_losses)
    
    
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    import matplotlib.ticker as ticker
    import numpy as np
    
    
    def showPlot(points):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)
    
    
    
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
    
            for di in range(1):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                #if topi.item() == EOS_token:
                #    decoded_words.append('<EOS>')
                #    break
                #else:
                decoded_words.append(label.index2label[topi.item()])
    
                decoder_input = topi.squeeze().detach()
    
            return decoded_words, decoder_attentions[:di + 1]
    
    def evaluateRandomly(encoder, decoder, n=100):
        for i in range(n):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = evaluate(encoder, decoder, pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')
    
    
    hidden_size = 256
    encoder1 = EncoderRNN(utterance.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, label.n_labels, dropout_p=0.1).to(device)
    
    trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
    
    evaluateRandomly(encoder1, attn_decoder1)
    
    torch.save({
                'encoder': encoder1.state_dict(),
                'decoder': attn_decoder1.state_dict(),
                }, "models\\model.hd5")
    with open('models\\utterance.pk', 'wb') as handle:
        pickle.dump(utterance,handle,pickle.HIGHEST_PROTOCOL)
#    with open('data\\word2index.pk', 'wb') as handle:
#        pickle.dump(utterance.word2index,handle,pickle.HIGHEST_PROTOCOL)
#    with open('data\\word2count.pk', 'wb') as handle:
#        pickle.dump(utterance.word2count,handle,pickle.HIGHEST_PROTOCOL)
#    with open('data\\index2word.pk', 'wb') as handle:
#        pickle.dump(utterance.index2word,handle,pickle.HIGHEST_PROTOCOL)
    with open('models\\labels.pk', 'wb') as handle:
       pickle.dump(label, handle,pickle.HIGHEST_PROTOCOL)
#    with open('data\\label2index.pk', 'wb') as handle:
#        pickle.dump(label.label2index, handle,pickle.HIGHEST_PROTOCOL)    
#    with open('data\\index2label.pk', 'wb') as handle:
#        pickle.dump(label.index2label, handle,pickle.HIGHEST_PROTOCOL)    


    def showAttention(input_sentence, output_words, attentions):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)
    
        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' ') +
                           ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)
    
        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
        plt.show()
    
    
    def evaluateAndShowAttention(input_sentence):
        output_words, attentions = evaluate(
            encoder1, attn_decoder1, input_sentence)
        #print('input =', input_sentence)
        #print('output =', ' '.join(output_words))
        return output_words
    
        #showAttention(input_sentence, output_words, attentions)
    
    evaluateAndShowAttention("what is the latest news for apple")
    test = open('models/test.txt', encoding='utf-8').\
            read().strip().split('\n')
    intent = open('models/intentlist.txt', encoding='utf-8').read().strip().split('\n')
            
    x_test, y_test = zip(*[i.split('\t') for i in test])
    
    y_pred = [evaluateAndShowAttention(i) for i in x_test]
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    
    print(metrics.confusion_matrix(y_test, y_pred))
    print(np.mean(y_pred == y_test))
    print(metrics.classification_report(y_test, y_pred))


