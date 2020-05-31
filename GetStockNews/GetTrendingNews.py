from random import randint
import json
from pathlib import Path
import os
import pandas as pd
import argparse
import sys
import numpy as np
import glob
import re
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from datetime import timedelta

# COnstant for folder name
REUTER_DIR      = Path('GetStockNews/ReutersNews')
Stocktwits_DIR  = Path('GetStockNews/Stocktwits')
Trending_DIR    = Path('GetStockNews/TrendingNews')
NEWS_DIR        = Path('GetStockNews/news')
StockNewsReturn = str(NEWS_DIR) + '\StockNewsReturn.json'

# Column names from the topic text files
all_topic = ['All_Reuter_topic0', 'All_Reuter_topic1', 'All_Reuter_topic2']
uni_topic = ['Uni_Reuter_topic0', 'Uni_Reuter_topic1', 'Uni_Reuter_topic2']
bi_topic  = ['Bi_Reuter_topic0', 'Bi_Reuter_topic1','Bi_Reuter_topic2']
tri_topic = ['tri_Reuter_topic0', 'tri_Reuter_topic1', 'tri_Reuter_topic2']

stopword    = stopwords.words("English")
mystopwords = []
stopword.extend(mystopwords)

end_date = pd.to_datetime('01-13-2020') + timedelta(days=5)
end_date = str(end_date.month) + '-' + str(end_date.day) + '-' + str(end_date.year)

# Define FUnctions 

def topic_pre_process(text):
    # Remove all the special characters
    processed_text = re.sub(r'\W', ' ', str(text))
    # Substituting multiple spaces with single space
    processed_text= re.sub(r'\s+', ' ', processed_text, flags=re.I)
    # Substituting rows with only numbers with null
    processed_text = re.sub(r'[0-9]+', '', processed_text)
    processed_text = re.sub(r"\d", "", processed_text)
    processed_text = processed_text.lower()

    tokens = nltk.word_tokenize(processed_text)
    tokens=[ t for t in tokens if t not in stopword]
    text_after_process =" ".join(tokens)
    return(text_after_process)

# Get all stockName json file 
def GetTrendingNews():
    file = os.listdir(REUTER_DIR)
    
    Reuter_df = pd.DataFrame()
    for filename in os.listdir(REUTER_DIR):
         with open(REUTER_DIR / filename) as json_file:
            data = json.load(json_file)
            df = pd.DataFrame(data['news_items'])
            df = df[["date","headline","url"]].sort_values(by=['date'],ascending = False)
            Reuter_df = pd.concat([Reuter_df,df])
    Reuter_df['headline'] = Reuter_df.headline.apply(topic_pre_process)
    Reuter_df['date']= pd.to_datetime(Reuter_df['date']) 
    mask = (Reuter_df['date'] <= end_date)
    Reuter_df = Reuter_df.loc[mask]
    return Reuter_df

# Get the topics generated from the Topic Modelling
def GetTopics():
    
    # extract list of text files under the Trending folder
    filelist = []
    filesList = []
    file_ext = "*.txt"
    filenames = str(Trending_DIR / file_ext)
    # Build up list of files:
    for files in glob.glob(filenames):
        fileName, fileExtension = os.path.splitext(files)
        filelist.append(fileName) #filename without extension
        filesList.append(files) #filename with extension

    #Read the 3 topics text file from previous preprocessed Stocktwits datasets
    data = pd.concat([pd.read_csv(item, names=[item[26:-4]]) for item in filesList], axis=1)

    return data

# Get the bigram topics to search for matching string in Reuter headline and return 
# the matching headline data
def GetAnswer():
    Reuter_df = GetTrendingNews()
    topics = GetTopics()

    match_topic = pd.DataFrame()

    for col in bi_topic:   
        for i in range(len(topics)):
            df = pd.DataFrame()
            df = Reuter_df[Reuter_df['headline'].str.contains(topics[col][i])]
            match_topic = pd.concat([match_topic,df],ignore_index = True)

    match_topic = match_topic.drop_duplicates(subset='headline', keep='first')
    top5_topics = match_topic.sort_values(by='date',ascending = False)
    top5_topics = top5_topics.reset_index(drop = True)
    choice = randint(0, 10)

    if len(top5_topics) == 0:
        Reuter_df = Reuter_df.sort_values(by='date',ascending = False)
        text  = Reuter_df['headline'][choice]
        url   = Reuter_df['url'][choice]
        reply = '<a href="'+url+'">'+text+'</a>.'
        print("Reuter")
    else:
        text  = top5_topics['headline'][choice]
        url   = top5_topics['url'][choice]    
        reply = '<a href="'+url+'">'+text+'</a>.'
        print("topic")

    return (reply)