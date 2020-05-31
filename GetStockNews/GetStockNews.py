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

nltk.data.path.append("nltk_data\\")

# COnstant for folder name
REUTER_DIR      = Path('GetStockNews/ReutersNews')
Stocktwits_DIR  = Path('GetStockNews/Stocktwits')
NEWS_DIR        = Path('GetStockNews/news')
StockNewsReturn = str(NEWS_DIR) + '\StockNewsReturn.json'

# Column names from the topic text files
all_topic = ['All_topic0', 'All_topic1', 'All_topic2']
uni_topic = ['Uni_topic0', 'Uni_topic1', 'Uni_topic2']
bi_topic  = ['Bi_topic0', 'Bi_topic1','Bi_topic2']
tri_topic = ['tri_topic0', 'tri_topic1', 'tri_topic2']

stopword    = stopwords.words("english")
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

# Get respective stockName json file 
# Function will check for all the supported stock names, if 
# Stock Names not found will return the error message
def GetReuterNews(StockName):
    json_filename = StockName + ".json"
    file     = os.listdir(REUTER_DIR)

    # Check for input validation 
    count = 0
    for filename in os.listdir(REUTER_DIR):
        if json_filename not in filename:
            count +=1
    
    # Check if stockname is supported
    if count > (len(file)-1):
        print(stockName + " not in the list. Please try another stock name with -R option") 
        sys.exit(0)
    else:
        with open(REUTER_DIR / json_filename) as json_file:
            data = json.load(json_file)
            df   = pd.DataFrame(data['news_items'])
            df   = df[["date","headline","url"]].sort_values(by=['date'],ascending = False)
            df['date']= pd.to_datetime(df['date']) 
            mask = (df['date'] <= end_date)
            df   = df.loc[mask]  
    return df

# Get the topics generated from the Topic Modelling
def GetTopics():
    
    # extract list of text files under the stocktwist folder
    filelist  = []
    filesList = []
    file_ext  = "*.txt"
    filenames = str(Stocktwits_DIR / file_ext)

    # Build up list of files:
    for files in glob.glob(filenames):
        fileName, fileExtension = os.path.splitext(files)
        filelist.append(fileName) #filename without extension
        filesList.append(files) #filename with extension

    #Read the 3 topics text file from previous preprocessed Stocktwits datasets
    data = pd.concat([pd.read_csv(item, names=[item[24:-4]]) for item in filesList], axis=1)
    
    return data

# Get the bigram topics to search for matching string in Reuter headline and return 
# the matching headline data
def GetAnswer(stockName):
    Reuter_df = GetReuterNews(stockName)
    topics = GetTopics()

    match_topic = pd.DataFrame()

    for col in all_topic:   
        for i in range(len(topics)):
            df = pd.DataFrame()
            df = Reuter_df[Reuter_df['headline'].str.contains(topics[col][i])]
            match_topic = pd.concat([match_topic,df],ignore_index = True)

    match_topic = match_topic.drop_duplicates(subset='headline', keep='first')
    top5_topics = match_topic.sort_values(by='date',ascending = False).head(1)
    top5_topics = top5_topics.reset_index(drop = True)
    
    if len(top5_topics) == 0:
        Reuter_df = Reuter_df.sort_values(by='date',ascending = False).head(1)
        text  = Reuter_df['headline'][0]
        url   = Reuter_df['url'][0]
        reply = '<a href="'+url+'">'+text+'</a>'
        print("Reuter")
    else:
        text  = top5_topics['headline'][0]
        url   = top5_topics['url'][0]    
        reply = '<a href="'+url+'">'+text+'</a>'
        print("topic")

    return (reply)

