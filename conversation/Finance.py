"""
Created on Wed Mar 18 09:25:26 2020

@author: Donal
"""

import random
from utils.slots import getslots as slotsdetection
from GetStockNews import GetTrendingNews, GetStockNews, GetTodayNews
from SlotsFill import slotfill
import settings.store
from GetStockPredictions import stock_predictions as predictor
from datetime import date
import math
import numpy as np
import datetime

"""
Date (String)
-----
Dateforintent = context.user_data['DATE']

"""
def Finance_General_Whatcanido(utterance, context):
	replies = ["this is my reply"]
	slots = slotsdetection(utterance)
	print("slot : ", slots)
	return random.choice(replies)

def Finance_WatchlistE_Clear(utterance, context):
	replies = ["this is my reply"]
	slots = slotsdetection(utterance)
	print("slot : ", slots)
	return random.choice(replies)


def Finance_WatchlistE_Drop(utterance, context):
	replies = ["this is my reply"]
	slots = slotsdetection(utterance)
	print("slot : ", slots)	
	return random.choice(replies)


def Finance_WatchlistE_Add(utterance, context):
	replies = ["this is my reply"]
	slots = slotsdetection(utterance)
	print("slot : ", slots)
	return random.choice(replies)



def Finance_General_CurrentPrice(utterance, context):
	replies = ["this is my reply"]
	slots = slotsdetection(utterance)
	print("slot : ", slots)
	return random.choice(replies)



def Finance_General_Whatcanido(utterance, context):
	replies = [["I can help you with Technology Stock price direction predictions and the stock news itself or any trending technology stock news or topics"],
	           ["I can make sentiment predictions on Technology Stock price direction and retrieve stock news and trending topics"]]
	slots = slotsdetection(utterance)
	print("slot : ", slots)
	return random.choice(replies)


def Finance_News_Trending(utterance, context):
	slots = slotsdetection(utterance)
	print("slot : ", slots)
	print("stockname : ", slots.get('stockname'))

	if not bool(slots):
		reply = slotfill.stockname()
	else:
		results = GetTrendingNews.GetAnswer()
		replies = [["The Trending news are :"], ["The following are the trending news :"], ["Found trending news :"], ["Are you interested in the following trending news?"]]
		reply   = random.choice(replies)
		reply.append("\n\n")
		reply.append(results)
	return reply

def Finance_News_Watchlist(utterance, context):
	replies = ["this is my reply"]
	slots = slotsdetection(utterance)
	print("slot : ", slots)
	return random.choice(replies)


def Finance_News_Today(utterance, context):
	today = context.user_data['DATE']
	print("Date : ", today)

	if today == "":
		reply = slotfill.todayNews()
	else:
		results = GetTodayNews.GetAnswer(today)
		replies = [["The top 3 latest news are :"], ["The following are top 3 today news :"], ["Found top 3 latest news :"], ["Are you interested in the following top 3 latest news?"]]
		reply   = random.choice(replies)[0]
		reply += "\n\n"
		reply += results
		reply = "".join(reply)
	return reply


def Finance_News_Stock(utterance, context):
	print(settings.store)
	countlength = len(utterance.split())
	if countlength == 1:
		utterance = "for " + utterance
	slots = slotsdetection(utterance)
	print("slot : ", slots)
	print("stockname : ", slots.get('stockname'))
	if not bool(slots["stockname"]):
		reply = slotfill.stockname()
	else:
		stockName = slots.get('stockname')
		results   = GetStockNews.GetAnswer(stockName)
		replies   = [["The Stock news are :"], ["The following are the Stock news :"], ["Found the Stock news :"], ["Are you interested in the following Stock news?"]]
		reply     = random.choice(replies)
		reply.append("\n\n")
		reply.append(results)
	return (reply)


def Finance_Predictions_Sentiments_SingleStock(utterance, context):
	if context and 'DATE' in context.user_data:
		try:
			today_date = datetime.datetime.strptime(context.user_data["DATE"], '%m/%d/%Y').date()
		except Exception as e:
			print(e)
			return 'Please enter date in format mm/dd/yyyy eg: 01/13/2020'
		print(f'today:{today_date}')
	else:
		return 'No date set in context. Please use /date command to set date'
	slots = slotsdetection(utterance)
	print("slot : ", slots)
	if not slots or not slots['stockname']:
		return slotfill.stockname()
	else:
		stock_slot = slots['stockname']
		cols = ['bearish_score_mean', 'bullish_score_mean']
		sentiments = predictor.get_values(cols=cols, ticker=stock_slot, rdate=today_date).values.flatten().tolist()
		def sentiment_format(r): return str(math.ceil(r * 10000) / 100.0) + '%'
		bear_sent, bull_sent = [sentiment_format(s)for s in sentiments]
		sentiment_label = 'Bullish' if np.argmax(sentiments) else 'Bearish'
		replies = [f'The sentiment for {stock_slot} is {bear_sent} bearish and {bull_sent} bullish',
				   f'{stock_slot} is {bull_sent} bullish',
				   f'Overall sentiment for {stock_slot} is currently {sentiment_label}']
		return random.choice(replies)

def Finance_Predictions_Sentiments_Watchlist(utterance, context):
	replies = ["this is my reply"]
	slots = slotsdetection(utterance)
	print("slot : ", slots)
	return random.choice(replies)

def Finance_Predictions_Price_SingleStcok(utterance, context):
	if context and 'DATE' in context.user_data:
		try:
			today_date = datetime.datetime.strptime(context.user_data["DATE"], '%m/%d/%Y').date()
		except Exception as e:
			print(e)
			return 'Please enter date in format mm/dd/yyyy eg: 01/13/2020'
		print(f'today:{today_date}')
	else:
		return 'No date set in context. Please use /date command to set date'
	slots = slotsdetection(utterance)
	print("slot : ", slots)

	if not slots or 'stockname' not in slots:
		return slotfill.stockname()
	else:
		period_slot = slots['numberofdays'] if slots and 'numberofdays' in slots else None
		# period_slot = random.choice([1, 3, 5, None])
		stock_slot = slots['stockname']
		col_dict = {1: '1_day_return',
					3: '3_day_return',
					5: '5_day_return'}
		cols = [col_dict[period_slot]] if period_slot else list(col_dict.values())
		return_list = predictor.get_values(cols=cols, ticker=stock_slot, rdate=today_date).values.flatten().tolist()
		col_names_str = ", ".join([c[:-6].replace('_', ' ').rstrip() for c in cols])
		return_format = lambda r: str(math.ceil(r * 100) / 100.0) + '%'
		returns_str = ', '.join([return_format(r) for r in return_list])

		replies = [f'The {col_names_str} returns predicted for {stock_slot} are {returns_str}',
				   f'The predicted price movements for {col_names_str} is {returns_str}',
				   f'Here are the predicted prices for {col_names_str}: {returns_str}']

		return random.choice(replies)

def Finance_Predictions_Price_Bearish(utterance, context):
	if context and 'DATE' in context.user_data:
		try:
			today_date = datetime.datetime.strptime(context.user_data["DATE"], '%m/%d/%Y').date()
		except Exception as e:
			print(e)
			return 'Please enter date in format mm/dd/yyyy eg: 01/13/2020'
		print(f'today:{today_date}')
	else:
		return 'No date set in context. Please use /date command to set date'
	slots = slotsdetection(utterance)
	print("slot : ", slots)
	df_subset = predictor.get_values(cols=['bearish_score_mean', 'ticker'], rdate=today_date)
	symbols = df_subset.sort_values(by='bearish_score_mean', ascending=False)['ticker'][:3].to_list()
	symbols_str = ", ".join(symbols)
	replies = [f'The most bearish stocks are {symbols_str}',
			   f'Here are the most bearish stocks {symbols_str}',
			   f'{symbols_str} are the most bearish stocks in the market currently']
	return random.choice(replies)

def Finance_Predictions_Price_Bullish(utterance, context):
	if context and 'DATE' in context.user_data:
		try:
			today_date = datetime.datetime.strptime(context.user_data["DATE"], '%m/%d/%Y').date()
		except Exception as e:
			print(e)
			return 'Please enter date in format mm/dd/yyyy eg: 01/13/2020'
		print(f'today:{today_date}')
	else:
		return 'No date set in context. Please use /date command to set date'
	slots = slotsdetection(utterance)
	print("slot : ", slots)
	df_subset = predictor.get_values(cols=['bullish_score_mean', 'ticker'], rdate=today_date)
	symbols = df_subset.sort_values(by='bullish_score_mean', ascending=False)['ticker'][:3].to_list()
	symbols_str = ", ".join(symbols)
	replies = [f'The most bullish stocks are {symbols_str}',
			   f'Here are the most bullish stocks {symbols_str}',
			   f'{symbols_str} are the most bullish stocks in the market currently']
	return random.choice(replies)

