# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:39:48 2020

@author: Donal
"""
import random
def stockname():
    replies = [['For which stock?'],
             ['Which stock is that?'],
             ['Can I get the name of the stock please?'],
             ['Which stock are you refering to?'],
             ['What stock are you refering to?']]
    return random.choice(replies)
    
    
def numberofdays():
    replies = [['For how many days?'],
               ['How many days?']]
    return random.choice(replies)


def stocksymb():
    replies = [['For which stock?'],
             ['Which stock is that?'],
             ['Can I get the name of the stock please?'],
             ['Which stock are you refering to?'],
             ['What stock are you refering to?']]
    return random.choice(replies)

def todayNews():
    replies = [['Are you looking for today news?'],
             ['Are you searching for latest news?'],
             ['Latest news ?'],
             ['Today news']]
    return random.choice(replies)
    