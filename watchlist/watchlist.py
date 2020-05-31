# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:12:42 2020

@author: Donal
"""
import glob
import numpy as np
from numpy import genfromtxt
import pandas as pd

class UserWatchlist:
    def __init__(self,identity):
        self.identity = identity
        
    def getWatchlist(self):
        try:
            file = glob.glob("watchlist\\userfiles\\"+str(self.identity)+".csv")
            self.watchlist = list(genfromtxt(file[0], delimiter=',',dtype=str))
            return self.watchlist
        except:
            self.watchlist = []
            np.savetxt("watchlist\\userfiles\\"+str(self.identity)+".csv",self.watchlist, delimiter=',',fmt='%s')
            return self.watchlist
    
    def addWatchlist(self, stock):
        existing = []
        added = []
        for i in stock:
            try: 
                self.watchlist.index(i)
                existing.append(i)
            except:
                self.watchlist.append(i)
                added.append(i)
        df = pd.DataFrame(self.watchlist)
        df.to_csv("watchlist\\userfiles\\"+str(self.identity)+".csv" )
        return existing, added, self.watchlist


