#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:10:19 2017

@author: rhdzmota
"""

# %% Imports 

import numpy as np
import pandas as pd
import datetime as dt

import matplotlib.pyplot as plt

# %% 
def wakeUpPlease():
    """
    Please wakeup!
    """
    import requests
    try:
        r = requests.get("https://mxquants-datarepo.herokuapp.com")
        return True 
    except:
        return False 
    
def downloadPrices(stock_name=None,columns=["Timestamp","AdjPrices"]):
    import requests 
    
    # check for type 
    if type(stock_name) != str :
        print("Error: stockname (str) must be provided." )
        return None
    
    
    # correct columnnames 
    def correctColumnNames(cols):
        correct = {"Timestamp":"date","AdjPrices":"unadjclose"}
        def try2correct(x):
            try: 
                return correct[x]
            except:
                return x
        return [try2correct(c) for c in cols]
    
    # url for request 
    url = """https://mxquants-datarepo.herokuapp.com/DefaultDownload"""
    parameters = {"columns":str(correctColumnNames(columns)),"stock_name":stock_name.upper(),"pwd":"mxquants-rules"}
    
    # make request 
    r = requests.get(url,params=parameters).json()
    
    try:
        flag = not r["error"]
    except:
        flag = True
    
    if not flag:
        print("Error: data coudn't be downloaded. Check your internet connection or variables stock_name, columns.")
        return None 
    data = r["data"]
    
    # get date as string from timestamp 
    def timestamp2Date(x):
        return dt.datetime.fromtimestamp(int(x)).strftime("%d-%m-%Y")
    
    data["date"] = list(map(timestamp2Date,data["date"]))
    
    # create dataframe 
    df = pd.DataFrame(data)
    df.columns = columns 
    
    return df 
    
# %% Calculate returns

def getReturns(price_vector,_type="log"):
    price_vector = np.asarray(price_vector)
    if _type != "log":
        return price_vector[1:]/price_vector[:-1] - 1
    return np.log(price_vector[1:]/price_vector[:-1] )
    
    
# %% Main function 

def main():
    wakeUpPlease()
    
# %% 

if __name__ == '__main__':
    main()

# %% 


# %% 
    