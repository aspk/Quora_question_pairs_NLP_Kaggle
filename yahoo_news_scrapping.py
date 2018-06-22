"""
Akhilesh Khope
PhD Candidate
Electrical Engineering
UC Santa Barbara
"""
import sys, os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib import request
from bs4 import BeautifulSoup
import datetime
import dateutil.relativedelta as dr
import numpy as np
import logging

filename = 'E:/news_scrapping/yahoo_finance_news.h5'

def get_constituents(): 
	# source https://pythonprogramming.net/sp500-company-price-data-python-programming-for-finance/
    # URL request, URL opener, read content
    req = request.Request('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    opener = request.urlopen(req)
    content = opener.read().decode() # Convert bytes to UTF-8

    soup = BeautifulSoup(content,"lxml")
    tables = soup.find_all('table') # HTML table we actually need is tables[0] 

    external_class = tables[0].findAll('a', {'class':'external text'})

    tickers = []

    for ext in external_class:
        if not 'reports' in ext:
            tickers.append(ext.string)

    return tickers

def get_data(ticker):
    yahoo_news = requests.get('http://finance.yahoo.com/rss/headline?s='+ticker.lower())
    soup = BeautifulSoup(yahoo_news.text, 'html.parser')
    headlines = [title.text for title in soup.find_all('title')]
    pubdate = [date.text for date in soup.find_all('pubdate')]
    description =[des.text for des in soup.find_all('description')]
    data = list(zip(headlines[1:-1],description[1:],pubdate))
    return data
def main():

    #get tickers of sp500 companies
    print('getting tickers')
    sp500 = get_constituents()
    sp500.append('SPY')

    #put data news data together for all 
    print('getting data')
    data = {}
    count = 0
    for ticker in sp500:
        count+=1
        if count%50==0:
            print(ticker)
            print(count)
        data[ticker] = get_data(ticker)


    # update with daily news
    print('storing data')
    if os.path.exists(filename) == False:
        store = pd.HDFStore(filename) #hdf5 files format is used for large file storage on the the hard drive
        store.close()
    store = pd.HDFStore(filename)   
    for key in data.keys():
        if ("/"+key in store) == False:
            df = pd.DataFrame()
            store[key] = df 
        if data[key]!=[]:
            title,description,pubdate = np.transpose(data[key])
            df = pd.DataFrame()
            df['pubdate'] = pubdate
            df['title'] = title
            df['description'] = description
            df2 = store[key]
            df3 = pd.concat([df,df2])
            df3.reset_index(inplace=True,drop = True)
            df3.drop_duplicates(inplace=True)
            store[key] = df3

    #remember to close the file
    store.close()
    return

if __name__ == "__main__":
    try:
        main()    
    except Exception as e:
        f = open('log.txt', 'w')
        f.write('An exceptional thing happed - %s' % e)
        f.close()