import unicodedata
import requests                                                      #Scrapes HTML code from URL
from bs4 import BeautifulSoup                                        #Removes HTML syntax and leaves plain text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #Preforms sentiment analysis on text
import pandas as pd                                                  #Data manipulation library
#import streamlit as st                                              #TODO where is streamlit, get this installed

target_url = 'https://blog.marketresearch.com/topic/emerging-markets'
#headers={"accept-language":"en-US,en;q=0.9", "accept-encoding":"gzip, deflate, br", "accept-charset":"utf-8", "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"}
headers={"accept-language": "en-US,en;q=0.9", "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"}
response = requests.get(target_url, verify=True, headers=headers) 
soup = BeautifulSoup(response.text, 'html.parser')
#For the above block:
#Currently only uses one URL
#Common headers used by websites
#response is the requested HTML from the URL
#soup cleans the text

f = open ("out.txt", "w")              #opens a file
fullText = soup.get_text(strip = True) #removes blank text
#fullText = response.text
#fullText = unicodedata.normalize('NFKD', fullText).encode('ascii', 'ignore').decode('ascii') #If text is unicode, this converts to ascii
f.write(fullText)                      #writes the text to the file
f.close()                              #closes the file

sentiment = SentimentIntensityAnalyzer()       #sentiment object to preform analysis
sentText = sentiment.polarity_scores(fullText) #analyzes sentiment and creates a string
print(sentText)                                #prints the string to console
