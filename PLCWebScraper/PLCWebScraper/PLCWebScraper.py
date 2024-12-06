import requests                                                      #Scrapes HTML code from URL
from bs4 import BeautifulSoup                                        #Removes HTML syntax and leaves plain text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #Preforms sentiment analysis on text
import pandas as pd                                                  #Data manipulation library
#import streamlit as st                                              #TODO where is streamlit, get this installed

#headers={"accept-language": "en-US,en;q=0.9", "accept-encoding": "gzip, deflate, br", "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"}
headers={"accept-language":"en-US,en;q=0.9", "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"}
sentiment = SentimentIntensityAnalyzer()

with open('total_sent.txt', 'w') as total_sent:
    with open('url_list.txt', 'r') as url_list:
        count = 0
        pos_sentiment = 0
        neg_sentiment = 0
        for target_url in url_list:
            try:
                response = requests.get(target_url[:-1], verify = True, headers = headers)
                soup_text = BeautifulSoup(response.text, 'html.parser').getText(strip = True)
                sent_text = sentiment.polarity_scores(soup_text)
                total_sent.write(target_url)
                total_sent.write(str(sent_text) + '\n')
                neg_sentiment += sent_text["neg"]
                pos_sentiment += sent_text["pos"]           
                count += 1

                print(count)
                print(target_url[:-1])
                print(sent_text)
            except:
                print("\nException has occured when requesting from url: " + target_url)
        avg_neg = neg_sentiment / count
        avg_pos = pos_sentiment / count
        avg_sent = "{'avg_neg': " + str(round(avg_neg, 4)) + ", 'avg_pos': " + str(round(avg_pos, 4)) + "}"

        total_sent.write(avg_sent)
        print(avg_sent)

    
