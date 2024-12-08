import requests                                                      #Scrapes HTML code from URL
from bs4 import BeautifulSoup                                        #Removes HTML syntax and leaves plain text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #Preforms sentiment analysis on text
import pandas as pd                                                  #Data manipulation library
#import streamlit as st                                              #TODO where is streamlit, get this installed

#headers={"accept-language": "en-US,en;q=0.9", "accept-encoding": "gzip, deflate, br", "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"}
headers={"accept-language":"en-US,en;q=0.9", "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"}
#Above are common headers used by websites. This allows requests to get HTML from websites. 
#BUG: The "accept-encoding" header list was preventing some websites from being read.

sentiment = SentimentIntensityAnalyzer() #Sentiment Analyzer Object to preform analysis

with open('total_sent.txt', 'w') as total_sent: #Opens 'total_sent.txt' to write to. Opening this way allows the file to auto close at the end of the block
    with open('url_list.txt', 'r') as url_list: #Opens 'url_list.txt' to read from. Opening this way allows the file to auto close
        count = 0                               #count tracks the number of successful reads from the url list to calculate the average sentiment
        pos_sentiment = 0                       #pos_sentiment adds all positive sentiments to calculate average
        neg_sentiment = 0                       #neg_sentiment adds all negatibe sentiments to calculate average
        for target_url in url_list:                                                             #Loops over the url list
            try:                                                                                #Try to get the response using requests, if something goes wrong we assume the website did something unexpected
                response = requests.get(target_url[:-1], verify = True, headers = headers)      #gets the html from target_url[:-1] using the hearders list (Target url is only line, [:-1] removes the newline char ), verify makes sure the website is safe
                soup_text = BeautifulSoup(response.text, 'html.parser').getText(strip = True)   #Parse response text using the html parser, remove blank space where html code was and return as str
                sent_text = sentiment.polarity_scores(soup_text)                                #Preform analysis and store in sent_text    
                total_sent.write(target_url)                                                    #Write url to file (\n already included)
                total_sent.write(str(sent_text) + '\n')                                         #Write sent_text to file with \n added
                neg_sentiment += sent_text["neg"]                                               #Add negative sentiment to aggregate
                pos_sentiment += sent_text["pos"]                                               #Add positive sentiment to aggregate
                count += 1                                                                      #increment count

                print(count)                                                                    #Print statements, can be commented out
                print(target_url[:-1])
                print(sent_text)
            except:
                print("\nException has occured when requesting from url: " + target_url)        #if there is in a failure in any of that is is our error statement
        avg_neg = neg_sentiment / count                                                                     #Calulates average negative sentiment
        avg_pos = pos_sentiment / count                                                                     #Calculates average positive sentiment
        avg_sent = "{'avg_neg': " + str(round(avg_neg, 4)) + ", 'avg_pos': " + str(round(avg_pos, 4)) + "}" #Stores the averages in a str

        total_sent.write(avg_sent)                                                                          #Writes averages to total_sent file
        print(avg_sent)                                                                                     #prints averages, can be commented out

    
