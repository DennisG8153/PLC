import requests                                                      #Scrapes HTML code from URL
from bs4 import BeautifulSoup                                        #Removes HTML syntax and leaves plain text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #Preforms sentiment analysis on text
import pandas as pd                                                  #Data manipulation library
import csv                                                           #Read and write from CSV files

#headers={"accept-language": "en-US,en;q=0.9", "accept-encoding": "gzip, deflate, br", "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"}
headers={"accept-language":"en-US,en;q=0.9", "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"}
#Above are common headers used by websites. This allows requests to get HTML from websites. 
#BUG: The "accept-encoding" header list was preventing some websites from being read.

sentiment = SentimentIntensityAnalyzer() #Sentiment Analyzer Object to preform analysis

with open('url_list.txt', 'r') as url_list:                     #Opens 'total_sent.txt' to write to. Opening this way allows the file to auto close at the end of the block
    with open('total_sent.csv', 'w', newline='') as total_sent: #Opens 'url_list.txt' to read from. Opening this way allows the file to auto close
        writer = csv.writer(total_sent)                         #Creates the csv writer
        fields = ['pos', 'neg', 'url']                          #Creates fields for colomns of the CSV file
        writer.writerow(fields)                                 #Writes the fields as the first line of the CSV file
        print(fields)                                           #Prints to the terminal
        count = 0                                               #count tracks the number of successful reads from the url list to calculate the average sentiment
        for target_url in url_list:                                                             #Loops over the url list
            try:                                                                                #Try to get the response using requests, if something goes wrong we assume the website did something unexpected
                target_url = target_url.rstrip("\n")                                            #Removes new line chars, TODO: There is likely a better way to do this
                response = requests.get(target_url, verify = True, headers = headers)           #gets the html from target_url[:-1] using the hearders list (Target url is only line, [:-1] removes the newline char ), verify makes sure the website is safe
                soup_text = BeautifulSoup(response.text, 'html.parser').getText(strip = True)   #Parse response text using the html parser, remove blank space where html code was and return as str
                sent_text = sentiment.polarity_scores(soup_text)                                #Preform analysis and store in sent_text    
                data = [sent_text["pos"], sent_text["neg"], target_url]                         #Adds the pos and neg sentiment to a Dataframe(?) object
                writer.writerow(data)                                                           #Writes the pos and neg sentiment percentages and the associated URL to the CSV
                count += 1                                                                      #increment count

                print(str(count) + " " + str(response))                                                                   #Print statements, can be commented out
                print(data)
            except:
                print("\nException has occured when requesting from url: " + target_url)        #if there is in a failure in any of that is is our error statement