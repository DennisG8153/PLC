import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

subdirectory = "PLCWebScraper"
file_name = "total_sent.csv"
file_path = os.path.join(subdirectory, file_name)

#pos_sentiment = 0                            #pos_sentiment adds all positive sentiments to calculate average
#neg_sentiment = 0                            #neg_sentiment adds all negatibe sentiments to calculate average
#neg_sentiment += sent_text["neg"]                                              #Add negative sentiment to aggregate
#pos_sentiment += sent_text["pos"]                                              #Add positive sentiment to aggregate
#avg_neg = neg_sentiment / count                                                                     #Calulates average negative sentiment
#avg_pos = pos_sentiment / count                                                                     #Calculates average positive sentiment
#avg_sent = "{'avg_neg': " + str(round(avg_neg, 4)) + ", 'avg_pos': " + str(round(avg_pos, 4)) + "}" #Stores the averages in a str


data_points = [] 
with open(file_path, "r") as total_sent:
    for i, line in enumerate(total_sent):
        if i % 2 == 1:
            
           
            #print(data_points)

"""
st.title("Sentiment Analysis")

fig, plot = plt.subplots(figsize=(7, 5))

plot.set_xlim(0, 100)
plot.set_ylim(0, 100)

plot.set_title('Sentiment Analysis')
plot.set_xlabel('Positive Sentiment Score')
plot.set_ylabel('Negative Sentiment Score')

st.pyplot(fig)
"""