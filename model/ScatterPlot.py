import os                       #Allows manipulation of file paths
import pandas as pd             #Data manipulation library
import streamlit as st          #UI Library
import matplotlib.pyplot as plt #Plots graphs

subdirectory = "..\PLCWebScraper"                   #Directory of csv file
file_name = "total_sent.csv"                        #Name of csv file
file_path = os.path.join(subdirectory, file_name)   #Making path to file
total_sent = pd.read_csv(file_path)                 #Reading file to dataframe

def draw():
    st.title("Sentiment Analysis")

    fig, plot = plt.subplots(figsize=(7, 5))

    plot.set_xlim(0, total_sent['pos'].max() + 0.01)
    plot.set_ylim(0, total_sent['neg'].max() + 0.01)

    plot.set_title('Market Blogs Sentiments')
    plot.set_xlabel('Positive Sentiment Score')
    plot.set_ylabel('Negative Sentiment Score')

    plot.scatter(total_sent['pos'].tolist(), total_sent['neg'].tolist(), s=5)
    avg_pos = total_sent['pos'].mean()
    avg_neg = total_sent['neg'].mean()
    print("avg_pos = " + str(avg_pos) + " avg_neg = " + str(avg_neg))

    st.pyplot(fig)