import os                       #Allows manipulation of file paths
import numpy as np              #Number manipulation library
import statistics as stats      #Pref
import pandas as pd             #Data manipulation library
import streamlit as st          #UI Library
import matplotlib.pyplot as plt #Plots graphs

subdirectory = "..\PLCWebScraper"                   #Directory of csv file
file_name = "total_sent.csv"                        #Name of csv file
file_path = os.path.join(subdirectory, file_name)   #Making path to file
total_sent = pd.read_csv(file_path)                 #Reading file to dataframe

def normal_curve(bins, mean, stdev):
    return (1 / (np.sqrt(2 * np.pi) * stdev)) * np.exp(-0.5 * (1 / stdev * (bins - mean))**2)

def draw():
    #Storing useful values
    avg_pos = total_sent['pos'].mean()
    avg_neg = total_sent['neg'].mean()
    stdev_pos = stats.stdev(total_sent['pos'])
    stdev_neg = stats.stdev(total_sent['neg'])
    #Number of bins for histograms
    num_bins = 50

    #Page Title
    st.title("Sentiment Analysis")
    #Scatter Plot:
    #Size
    plt.figure(figsize=(11, 7))
    #Limits
    plt.xlim(0, total_sent['pos'].max() + 0.01)
    plt.ylim(0, total_sent['neg'].max() + 0.01)
    #Titles
    plt.title('Market Blogs Sentiments')
    plt.xlabel('Positive Sentiment Score')
    plt.ylabel('Negative Sentiment Score')
    #Fill with data points
    plt.scatter(total_sent['pos'].tolist(), total_sent['neg'].tolist(), s=5)
    #Draw Scatter Plot
    st.pyplot(plt)

    #TODO: TABLE OF IMPORTANT VALUES

    #Positive Histogram:
    plt.figure(figsize=(11, 7))
    plt.title('Positive Sentiment Distribution')
    plt.xlabel('Positive Sentiment Score')
    plt.ylabel('Frequency')
    plt.xlim(total_sent['pos'].min() - 0.01, total_sent['pos'].max() + 0.01)
    n, bins, patches = plt.hist(total_sent['pos'].tolist(), bins=num_bins)
    plt.plot(bins, normal_curve(bins, avg_pos, stdev_pos), '--', color='black')
    st.pyplot(plt)

    #Negative Histogram:
    plt.figure(figsize=(11, 7))
    plt.title('Negative Sentiment Distribution')
    plt.xlabel('Negative Sentiment Score')
    plt.ylabel('Frequency')
    plt.xlim(total_sent['neg'].min() - 0.01, total_sent['neg'].max() + 0.01)
    n, bins, patches = plt.hist(total_sent['neg'].tolist(), bins=num_bins)
    plt.plot(bins, normal_curve(bins, avg_neg, stdev_neg), '--', color='black')
    st.pyplot(plt)
