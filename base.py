from transformers import AutoTokenizer , AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import googleapiclient.discovery
import pandas as pd
import numpy as np
import spacy
from matplotlib import pyplot as plt

# Instantiate the model
# https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Set up the YouTube API client
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyA4-dDoBsXpMbyfvXEXkE3ZWZTqytHsosE"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

# Define the video ID and the number of comments to retrieve per request
video_id = "SIm2W9TtzR0"
max_results = 100

# Function to get comments from a YouTube video
def get_comments(youtube, video_id, max_results):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results
    )
    response = request.execute()

    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        if 'nextPageToken' in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                pageToken=response['nextPageToken']
            )
            response = request.execute()
        else:
            break

    return comments

# Get the comments and store them in a list
comments = get_comments(youtube, video_id, max_results)

# Create a DataFrame from the list of comments
df = pd.DataFrame(comments, columns=['Review'])

nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    doc= nlp(text)
    filtered_tokens = []

    for token in doc:
        if not token.is_stop and not token.is_punct:
            filtered_tokens.append(token.text)
            
    return " ".join(filtered_tokens)

df['filtered_review'] = df['Review'].apply(preprocess)

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

df['sentiment'] = df['filtered_review'].apply(lambda x: sentiment_score(x[:512]))


sentiment_counts = df['sentiment'].value_counts().sort_index()
# Create a list of all possible sentiment scores (1 to 5)
all_sentiment_scores = [1, 2, 3, 4, 5]

# Create a list of corresponding counts or zeros if no count exists
count_values = [sentiment_counts.get(score, 0) for score in all_sentiment_scores]

plt.barh(all_sentiment_scores, count_values)  # Use barh for horizontal bar chart
plt.xlabel('Number of Reviews')
plt.ylabel('Sentiment Score')
plt.title('Distribution of Sentiment Scores')

# Set y-axis ticks to all possible sentiment scores
plt.yticks(all_sentiment_scores)

# Set x-axis ticks based on the range of counts (minimum to maximum count)
plt.xticks(range(min(count_values), max(count_values) + 1))

plt.show()
