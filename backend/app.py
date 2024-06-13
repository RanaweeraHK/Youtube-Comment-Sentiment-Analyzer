from flask import Flask, request, jsonify, send_file ,render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import googleapiclient.discovery
import pandas as pd
import spacy
import re
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

# Load environment variables
DEVELOPER_KEY = os.getenv('YOUTUBE_API_KEY', 'AIzaSyA4-dDoBsXpMbyfvXEXkE3ZWZTqytHsosE')

# Instantiate the model
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Set up the YouTube API client
api_service_name = "youtube"
api_version = "v3"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(filtered_tokens)

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt', truncation=True, max_length=512)
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

def extract_video_id(url):
    regex = r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/|youtube\.com\/(?:shorts\/|embed\/|v\/|vi\/|u\/\w\/|watch\?v=|watch\?vi=|&v=|&vi=))([^\"&?\/\s]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    else:
        return None


def get_comments(youtube, video_id, max_results=100):
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

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    video_url = data.get('video_url')

    # Extract video ID from URL
    video_id = extract_video_id(video_url)
    
    if not video_id:
        return jsonify({'error': 'Invalid video URL'}), 400

    # Fetch comments
    try:
        comments = get_comments(youtube, video_id)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    if not comments:
        return jsonify({'error': 'No comments found'}), 404

    # Create DataFrame and preprocess comments
    df = pd.DataFrame(comments, columns=['Review'])
    df['filtered_review'] = df['Review'].apply(preprocess)

    # Perform sentiment analysis
    df['sentiment'] = df['filtered_review'].apply(lambda x: sentiment_score(x))

    # Get sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()

    # Group sentiment scores into categories
    positive_count = sum(sentiment_counts[score] for score in [4, 5] if score in sentiment_counts)
    neutral_count = sentiment_counts.get(3, 0)
    negative_count = sum(sentiment_counts[score] for score in [1, 2] if score in sentiment_counts)

    # Create lists for the pie chart
    labels = ['Negative', 'Neutral', 'Positive']
    colors = ['#FFC154', '#EC6B56', '#47B39C']
    count_values = [negative_count, neutral_count, positive_count]

    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(count_values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    
    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')
