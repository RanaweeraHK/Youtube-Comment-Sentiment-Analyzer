from fastapi import FastAPI, HTTPException
from typing import List
import googleapiclient.discovery
import re

app = FastAPI()

# Set up the YouTube API client
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "<YOUR_YOUTUBE_API_KEY>"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

# Define the function to get comments from a YouTube video
def get_comments(video_id, max_results):
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

def extract_video_id(url):
    regex = r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^\"&?\/\s]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    else:
        return None

@app.get("/get_comments/{video_id}/{max_results}", response_model=List[str])
async def fetch_comments(video_id: str, max_results: int):
    try:
        comments = get_comments(video_id, max_results)
        return comments
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
