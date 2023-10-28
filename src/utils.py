import pandas as pd
import numpy as np
import xmltodict
import yaml
import os
import glob
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import googleapiclient.discovery
from furl import furl
import re
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

def get_transcript_from_video_url(url_link):
    loader = YoutubeLoader.from_youtube_url(url_link, add_video_info=True)
    result = loader.load()
#    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
#    corpus = text_splitter.split_documents(result)
    return result


def get_summary(transcript_text):
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    result = chain.run(transcript_text)
    return result




def get_video_id(video_url):
    if len(video_url):
        f = furl(video_url)
        if len(f.args) and 'v' in list(f.args.keys()):
            return f.args['v']
    return None

def get_youtube_client():
    return googleapiclient.discovery.build(
        API_SERVICE_NAME, API_VERSION, developerKey = YOUTUBE_API_KEY)



def get_video_info(video_id):
    request = get_youtube_client().videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    )
    return request.execute()


def get_video_title(video_info):
    return video_info['items'][0]['snippet']['title']

def get_channel_title(video_info):
    return video_info['items'][0]['snippet']['channelTitle']


def get_comments_by_video(video_id, nextPageToken=None):
    request = get_youtube_client().commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        maxResults=100,
        pageToken=nextPageToken
    )
    return request.execute()

def get_all_comments(video_id, max_comments):
    comments_list = []
    nextPageToken = None

    while True:
        comments = get_comments_by_video(video_id, nextPageToken)
        comments_list.extend(comments['items'])

        if len(comments_list) >= max_comments or 'nextPageToken' not in comments:
            break

        nextPageToken = comments['nextPageToken']
    return comments_list



def get_comments_dataframe(VIDEO_URL,MAX_COMMENTS=10*100):
    video_id = get_video_id((VIDEO_URL))
    video_info = get_video_info(video_id)

    video_title = get_video_title(video_info)
    channel_title = get_channel_title(video_info)

    clean_title = re.sub(r'[^\w\s]+', '', video_title)
    filename = f'{channel_title} - {clean_title[:30]} - {video_id}'

    # Top Level Comments
    comments = get_all_comments(video_id, MAX_COMMENTS)
    df_comments = pd.json_normalize(comments)
    df_comments.drop(columns=['kind', 'etag', 'id', 'snippet.videoId', 'snippet.topLevelComment.snippet.channelId'], inplace=True)
    df_comments.columns = df_comments.columns.str.removeprefix('snippet.topLevelComment.').str.removeprefix('snippet.').str.removesuffix('.value').str.removesuffix('.comments')
#    df_comments.to_csv('comments-' + filename + '.csv', index=False)
    return df_comments




if __name__ == "__main__":
    print(OPENAI_API_KEY)

#    corpus = get_transcript_from_video_url(url_link="https://www.youtube.com/watch?v=0CmtDk-joT4&ab_channel=CajunKoiAcademy")
#    output = get_summary(transcript_text=corpus)
    url = 'https://www.youtube.com/watch?v=nBydCvT195k&ab_channel=PaulB'
    df = get_comments_dataframe(VIDEO_URL=url)

    print(df['textOriginal'])
