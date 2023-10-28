import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.tools import BaseTool
import requests
import json
from langchain.schema import SystemMessage
import streamlit as st
from src.utils import get_transcript_from_video_url, get_summary, get_comments_dataframe
import time
import pandas as pd

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def resize_video(DEFAULT_WIDTH, url):
    width = DEFAULT_WIDTH

    width = max(width, 0.01)
    side = max((100 - width) / 2, 0.01)

    _, container, _ = st.columns([side, width, side])
    container.video(data=url)


@st.cache_resource
def get_custom_summary(_docs, _llm, custom_prompt, chain_type):
    custom_prompt = custom_prompt + """:\n\n {text}"""

    map_prompt = """
    Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:
    """
    MAP_PROMPT = PromptTemplate(template=map_prompt, input_variables=["text"])

    COMBINE_PROMPT = PromptTemplate(template=custom_prompt, input_variables=["text"])

    if chain_type == "map_reduce":
        chain = load_summarize_chain(
            _llm,
            chain_type=chain_type,
            map_prompt=MAP_PROMPT,
            combine_prompt=COMBINE_PROMPT,
        )
    else:
        chain = load_summarize_chain(_llm, chain_type=chain_type)

    output = chain.run(_docs)
    return output


@st.cache_data
def get_corpus(url, chunk_size, chunk_overlap):
    result = get_transcript_from_video_url(url)
    docs = result[0].page_content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    corpus = text_splitter.create_documents([docs])
    return corpus


def main():
    st.markdown(
        "<h1 style='text-align: center; color: red;'>Summarize YouTube Video </>",
        unsafe_allow_html=True,
    )
    # st.markdown("<h3 style='text-align: center; color: grey;'>Built by <a href='https://github.com/pjeena'>Piyush </a></h3>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='text-align: center; color:red;'>Enter your URL and Prompt👇</h4>",
        unsafe_allow_html=True,
    )

    chain_type = st.sidebar.selectbox("Chain Type", ["map_reduce", "stuff", "refine"])
    chunk_size = st.sidebar.slider(
        "Chunk Size", min_value=100, max_value=5000, step=100, value=2000
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap", min_value=100, max_value=10000, step=100, value=200
    )

    input_url = st.text_input("Enter a URL")
    input_user_prompt = st.text_input("Enter a prompt")

    temperature = st.sidebar.number_input(
        "ChatGPT Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.0
    )

    llm = OpenAI(temperature=temperature, openai_api_key=OPENAI_API_KEY)

    if input_url != "":
        if st.button("Summarize"):
            corpus = get_corpus(
                url=input_url, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            with st.sidebar:
                resize_video(DEFAULT_WIDTH=120, url=input_url)
            #            summarized_video = get_custom_summary(
            #                _docs=corpus,
            #                _llm=llm,
            #                custom_prompt=input_user_prompt,
            #                chain_type=chain_type,
            #            )
            summarized_video = "yeah"
            video_details = get_transcript_from_video_url(input_url)[0].metadata
            st.markdown("**:red[Title]** : *{}*".format(video_details["title"]))
            st.markdown("**:red[Summary]** : *{}*".format(summarized_video))

    else:
        st.markdown("")


if __name__ == "__main__":
    main()
