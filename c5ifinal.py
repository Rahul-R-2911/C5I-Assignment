import streamlit as st                        # Dashboard
import assemblyai as aai                      # Audio to Text transcription
import spacy                                  # Text preprocessing
from gensim import corpora, models            # LDA for Topic Extraction
import string
from transformers import pipeline             # HuggingFace model for Sentiment analysis
from summa import summarizer                  # Text summarization model
from moviepy.editor import VideoFileClip      # MP4 to MP3 conversion
import os

# Initializations
aai.settings.api_key = "f70cf7708546487d8fa7e6d3f4d9022f"
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis")

# Function to convert MP4 to MP3
def convert_mp4_to_mp3(file_path):
    video = VideoFileClip(file_path)
    mp3_file = "temp_audio.mp3"
    video.audio.write_audiofile(mp3_file)
    return mp3_file

# Function to get audio transcript using AssemblyAI
def audio_to_text(file, file_extension):
    if file_extension == "mp4":
        with open("temp_video.mp4", "wb") as f:
            f.write(file.read())
        file_path = convert_mp4_to_mp3("temp_video.mp4")
    else:
        file_path = "temp_audio.mp3"
        with open(file_path, "wb") as f:
            f.write(file.read())

    config = aai.TranscriptionConfig(speaker_labels=True, auto_highlights=True)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(
        file_path,
        config=config
    )

    # Remove older files if exists
    if os.path.exists("temp_video.mp4"):
        os.remove("temp_video.mp4")
    if os.path.exists("temp_audio.mp3"):
        os.remove("temp_audio.mp3")

    if transcript.status == aai.TranscriptStatus.error:
        return f"Error: {transcript.error}", None, None, None, None

    transcript_text = ""
    for line in transcript.utterances:
        transcript_text += f"Speaker {line.speaker}: {line.text}\n\n"

    # Extract important topics
    imp_topics = ""
    if transcript.auto_highlights:
        for result in transcript.auto_highlights.results:
            imp_topics += f"{result.text}, "
    
    # Extract speaker information
    speaker_data = extract_speaker_data(transcript_text)
    
    # Extract sentiment of text
    sentiment = analyze_sentiment(transcript_text)

    # Extract summary of text
    summary = summarise(transcript_text)
    
    return transcript_text, imp_topics, speaker_data, sentiment, summary

# Function for text preprocessing
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in string.punctuation and not token.is_stop]
    return tokens

# Function to extract speaker data
def extract_speaker_data(transcript):
    word_count = len(transcript.split())
    
    speaker_durations = {}
    for line in transcript.splitlines():
        if line.startswith("Speaker"):
            speaker = line.split(":")[0]
            words = len(line.split())
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + words
    return [word_count, speaker_durations]

# Function for sentiment analysis
def analyze_sentiment(text):
    sentiment_result = sentiment_analyzer(text[:512])
    sentiment = sentiment_result[0]["label"]
    return sentiment

# Function for summarization
def summarise(text):
    summary = summarizer.summarize(text, ratio=0.2)
    return summary

def main():
    st.title("Conversational Insights Platform")
    st.write("Application to extract the transcript and additional details of the conversation from an audio or video file.")

    uploaded_file = st.file_uploader("Choose file", type=["mp3", "mp4"])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        st.audio(uploaded_file, format=f"audio/{file_extension}")
        with st.spinner("Transcribing..."):
            transcript_text, topics, speaker_data, sentiment, summary = audio_to_text(uploaded_file, file_extension)
            
            st.text_area("Transcript", transcript_text, height=400)
            st.text_area("Topics", topics, height=200)
            st.text_area("Speaker Data", f"Total word count: {speaker_data[0]}\nIndividual count: {speaker_data[1]}", height=100)
            st.text_area("Sentiment Analysis", f"The conversation is {sentiment}.", height=50)
            st.text_area("Summary",summary, height=200)

if __name__ == "__main__":
    main()
