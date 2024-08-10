# Importing Libraries
import os
import yt_dlp
from pydub import AudioSegment
import whisper
from fpdf import FPDF
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
import re
from streamlit_chat import message
import streamlit.components.v1 as components
from markdown import markdown

def sanitize_filename(filename):
    """
    Sanitize a filename by removing or replacing illegal characters.
    
    Args:
        filename (str): The filename to sanitize.
    
    Returns:
        str: The sanitized filename.
    """
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def download_audio(youtube_url, output_path):
    """
    Download audio from a YouTube video and convert it to a sanitized .m4a file.

    Args:
        youtube_url (str): The URL of the YouTube video.
        output_path (str): The directory where the audio file will be saved.

    Returns:
        str: The path to the sanitized .m4a audio file.
    """
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        audio_file = ydl.prepare_filename(info_dict)
        sanitized_audio_file = os.path.join(output_path, sanitize_filename(os.path.basename(audio_file)).replace('.webm', '.m4a').replace('.mp4', '.m4a').replace('.opus', '.m4a'))
        
        video_title = info_dict.get('title', 'Unknown Title')
        st.session_state.video_title = video_title
        
        if os.path.exists(audio_file):
            os.rename(audio_file, sanitized_audio_file)
    
    return sanitized_audio_file

def convert_audio_to_wav(m4a_file):
    """
    Convert an .m4a audio file to a .wav file.

    Args:
        m4a_file (str): The path to the .m4a audio file.

    Returns:
        str: The path to the converted .wav audio file.
    """
    wav_file = m4a_file.replace('.m4a', '.wav')
    sound = AudioSegment.from_file(m4a_file, format="m4a")
    sound.export(wav_file, format="wav")
    return wav_file

def transcribe_audio_whisper(wav_file):
    """
    Transcribe the audio from a .wav file using the Whisper model.

    Args:
        wav_file (str): The path to the .wav audio file.

    Returns:
        str: The transcribed text.
    """
    model = whisper.load_model("base")
    result = model.transcribe(wav_file)
    return result["text"]

def summarize_text_with_chatollama(text):
    """
    Summarize the given text using the ChatOllama model.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    model = ChatOllama(model="llama3.1:8b", temperature=0.7)
    prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"
    summary = model.predict(prompt)
    return summary

def save_text_to_pdf(text, pdf_file):
    """
    Save transcribed text to a PDF file.

    Args:
        text (str): The text to be saved.
        pdf_file (str): The path where the PDF will be saved.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)

    pdf.output(pdf_file)

def load_db(file, chain_type, k):
    """
    Load a PDF file into a vector store, split the text into chunks, and set up a ConversationalRetrievalChain.

    Args:
        file (str): The path to the PDF file.
        chain_type (str): The type of chain to use (e.g., "stuff").
        k (int): The number of documents to retrieve for each query.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain ready for interaction.
    """
    loader = PyPDFLoader(file)
    documents = loader.load()
    if not documents:
        raise ValueError("No documents loaded from PDF")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    if not docs:
        raise ValueError("Document splitting resulted in no documents")

    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOllama(model="llama3.1:8b", temperature=0.7),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

# Streamlit App

def generate_pdf_from_youtube(url):
    output_path = "temp_audio"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    m4a_file = download_audio(url, output_path)
    wav_file = convert_audio_to_wav(m4a_file)
    transcript_text = transcribe_audio_whisper(wav_file)
    
    summary_text = summarize_text_with_chatollama(transcript_text)
    
    video_info = f"Title: {st.session_state.video_title}\n\nSummary: {summary_text}\n\n"
    combined_text = video_info + transcript_text
    
    pdf_file = "transcript.pdf"
    save_text_to_pdf(combined_text, pdf_file)
    
    st.session_state.loaded_file = pdf_file
    st.session_state.qa = load_db(pdf_file, "stuff", 6)
    st.session_state.chat_history = []
    
    st.success("Video Loaded Successfully")

def convchain(query):
    """
    Process a user's query through the conversational retrieval chain without using chat history.

    Args:
        query (str): The user's question or query.
    """
    if not query:
        st.write("Please enter a query.")
        return
    if 'qa' not in st.session_state or st.session_state.qa is None:
        st.write("No PDF loaded. Please provide a YouTube URL and generate the PDF first.")
        return
    
    # Pass only the current query without chat history
    result = st.session_state.qa({"question": query, "chat_history": []})
    
    st.session_state.chat_history.extend([(query, result["answer"])])
    st.session_state.db_query = result["generated_question"]
    st.session_state.db_response = result["source_documents"]
    st.session_state.answer = result['answer']
    
    st.write(f'User: {query}')
    st.write(f'ChatBot: {st.session_state.answer}')


def clr_history():
    """
    Clear the chat history stored in the session state.
    """
    st.session_state.chat_history = []

# Initialize session state variables if not already initialized
if 'loaded_file' not in st.session_state:
    st.session_state.loaded_file = None
if 'qa' not in st.session_state:
    st.session_state.qa = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'db_query' not in st.session_state:
    st.session_state.db_query = ""
if 'db_response' not in st.session_state:
    st.session_state.db_response = []
if 'answer' not in st.session_state:
    st.session_state.answer = ""
if 'video_title' not in st.session_state:
    st.session_state.video_title = ""

# Set up the Streamlit app configuration
st.set_page_config(
    page_title="ChatTube",
    page_icon="▶️"
)

# Display the app title with custom HTML formatting
st.markdown('<h1> <span style="color:red;">Chat</span>Tube <span style="color:red;">▶️</span></h1>', unsafe_allow_html=True)

# Input field for YouTube URL and button to start processing
url = st.text_input("Enter YouTube URL")
if st.button('Upload'):
    generate_pdf_from_youtube(url)
if st.session_state.video_title != "":
    video_title_html = f'<span style="color:red;font-weight:bold">Video Title:</span> {st.session_state.video_title}'
    st.markdown(video_title_html, unsafe_allow_html=True)

# If a PDF has been loaded, allow user to enter a query and generate an answer
if st.session_state.loaded_file:
    query = st.text_input("Enter your question here:")
    if st.button('Generate Answer'):
        convchain(query)

# Display chat history using custom HTML for formatting
if st.session_state.chat_history:
    st.header("Chat History")
    chat_history_html = '''
    <style>
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ccc;
        }
        .chat-container .message {
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            white-space: pre-wrap;
            font-family: 'Arial', sans-serif;  /* Use Streamlit's default font */
            font-size: 16px;  /* Adjust font size as needed */
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
        .chat-container .user {
            background: rgba(104, 109, 118,0.2);
        }
        .chat-container .assistant {
            background: rgb(55, 58, 64);
        }
        .chat-container .user::before {
            content: "User: ";
            font-weight: bold;
            color: red;
        }
        .chat-container .assistant::before {
            content: "Chatbot: ";
            font-weight: bold;
            color: red;
        }
    </style>
    <div class="chat-container">
    '''
    
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for i, (query, response) in enumerate(st.session_state.chat_history):
            user_message = markdown(f"{query}")
            bot_message = markdown(f"{response}")
            chat_history_html += f'<div class="message user">{user_message}</div>'
            chat_history_html += f'<div class="message assistant">{bot_message}</div>'
    chat_history_html += '</div>'
    components.html(chat_history_html, height=400)