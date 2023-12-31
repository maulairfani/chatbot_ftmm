from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
import chromadb
import argparse
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
import google.generativeai as genai
genai.configure(api_key=st.secrets["google_api_key"])
import pandas as pd

# Input
parser = argparse.ArgumentParser()
parser.add_argument("--embeddings")
args = parser.parse_args()

# Load knowledge Documents
print("Load & Process Documents...")
loader = DirectoryLoader("knowledges")
docs = loader.load()

# Split into chunks
text_splitter = TokenTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 200
)
chunks = text_splitter.split_documents(docs)

# Selain embeddings
metadata = [chunks[i].metadata for i in range(len(chunks))]
documents = [chunks[i].page_content for i in range(len(chunks))]
ids = [f"doc{i}" for i in range(len(chunks))]
print(f"Found {len(documents)} chunks of documents...")

# define persistent client and embeddings method 
client = chromadb.PersistentClient(path="knowledge_database")
embeddings_method = args.embeddings

if embeddings_method == "OpenAI":
    collection_ftmm = client.get_or_create_collection(name="ftmm")
    embed = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])
    print("Generating OpenAI (Ada) Embeddings...")
    embeddings = embed.embed_documents(documents)
    
elif embeddings_method == "Gemini":
    collection_ftmm = client.get_or_create_collection(name="gemini_ftmm")
    df = pd.DataFrame(documents)
    df.columns = ['Text']

    def embed_fn(text):
        return genai.embed_content(model="models/embedding-001",
                                    content=text,
                                    task_type="retrieval_document")["embedding"]
    print("Generating Gemini Embeddings...")
    df['Embeddings'] = df['Text'].apply(embed_fn)
    embeddings = df["Embeddings"].to_list()

elif embeddings_method == "Huggingface":
    collection_ftmm = client.get_or_create_collection(name="hf_ftmm")
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    print("Generating HuggingFace (paraphrase-multilingual-mpnet-base-v2) Embeddings...")
    embeddings = embed.embed_documents(documents)

# Add to database
print("Saving to database...")
collection_ftmm.add(
    ids=ids,
    documents=[c.page_content for c in chunks],
    embeddings=embeddings,
    metadatas=metadata
)
print("Saved to folder database✅")


