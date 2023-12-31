from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.memory import ConversationBufferWindowMemory
import chromadb
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Template prompt
template = """Jawablah pertanyaan di bawah ini berdasarkan konteks yang diberikan! \
Jika dalam pertanyaan merujuk ke histori chat sebelumnya, maka gunakan konteks dari pertanyaan \
sebelumnya untuk menjawab!
Konteks:
{context}

Pertanyaan:
{question}
"""

template_system = """Namamu adalah Iris, sebuah chatbot Fakultas Teknologi Maju dan \
Multidisiplin (FTMM), Universitas Airlangga. Kamu siap menjawab pertanyaan apapun \
seputar FTMM. Kamu menjawab setiap pertanyaan dengan ceria, sopan, dan asik!
"""

# Prompt
prompt_template = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(template_system),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(template),
    ]
)

# Memory
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=1)

# Embeddings
model = "Huggingface"
if model == "OpenAI":
    embed = OpenAIEmbeddings(api_key=st.secrets["openai_key"])
elif model == "Gemini":
    embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyA_mBeqs078kUm2ZUL_ontJa8lfd0LxVng")
elif model == "Huggingface":
    embed = HuggingFaceInferenceAPIEmbeddings(
        api_key=st.secrets["hf_key"], model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

# Vector store
client = chromadb.PersistentClient(path="database")
collection_ftmm = client.get_or_create_collection(name="ftmm")
docsearch = Chroma(persist_directory="database", embedding_function=embed, collection_name="ftmm")