import streamlit as st
from utils import *
import openai
import os

st.title("💬 FTMM-QA") 
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

saved_contexts = list()
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Retrieval
    contexts = [c for c in docsearch.similarity_search(prompt, k=2) if c not in saved_contexts]
    saved_contexts = contexts
    with st.chat_message("assistant"):
        # Send to LLM
        stream_handler = StreamHandler(st.empty())
        # LLM
        chat_model = ChatOpenAI(api_key=st.secrets["openai_key"], streaming=True, callbacks=[stream_handler])
        chain = prompt_template | chat_model
        response = chain.invoke({"context": contexts, "question":prompt, "chat_history": memory.buffer_as_messages})
        # st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response.content})
    memory.save_context({"input": template.format(context=contexts, question=prompt)}, 
                        {"output": response.content})
    # st.chat_message("assistant").write(response.content)