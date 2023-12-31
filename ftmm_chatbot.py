from utils_cmd import *

class FTMMChatbot:
    def __init__(self):
        self.halo = "halo!"

    def build(self, embedding_method, LLM):
        if embedding_method == "1":
            embed = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])
            docsearch = Chroma(persist_directory="knowledge_database", collection_name="ftmm", embedding_function=embed)

        elif embedding_method == "2":
            docsearch = Chroma(persist_directory="knowledge_database", collection_name="gemini_ftmm")

        elif embedding_method == "3":
            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
            docsearch = Chroma(persist_directory="knowledge_database", collection_name="hf_ftmm", embedding_function=embed)
        
        if LLM == "1":
            chat_model = ChatOpenAI(model = "gpt-3.5-turbo-16k", 
                                    temperature=0, openai_api_key=st.secrets["openai_key"])
        elif LLM == "2":
            chat_model = ChatGoogleGenerativeAI(
                model="gemini-pro", 
                temperature=0, 
                google_api_key=st.secrets["google_api_key"],
                convert_system_message_to_human=True)
            
        chain = prompt_template | chat_model

        return docsearch, chain

    def return_context(self, docsearch, search_method, prompt, k):
        if docsearch._collection.name == "ftmm" or docsearch._collection.name == "hf_ftmm":
            if search_method == "1":
                contexts = docsearch.similarity_search(prompt, k=k)
            elif search_method == "2":
                contexts = docsearch.max_marginal_relevance(prompt, k=k)
        elif docsearch._collection.name == "gemini_ftmm":
            embeddings = embed_fn(prompt)
            if search_method == "1":
                contexts = docsearch.similarity_search_by_vector(embeddings, k=k)
            elif search_method == "2":
                contexts = docsearch.max_marginal_relevance_by_vector(embeddings, k=k)
        return contexts
    
    def respond(self, docsearch, chain, search_method, prompt):
        contexts = chatbot.return_context(docsearch, search_method, prompt, k=5)
        response = chain.stream({"context": contexts, "question":prompt, "chat_history": memory.buffer_as_messages})
        return response


if __name__ == "__main__":
    chatbot = FTMMChatbot()

    while True:
        search_method = input("Pilih Searching Method [1] Similarity Search, [2] MMR Search \n-> ")
        if search_method == "1" or search_method == "2":
            break
        else:
            pass

    while True:
        embedding_method = input("Pilih Embeddings [1] OpenAI Ada, [2] Google Embedding-001, [3] MPNet-Multilingual \n-> ")
        if embedding_method in ["1", "2", "3"]:
            break
        else:
            pass

    while True:
        LLM = input("Pilih LLM [1] OpenAI, [2] Google Gemini \n-> ")
        if LLM in ["1", "2"]:
            break
        else:
            pass

    docsearch, chain = chatbot.build(embedding_method, LLM)
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Selamat datang di FTMMQA!")
    while True:
        prompt = input("🥸 : ")
        
        if prompt.lower() == 'keluar':
            print("Terima kasih sudah menggunakan FTMMQA. Sampai jumpa!")
            break
        
        print("🤖 :", end=" ")
        full_response = ""
        for r in chatbot.respond(docsearch, chain, search_method, prompt):
            print(r.content, end="", flush=True)
            full_response += r.content
        
        contexts = chatbot.return_context(docsearch, search_method, prompt, k=5)
        memory.save_context({"input": template.format(context=contexts, question=prompt)}, {"output": full_response})
        print("\n")