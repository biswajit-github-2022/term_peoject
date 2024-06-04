import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class PDFChatBot:

    def __init__(self):
        self.data_path = os.path.join('data')
        self.db_faiss_path = os.path.join('vectordb', 'db_faiss')

    def create_vector_db(self):
        '''Function to create vector DB provided the PDF files'''
        if not os.path.exists(self.db_faiss_path):
            os.makedirs(self.db_faiss_path)

        loader = DirectoryLoader(self.data_path, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(self.db_faiss_path)
        print(f"Vector database created and saved at {self.db_faiss_path}")

    def load_llm(self):
        # Load the locally downloaded model here
        llm = CTransformers(
            model="./models/llama-2-7b-chat.ggmlv3.q8_0.bin",
            model_type="llama",
            max_new_tokens=2000,
            temperature=0.5
        )
        return llm

    def conversational_chain(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(self.db_faiss_path, embeddings, allow_dangerous_deserialization=True)
        
        # initializing the conversational chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.load_llm(),
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            verbose=True,
            memory=memory
        )
        system_template = """Only answer questions related to the following pieces of text.\n- Strictly not answer the question if user asked question is not present in the below text.
        Take note of the sources and include them in the answer in the format: "\nSOURCES: source1 \nsource2", use "SOURCES" in capital letters regardless of the number of sources.
        If you don't know the answer, just say that "I don't know", don't try to make up an answer.
        ----------------
        {summaries}"""
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)

        chain_type_kwargs = {"prompt": prompt}        
        conversational_chain1 = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.load_llm(),
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )        

        return conversational_chain

def initialize_chain():
    bot = PDFChatBot()
    bot.create_vector_db()  # Ensure the vector DB is created
    conversational_chain = bot.conversational_chain()
    return conversational_chain

chat_history = []

chain = initialize_chain()

print("Question will be asked now")
query = "who is BALRAM HALWAI"
result = chain({"question": query})

print("See the answer")
print(result['answer'])

print("See detailed answer")
print(result)

